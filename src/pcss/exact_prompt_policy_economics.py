"""Fail-closed mixed exact-hit/edited-miss policy composition.

This combines independently paired serial controls from immutable receipts. It
models one archive lookup at the miss request boundary and never changes target
output or attaches a provider to llama.cpp.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import ContractError, require_int
from .metrics import clustered_bootstrap_ci
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)


def _load(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to load policy economics receipt: {path}") from error
    if not isinstance(value, dict):
        raise ContractError("policy economics receipt must be an object")
    return value


def _payload(receipt: Mapping[str, Any], *, status: str, name: str) -> Mapping[str, Any]:
    if receipt.get("status") != status or receipt.get("evidence_class") != "LOCAL_MEASURED":
        raise ContractError(f"{name} receipt has incompatible status or evidence")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        raise ContractError(f"{name} receipt lacks a payload")
    return payload


def _inputs(receipt: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = receipt.get("input_fingerprints")
    if not isinstance(value, dict):
        raise ContractError(f"{name} receipt lacks input fingerprints")
    return value


def _serial(row: Mapping[str, Any]) -> float:
    before = require_int(row.get("serial_before_ns"), "measurement.serial_before_ns", minimum=1)
    after = require_int(row.get("serial_after_ns"), "measurement.serial_after_ns", minimum=1)
    return (before + after) / 2.0


def _derive(hit: Mapping[str, Any], edited: Mapping[str, Any], miss: Mapping[str, Any]) -> dict[str, Any]:
    hit_payload = _payload(hit, status="EXACT_PROMPT_ARCHIVE_HARNESS_POSITIVE", name="hit harness")
    edited_payload = _payload(edited, status="CONTROLLED_LIVE_DECODE_INCONCLUSIVE", name="edited control")
    miss_payload = _payload(miss, status="EXACT_PROMPT_LOOKUP_PROFILE_VALID", name="miss profile")
    hit_inputs = _inputs(hit, "hit harness")
    edited_inputs = _inputs(edited, "edited control")
    miss_inputs = _inputs(miss, "miss profile")
    for field in ("manifest", "config", "bridge_binary", "model_artifact", "llama_cpp_library"):
        if hit_inputs.get(field) != edited_inputs.get(field) or hit_inputs.get(field) is None:
            raise ContractError(f"hit harness and edited control disagree on {field}")
    if hit_inputs.get("manifest") != miss_inputs.get("manifest"):
        raise ContractError("hit harness and miss profile disagree on manifest")
    if miss_payload.get("expect_hit") is not False:
        raise ContractError("policy economics requires an archive miss profile")
    if hit_inputs.get("archive_manifest") != miss_inputs.get("archive_manifest"):
        raise ContractError("hit harness and miss profile disagree on archive seal")
    miss_ci = miss_payload.get("net_lookup_and_cursor_ci_ns_per_call")
    if not isinstance(miss_ci, dict):
        raise ContractError("miss profile lacks lookup confidence interval")
    miss_upper = miss_ci.get("upper")
    if isinstance(miss_upper, bool) or not isinstance(miss_upper, (int, float)) or miss_upper < 0:
        raise ContractError("miss lookup upper bound is invalid")
    ratios_by_request: dict[str, list[float]] = {}
    hit_rows = 0
    miss_rows = 0
    for case in hit_payload.get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("response"), dict):
            raise ContractError("hit harness case is invalid")
        response = case["response"]
        archive_rows = case.get("archive_measurements")
        request_id = case.get("request_id")
        if response.get("exact_output_and_state") is not True or not isinstance(archive_rows, list):
            raise ContractError("hit harness case lacks exact archive measurements")
        measurements = response.get("measurements")
        if not isinstance(request_id, str) or not isinstance(measurements, list) or len(measurements) != len(archive_rows):
            raise ContractError("hit harness measurement alignment is invalid")
        ratios: list[float] = []
        for bridge_row, archive_row in zip(measurements, archive_rows, strict=True):
            combined = archive_row.get("lookup_inclusive_candidate_ns") if isinstance(archive_row, dict) else None
            if isinstance(combined, bool) or not isinstance(combined, (int, float)) or combined <= 0:
                raise ContractError("hit harness combined candidate duration is invalid")
            ratios.append(_serial(bridge_row) / combined)
            hit_rows += 1
        ratios_by_request[request_id] = ratios
    for case in edited_payload.get("cases", []):
        if not isinstance(case, dict) or not isinstance(case.get("response"), dict):
            raise ContractError("edited control case is invalid")
        response = case["response"]
        request_id = case.get("request_id")
        rows = response.get("measurements")
        if response.get("exact_output_and_state") is not True or not isinstance(request_id, str) or not isinstance(rows, list):
            raise ContractError("edited control lacks exact serial measurements")
        ratios: list[float] = []
        for row in rows:
            serial = _serial(row)
            ratios.append(serial / (serial + float(miss_upper)))
            miss_rows += 1
        ratios_by_request[request_id] = ratios
    if not ratios_by_request or hit_rows == 0 or miss_rows == 0:
        raise ContractError("policy economics requires non-empty exact-hit and edited-miss clusters")
    interval = clustered_bootstrap_ci(ratios_by_request, bootstrap_replicates=2_000, seed=53)
    return {
        "synthetic_mix": {"exact_hit_request_clusters": len(hit_payload["cases"]), "edited_miss_request_clusters": len(edited_payload["cases"])},
        "exact_hit_measurement_rows": hit_rows,
        "edited_miss_measurement_rows": miss_rows,
        "miss_lookup_upper_ns_per_request": miss_upper,
        "miss_assumption": "one exact-prompt archive miss lookup at the miss request boundary/v1",
        "paired_serial_over_policy_ratio": asdict(interval),
        "decision": "MIXED_EXACT_PROMPT_POLICY_POSITIVE" if interval.lower > 1.0 else "MIXED_EXACT_PROMPT_POLICY_INCONCLUSIVE",
    }


def run_policy_economics(
    *,
    hit_path: Path,
    edited_path: Path,
    miss_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    hit, edited, miss = _load(hit_path), _load(edited_path), _load(miss_path)
    analysis = _derive(hit, edited, miss)
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.exact_prompt_policy_economics", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"hit_harness": file_sha256(hit_path), "edited_control": file_sha256(edited_path), "miss_profile": file_sha256(miss_path)},
        status=analysis["decision"],
        payload={
            "kind": "synthetic_exact_prompt_hit_miss_policy_bound",
            "analysis": analysis,
            "not_claimed": ["representative_traffic_mix", "integrated_provider_speedup", "end_to_end_request_speedup", "thermal_sustainability"],
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS mixed exact-prompt policy bound")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hit-harness", type=Path, required=True)
    parser.add_argument("--edited-control", type=Path, required=True)
    parser.add_argument("--miss-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_policy_economics(
        hit_path=parsed.hit_harness, edited_path=parsed.edited_control, miss_path=parsed.miss_profile,
        output_path=parsed.output, report_path=parsed.report, repository_root=parsed.repository_root,
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
