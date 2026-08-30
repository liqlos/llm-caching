"""Fail-closed composition of exact-prompt lookup and decode-only receipts."""

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


def _receipt(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to load economics receipt: {path}") from error
    if not isinstance(value, dict):
        raise ContractError("economics receipt must be an object")
    return value


def _require_status(receipt: Mapping[str, Any], status: str, *, name: str, evidence: str) -> Mapping[str, Any]:
    if receipt.get("status") != status or receipt.get("evidence_class") != evidence:
        raise ContractError(f"{name} receipt has an incompatible status or evidence class")
    payload = receipt.get("payload")
    if not isinstance(payload, dict):
        raise ContractError(f"{name} receipt lacks a payload")
    return payload


def _derive(live: Mapping[str, Any], lookup: Mapping[str, Any], build: Mapping[str, Any]) -> dict[str, Any]:
    live_payload = _require_status(
        live, "CONTROLLED_LIVE_DECODE_POSITIVE", name="live", evidence="LOCAL_MEASURED"
    )
    lookup_payload = _require_status(
        lookup, "EXACT_PROMPT_LOOKUP_PROFILE_VALID", name="lookup", evidence="LOCAL_MEASURED"
    )
    build_payload = _require_status(build, "EXACT_PROMPT_ARCHIVE_BUILT", name="archive build", evidence="SIMULATOR")
    live_inputs = live.get("input_fingerprints")
    lookup_inputs = lookup.get("input_fingerprints")
    if not isinstance(live_inputs, dict) or not isinstance(lookup_inputs, dict):
        raise ContractError("economics receipts lack input fingerprints")
    for field in ("manifest", "trace"):
        if live_inputs.get(field) != lookup_inputs.get(field):
            raise ContractError(f"live and lookup receipts disagree on {field}")
    if lookup_inputs.get("archive_manifest") != build_payload.get("archive_manifest_sha256"):
        raise ContractError("lookup receipt is not bound to the built archive")
    lookup_ci = lookup_payload.get("net_lookup_and_cursor_ci_ns_per_call")
    if not isinstance(lookup_ci, dict):
        raise ContractError("lookup receipt lacks a net lookup confidence interval")
    lookup_upper = lookup_ci.get("upper")
    if isinstance(lookup_upper, bool) or not isinstance(lookup_upper, (int, float)) or lookup_upper < 0:
        raise ContractError("lookup confidence upper bound is invalid")
    cases = live_payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("live receipt lacks cases")
    ratios_by_request: dict[str, list[float]] = {}
    total_lookup_ns = 0.0
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("response"), dict):
            raise ContractError("live receipt case is invalid")
        response = case["response"]
        if response.get("exact_output_and_state") is not True:
            raise ContractError("live receipt contains a non-exact output/state case")
        request_id = case.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ContractError("live receipt case lacks a request id")
        rows = response.get("measurements")
        if not isinstance(rows, list) or not rows:
            raise ContractError("live receipt case lacks measurements")
        ratios: list[float] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ContractError("live measurement row is invalid")
            for field in ("serial_before_ns", "serial_after_ns", "candidate_decode_ns", "verification_rounds"):
                require_int(row.get(field), f"live measurement {field}", minimum=0)
            if row["candidate_decode_ns"] <= 0:
                raise ContractError("live candidate duration must be positive")
            adjusted_candidate = row["candidate_decode_ns"] + row["verification_rounds"] * float(lookup_upper)
            paired_serial = (row["serial_before_ns"] + row["serial_after_ns"]) / 2.0
            ratios.append(paired_serial / adjusted_candidate)
            total_lookup_ns += row["verification_rounds"] * float(lookup_upper)
        ratios_by_request[request_id] = ratios
    confidence = clustered_bootstrap_ci(ratios_by_request, bootstrap_replicates=2_000, seed=41)
    return {
        "lookup_upper_ns_per_verification_round": lookup_upper,
        "worst_case_lookup_assumption": "one full sealed lookup plus cursor read at the lookup 95% upper bound for every target verification round/v1",
        "added_lookup_ns_across_all_measurements": total_lookup_ns,
        "lookup_inclusive_paired_serial_over_candidate_ratio": asdict(confidence),
        "decision": "COMPOSED_EXACT_PROMPT_DECODE_POSITIVE" if confidence.lower > 1.0 else "COMPOSED_EXACT_PROMPT_DECODE_INCONCLUSIVE",
    }


def run_exact_prompt_economics(
    *,
    live_path: Path,
    lookup_path: Path,
    build_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    live, lookup, build = _receipt(live_path), _receipt(lookup_path), _receipt(build_path)
    analysis = _derive(live, lookup, build)
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.exact_prompt_economics", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"live": file_sha256(live_path), "lookup": file_sha256(lookup_path), "archive_build": file_sha256(build_path)},
        status=analysis["decision"],
        payload={
            "kind": "composed_exact_prompt_lookup_inclusive_decode_bound",
            "analysis": analysis,
            "not_claimed": [
                "integrated_provider_speedup",
                "end_to_end_request_speedup",
                "representative_workload_speedup",
                "thermal_sustainability",
            ],
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS composed exact-prompt decode bound")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--archive-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_exact_prompt_economics(
        live_path=parsed.live,
        lookup_path=parsed.lookup,
        build_path=parsed.archive_build,
        output_path=parsed.output,
        report_path=parsed.report,
        repository_root=parsed.repository_root,
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
