"""External archive drafts verified in one continuous native pending trajectory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_int
from .exact_prompt_archive import open_exact_prompt_archive
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config
from .llamacpp_live_benchmark import _pair
from .manifest import validate_manifest
from .offline_runner import load_trace_fixture
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .verifier import TargetHostIdentity
from .verifier_runner import observe_target_host


@dataclass(frozen=True)
class ExternalDraftFault:
    """One deliberate external-draft mismatch in an otherwise exact archive stream."""

    round_index: int
    rejection_position: int


def _request(config: BridgeConfig, target: Mapping[str, Any], drafts: Sequence[Sequence[int]], *, k: int) -> dict[str, Any]:
    return {
        "schema_version": "pcss.llama-bridge-request/v1",
        "operation": "trajectory_pending",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "seed": {"seed_id": target["record_id"], "initial_context_tokens": list(target["content"]["prompt_token_ids"])},
        "trajectory": {
            "case_id": f"{target['record_id']}:external-archive", "k": k,
            "rounds": [{"draft_tokens": list(draft)} for draft in drafts],
        },
    }


def _run_bridge(config: BridgeConfig, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-trajectory-pending"), input=canonical_json(request) + "\n",
            capture_output=True, text=True, check=False, timeout=config.request_timeout_seconds,
        )
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ContractError("external trajectory bridge did not return one JSON response") from error
    if completed.returncode != 0 or not isinstance(response, dict):
        raise ContractError("external trajectory bridge failed")
    return response


def _build_drafts(
    tokens: Sequence[int], *, k: int, output_tokens: int, fault: ExternalDraftFault | None = None
) -> tuple[tuple[int, ...], ...]:
    if output_tokens % (k + 1) != 0:
        raise ContractError("external trajectory output boundary cannot form complete K-token rounds")
    round_count = output_tokens // (k + 1)
    if fault is not None and (
        fault.round_index < 0
        or fault.round_index >= round_count
        or fault.rejection_position < 0
        or fault.rejection_position >= k
    ):
        raise ContractError("external trajectory fault is outside the requested K-token rounds")
    cursor = 0
    drafts: list[tuple[int, ...]] = []
    for round_index in range(round_count):
        draft = tuple(tokens[cursor : cursor + k])
        if len(draft) != k:
            raise ContractError("external trajectory archive cursor is shorter than target output")
        if fault is not None and round_index == fault.round_index:
            altered = list(draft)
            original = altered[fault.rejection_position]
            altered[fault.rejection_position] = 0 if original != 0 else 1
            drafts.append(tuple(altered))
            cursor += fault.rejection_position + 1
        else:
            drafts.append(draft)
            cursor += k + 1
    return tuple(drafts)


def _drafts(
    archive,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    k: int,
    output_tokens: int,
    fault: ExternalDraftFault | None = None,
) -> tuple[tuple[int, ...], ...]:
    hits = archive.lookup(tuple(target["content"]["prompt_token_ids"]))
    if len(hits) != 1 or hits[0].record_id != source["record_id"]:
        raise ContractError("external trajectory archive lookup is absent, ambiguous, or has wrong provenance")
    tokens = hits[0].cursor.read(output_tokens)
    if len(tokens) != output_tokens:
        raise ContractError("external trajectory archive cursor is shorter than target output")
    return _build_drafts(tokens, k=k, output_tokens=output_tokens, fault=fault)


def _validate(response: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any], drafts: Sequence[Sequence[int]], *, k: int, output_tokens: int) -> dict[str, Any]:
    if response.get("operation") != "trajectory_pending" or response.get("status") != "ok":
        raise ContractError("external trajectory native result is not successful")
    if response.get("all_rounds_match") is not True or response.get("final_state_matches") is not True or response.get("continuation_matches") is not True:
        raise ContractError("external trajectory native state/continuation comparison failed")
    rounds = response.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != len(drafts):
        raise ContractError("external trajectory response round count differs from archive drafts")
    source_tokens = source["content"]["committed_token_ids"]
    committed: list[int] = []
    normalized_rounds: list[dict[str, Any]] = []
    source_cursor = 0
    for index, (round_response, draft) in enumerate(zip(rounds, drafts, strict=True)):
        if not isinstance(round_response, dict):
            raise ContractError("external trajectory round must be an object")
        if (
            round_response.get("mode") != "external_draft"
            or round_response.get("verification_targets_match") is not True
            or round_response.get("state_matches") is not True
            or round_response.get("tail_trimmed") is not True
        ):
            raise ContractError("external trajectory round did not preserve external draft state")
        accepted = require_int(round_response.get("accepted_draft_tokens"), "accepted_draft_tokens", minimum=0)
        if accepted > k:
            raise ContractError("external trajectory accepted more than K draft tokens")
        expected_window = source_tokens[source_cursor : source_cursor + k + 1]
        if len(expected_window) != k + 1:
            raise ContractError("external trajectory source is shorter than its requested rounds")
        expected_accepted = 0
        while expected_accepted < k and draft[expected_accepted] == expected_window[expected_accepted]:
            expected_accepted += 1
        if accepted != expected_accepted:
            raise ContractError("external trajectory native accepted prefix differs from supplied draft")
        expected_rejection = -1 if accepted == k else accepted
        if round_response.get("rejection_position") != expected_rejection:
            raise ContractError("external trajectory rejection position differs from supplied draft")
        pending = require_int(round_response.get("pending_token"), "pending_token", minimum=0)
        expected_pending = expected_window[k] if accepted == k else expected_window[accepted]
        if pending != expected_pending:
            raise ContractError("external trajectory pending token differs from archive cursor")
        committed.extend(draft[:accepted])
        committed.append(pending)
        normalized_rounds.append({
            "round": index, "draft_tokens": list(draft), "accepted_draft_tokens": accepted,
            "rejection_position": expected_rejection,
            "pending_kind": "bonus" if accepted == k else "correction", "pending_token": pending,
            "state_sha256": round_response.get("verifier_state_sha256"), "state_bytes": round_response.get("state_bytes"),
        })
        source_cursor += accepted + 1
    expected = target["content"]["committed_token_ids"][:source_cursor]
    if committed != expected:
        raise ContractError("external continuous trajectory output differs from recorded target")
    return {"rounds": normalized_rounds, "output_tokens": committed, "exact_output": True}


def run_external_trajectory(
    *,
    config_path: Path,
    manifest_path: Path,
    trace_path: Path,
    archive_root: Path,
    pairs: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    k: int = 2,
    output_tokens: int = 48,
    fault: ExternalDraftFault | None = None,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
) -> ResultReceipt:
    if k < 1 or output_tokens < k + 1 or output_tokens % (k + 1) != 0 or not pairs or len(set(pairs)) != len(pairs):
        raise ContractError("external trajectory requires output divisible by K+1 and unique pairs")
    config = load_bridge_config(config_path)
    records = load_trace_fixture(trace_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    host = target_observer()
    if manifest["evidence_class"] != "LOCAL_MEASURED" or manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("external trajectory requires matching LOCAL_MEASURED target manifest")
    model_hash = file_sha256(config.model_artifact)
    bridge_hash = file_sha256(config.bridge_executable)
    binding = manifest["verifier_gate"]["adapter_binding"]
    if manifest["model_artifact"]["sha256"] != model_hash or binding["model_artifact_sha256"] != model_hash:
        raise ContractError("external trajectory model differs from observed manifest")
    if binding["backend_binary_sha256"] != bridge_hash:
        raise ContractError("external trajectory bridge differs from observed manifest")
    if binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint:
        raise ContractError("external trajectory runtime differs from observed manifest")
    archive = open_exact_prompt_archive(archive_root)
    cases = []
    for pair_spec in pairs:
        source, target = _pair(records, pair_spec, pair_mode="exact_repeat")
        drafts = _drafts(archive, source, target, k=k, output_tokens=output_tokens, fault=fault)
        response = bridge_runner(config, _request(config, target, drafts, k=k))
        verified = _validate(response, source, target, drafts, k=k, output_tokens=output_tokens)
        cases.append({"source_record_id": source["record_id"], "target_record_id": target["record_id"], "request_id": target["request_id"], **verified})
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.exact_prompt_external_trajectory", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "manifest": file_sha256(manifest_path), "trace": file_sha256(trace_path),
            "archive_manifest": file_sha256(archive_root / "archive-manifest.json"),
            "config": file_sha256(config.config_path), "bridge_binary": file_sha256(config.bridge_executable),
            "model_artifact": file_sha256(config.model_artifact), "llama_cpp_library": file_sha256(config.llama_cpp_library),
        },
        status="EXTERNAL_ARCHIVE_TRAJECTORY_VERIFIED",
        payload={
            "kind": "sealed_archive_external_drafts_continuous_native_trajectory",
            "host_identity": host.as_dict(), "k": k,
            "external_draft_fault": None if fault is None else {
                "round_index": fault.round_index, "rejection_position": fault.rejection_position,
            },
            "case_count": len(cases), "cases": cases,
            "round_count": sum(len(case["rounds"]) for case in cases),
            "runtime_claim": "external archive drafts in continuous isolated native trajectory; no provider hook, throughput, or production integration claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS external archive continuous trajectory")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=48)
    parser.add_argument("--reject-round", type=int)
    parser.add_argument("--reject-position", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    if (parsed.reject_round is None) != (parsed.reject_position is None):
        raise ContractError("--reject-round and --reject-position must be supplied together")
    fault = None if parsed.reject_round is None else ExternalDraftFault(parsed.reject_round, parsed.reject_position)
    run_external_trajectory(
        config_path=parsed.config, manifest_path=parsed.manifest, trace_path=parsed.trace,
        archive_root=parsed.archive_root, pairs=parsed.pair, output_path=parsed.output,
        report_path=parsed.report, repository_root=parsed.repository_root, command_arguments=arguments,
        k=parsed.k, output_tokens=parsed.output_tokens, fault=fault,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
