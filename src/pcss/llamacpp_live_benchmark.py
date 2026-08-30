"""Controlled target decode benchmark for the exact pending-token protocol.

Only exact cross-session trace repeats are accepted.  Provider lookup, prompt
prefill, and state restore are deliberately outside the measured region, so the
receipt is a bounded best-case decode result rather than a general PCSS claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, sha256
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config
from .manifest import validate_manifest
from .metrics import clustered_bootstrap_ci
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


REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
RESPONSE_SCHEMA_VERSION = "pcss.llama-bridge-pending-benchmark/v2"
_RESPONSE_FIELDS = {
    "schema_version", "operation", "status", "backend", "case_id", "seed_id", "k",
    "output_tokens", "warmup_repetitions", "measurement_repetitions", "authoritative_tokens",
    "reanchor_min_tokens",
    "candidate_prefix_tokens", "final_state_sha256", "final_state_bytes", "measurements",
    "measurement_clock", "measurement_scope", "provider_integration", "exact_output_and_state",
    "configuration", "observed_context", "state_export", "state_api", "generation_mode",
    "greedy_tie_break", "tokenization_boundary", "chat_template_applied", "thinking_mode_applied",
    "claim_boundary",
}
_MEASUREMENT_FIELDS = {
    "iteration", "serial_before_ns", "candidate_decode_ns", "serial_after_ns", "proposed_tokens",
    "accepted_tokens", "verification_rounds", "serial_fallback_steps",
    "state_checkpoint_rounds", "serial_state_reconstruction_rounds",
    "reanchor_attempts", "reanchor_successes",
}
_FINGERPRINT_FIELDS = (
    "model_fingerprint", "tokenizer_fingerprint", "chat_template_fingerprint",
    "generation_protocol_fingerprint",
)


def _record_by_id(records: Sequence[Mapping[str, Any]], record_id: str) -> Mapping[str, Any]:
    matches = [record for record in records if record["record_id"] == record_id]
    if len(matches) != 1:
        raise ContractError(f"trace must contain exactly one record_id {record_id!r}")
    return matches[0]


def _pair(
    records: Sequence[Mapping[str, Any]], pair_spec: str, *, pair_mode: str = "exact_repeat"
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    parts = pair_spec.split(":")
    if len(parts) != 2 or not all(parts):
        raise ContractError("benchmark pair must be SOURCE_RECORD_ID:TARGET_RECORD_ID")
    source, target = (_record_by_id(records, item) for item in parts)
    if source["chronological_index"] >= target["chronological_index"]:
        raise ContractError("benchmark source must precede target chronologically")
    if source["session_id"] == target["session_id"]:
        raise ContractError("benchmark pair must cross session boundaries")
    if source["trusted_scope"]["repository_id"] != target["trusted_scope"]["repository_id"]:
        raise ContractError("benchmark pair must remain inside one repository")
    if any(source[field] != target[field] for field in _FINGERPRINT_FIELDS):
        raise ContractError("benchmark pair fingerprints are incompatible")
    if pair_mode not in {"exact_repeat", "source_reuse"}:
        raise ContractError("benchmark pair_mode must be exact_repeat or source_reuse")
    source_content, target_content = source["content"], target["content"]
    if pair_mode == "exact_repeat" and source_content["prompt_token_ids"] != target_content["prompt_token_ids"]:
        raise ContractError("controlled exact-repeat benchmark requires identical prompt tokens")
    if pair_mode == "exact_repeat" and source_content["committed_token_ids"] != target_content["committed_token_ids"]:
        raise ContractError("controlled exact-repeat benchmark requires identical committed tokens")
    if source_content["committed_token_status"] != "complete" or target_content["committed_token_status"] != "complete":
        raise ContractError("controlled benchmark requires complete committed token streams")
    return source, target


def _request(
    config: BridgeConfig,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    k: int,
    output_tokens: int,
    warmup_repetitions: int,
    measurement_repetitions: int,
    reanchor_min_tokens: int = 0,
    forced_rejection_position: int | None = None,
    candidate_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    prompt = tuple(target["content"]["prompt_token_ids"])
    candidate = list(source["content"]["committed_token_ids"] if candidate_tokens is None else candidate_tokens)
    if not prompt:
        raise ContractError("benchmark prompt token stream must not be empty")
    if k < 1 or k + 1 > config.runtime["n_batch"]:
        raise ContractError("benchmark k exceeds the pending-token batch boundary")
    if output_tokens < 2 or output_tokens > len(candidate):
        raise ContractError("benchmark output_tokens must be between two and the candidate length")
    if warmup_repetitions < 0 or measurement_repetitions < 2:
        raise ContractError("benchmark requires non-negative warmups and at least two measurements")
    if reanchor_min_tokens < 0 or reanchor_min_tokens > output_tokens:
        raise ContractError("benchmark reanchor_min_tokens is outside the output boundary")
    if forced_rejection_position is not None:
        if forced_rejection_position < 0 or forced_rejection_position >= output_tokens:
            raise ContractError("forced rejection position is outside the output boundary")
        original = candidate[forced_rejection_position]
        replacement = next((token for token in candidate if token != original), None)
        if replacement is None:
            raise ContractError("forced rejection requires at least two distinct candidate tokens")
        candidate[forced_rejection_position] = replacement
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "benchmark_pending",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "seed": {"seed_id": target["record_id"], "initial_context_tokens": list(prompt)},
        "benchmark": {
            "case_id": f"{source['record_id']}--{target['record_id']}",
            "candidate_tokens": list(candidate),
            "k": k,
            "output_tokens": output_tokens,
            "warmup_repetitions": warmup_repetitions,
            "measurement_repetitions": measurement_repetitions,
            "reanchor_min_tokens": reanchor_min_tokens,
        },
    }


def _run_bridge(config: BridgeConfig, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-benchmark-pending"),
            input=canonical_json(request) + "\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ContractError("pcss_llama_bridge benchmark did not return one JSON response") from error
    if completed.returncode != 0 or not isinstance(response, dict):
        raise ContractError("pcss_llama_bridge benchmark failed")
    return response


def _positive_int(value: object, field: str) -> int:
    return require_int(value, field, minimum=1)


def _validate_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    config: BridgeConfig,
    *,
    expected_authoritative_tokens: Sequence[int] | None = None,
) -> dict[str, Any]:
    require_exact_keys(response, _RESPONSE_FIELDS, "benchmark response")
    benchmark = request["benchmark"]
    expected = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "operation": "benchmark_pending",
        "status": "ok",
        "backend": "llama.cpp",
        "case_id": benchmark["case_id"],
        "seed_id": request["seed"]["seed_id"],
        "k": benchmark["k"],
        "output_tokens": benchmark["output_tokens"],
        "warmup_repetitions": benchmark["warmup_repetitions"],
        "measurement_repetitions": benchmark["measurement_repetitions"],
        "reanchor_min_tokens": benchmark["reanchor_min_tokens"],
        "candidate_prefix_tokens": benchmark["candidate_tokens"][: benchmark["output_tokens"]],
        "exact_output_and_state": True,
        "measurement_clock": "std::chrono::steady_clock/nanoseconds/v1",
        "measurement_scope": "decode-and-synchronize/prefill-restore-provider-excluded/v1",
        "provider_integration": "static-trace-candidate/provider-lookup-excluded/v1",
        "state_export": "model-architecture+memory-state/v2",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
        "claim_boundary": "controlled-static-candidate-live-decode/not-general-pcss/v1",
    }
    for field, value in expected.items():
        if response[field] != value:
            raise ContractError(f"benchmark response {field} differs from request contract")
    expected_tokens = list(
        benchmark["candidate_tokens"][: benchmark["output_tokens"]]
        if expected_authoritative_tokens is None
        else expected_authoritative_tokens[: benchmark["output_tokens"]]
    )
    if response["authoritative_tokens"] != expected_tokens:
        raise ContractError("benchmark target output differs from the recorded target trace")
    digest = response["final_state_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ContractError("benchmark final state digest is invalid")
    _positive_int(response["final_state_bytes"], "benchmark final_state_bytes")
    expected_configuration = {
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
    }
    if response["configuration"] != expected_configuration:
        raise ContractError("benchmark response configuration differs from configured runtime")
    observed = response["observed_context"]
    expected_observed = {
        "actual_n_ctx": config.runtime["n_ctx"], "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }
    if observed != expected_observed:
        raise ContractError("benchmark observed context differs from configured runtime")
    measurements = response["measurements"]
    if not isinstance(measurements, list) or len(measurements) != benchmark["measurement_repetitions"]:
        raise ContractError("benchmark response has the wrong measurement count")
    normalized = []
    for index, row in enumerate(measurements):
        if not isinstance(row, dict):
            raise ContractError("benchmark measurement must be an object")
        require_exact_keys(row, _MEASUREMENT_FIELDS, "benchmark measurement")
        if row["iteration"] != index:
            raise ContractError("benchmark measurement order is not deterministic")
        for field in ("serial_before_ns", "candidate_decode_ns", "serial_after_ns"):
            _positive_int(row[field], f"benchmark measurement.{field}")
        for field in (
            "proposed_tokens", "accepted_tokens", "verification_rounds", "serial_fallback_steps",
            "state_checkpoint_rounds", "serial_state_reconstruction_rounds",
            "reanchor_attempts", "reanchor_successes",
        ):
            require_int(row[field], f"benchmark measurement.{field}", minimum=0)
        if row["accepted_tokens"] > row["proposed_tokens"]:
            raise ContractError("benchmark accepted more tokens than it proposed")
        if row["state_checkpoint_rounds"] != row["serial_state_reconstruction_rounds"]:
            raise ContractError("benchmark checkpoint and reconstruction counts disagree")
        if response["k"] <= 2 and row["state_checkpoint_rounds"] != 0:
            raise ContractError("native-tail-trim benchmark unexpectedly checkpointed state")
        if row["reanchor_successes"] > row["reanchor_attempts"]:
            raise ContractError("benchmark has more re-anchor successes than attempts")
        normalized.append(dict(row))
    return {**dict(response), "measurements": normalized}


def run_live_benchmark(
    *,
    config_path: Path,
    manifest_path: Path,
    trace_path: Path,
    pairs: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    k: int = 2,
    output_tokens: int = 48,
    warmup_repetitions: int = 2,
    measurement_repetitions: int = 8,
    pair_mode: str = "exact_repeat",
    reanchor_min_tokens: int = 0,
    forced_rejection_position: int | None = None,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
) -> ResultReceipt:
    if not pairs or len(set(pairs)) != len(pairs):
        raise ContractError("benchmark requires unique trace pairs")
    config = load_bridge_config(config_path)
    records = load_trace_fixture(trace_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    host = target_observer()
    if manifest["evidence_class"] != "LOCAL_MEASURED" or manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("benchmark requires the matching LOCAL_MEASURED target manifest")
    model_hash = file_sha256(config.model_artifact)
    binding = manifest["verifier_gate"]["adapter_binding"]
    if manifest["model_artifact"]["sha256"] != model_hash or binding["model_artifact_sha256"] != model_hash:
        raise ContractError("benchmark model differs from the observed manifest")
    bridge_hash = file_sha256(config.bridge_executable)
    if binding["backend_binary_sha256"] != bridge_hash:
        raise ContractError("benchmark bridge differs from the observed manifest")
    if binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint:
        raise ContractError("benchmark runtime differs from the observed manifest")

    cases = []
    ratios_by_request: dict[str, list[float]] = {}
    request_fingerprints: dict[str, str] = {}
    for pair_spec in pairs:
        source, target = _pair(records, pair_spec, pair_mode=pair_mode)
        request = _request(
            config, source, target, k=k, output_tokens=output_tokens,
            warmup_repetitions=warmup_repetitions, measurement_repetitions=measurement_repetitions,
            reanchor_min_tokens=reanchor_min_tokens,
            forced_rejection_position=forced_rejection_position,
        )
        response = _validate_response(
            bridge_runner(config, request), request, config,
            expected_authoritative_tokens=target["content"]["committed_token_ids"],
        )
        ratios = []
        for row in response["measurements"]:
            paired_serial = (row["serial_before_ns"] + row["serial_after_ns"]) / 2
            ratios.append(paired_serial / row["candidate_decode_ns"])
        ratios_by_request[target["request_id"]] = ratios
        request_fingerprints[response["case_id"]] = sha256(request)
        cases.append({
            "source_record_id": source["record_id"], "target_record_id": target["record_id"],
            "request_id": target["request_id"], "response": response,
            "paired_decode_ratio_mean": fmean(ratios),
        })
    confidence = clustered_bootstrap_ci(ratios_by_request)
    analysis = {
        "paired_serial_over_candidate_decode_ratio": asdict(confidence),
        "case_count": len(cases),
        "all_exact_output_and_state": all(case["response"]["exact_output_and_state"] for case in cases),
        "decision": "CONTROLLED_LIVE_DECODE_POSITIVE" if confidence.lower > 1.0 else "CONTROLLED_LIVE_DECODE_INCONCLUSIVE",
    }
    revision = resolve_git_revision(repository_root)
    experiment_kind = "forced_rejection" if forced_rejection_position is not None else pair_mode
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.llamacpp_live_benchmark", command_arguments, repository_root),
        git_revision=revision,
        input_fingerprints={
            "bridge_binary": bridge_hash, "config": file_sha256(config.config_path),
            "llama_cpp_library": file_sha256(config.llama_cpp_library), "manifest": file_sha256(manifest_path),
            "model_artifact": model_hash, "trace": file_sha256(trace_path), **request_fingerprints,
        },
        status=analysis["decision"],
        payload={
            "kind": f"controlled_{experiment_kind}_pending_live_decode",
            "host_identity": host.as_dict(), "cases": cases, "analysis": analysis,
            "forced_rejection_position": forced_rejection_position,
            "runtime_claim": f"controlled {experiment_kind} decode-only A/B; provider lookup, prefill, and restore excluded; not general PCSS speedup",
            "not_claimed": ["representative workload speedup", "end-to-end provider speedup", "profitable production K", "thermal sustainability"],
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS controlled pending-token live decode")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=48)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--measurement-repetitions", type=int, default=8)
    parser.add_argument("--pair-mode", choices=("exact_repeat", "source_reuse"), default="exact_repeat")
    parser.add_argument("--reanchor-min-tokens", type=int, default=0)
    parser.add_argument("--forced-rejection-position", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_live_benchmark(
        config_path=parsed.config, manifest_path=parsed.manifest, trace_path=parsed.trace,
        pairs=parsed.pair, output_path=parsed.output, report_path=parsed.report,
        repository_root=parsed.repository_root, command_arguments=arguments, k=parsed.k,
        output_tokens=parsed.output_tokens, warmup_repetitions=parsed.warmup_repetitions,
        measurement_repetitions=parsed.measurement_repetitions,
        pair_mode=parsed.pair_mode,
        reanchor_min_tokens=parsed.reanchor_min_tokens,
        forced_rejection_position=parsed.forced_rejection_position,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
