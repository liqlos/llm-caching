"""Isolated perfect-draft verifier profiler for an already-authorized M4 target.

This profiler measures paired serial-decode and one-batch verification blocks.
It does not attach PCSS to live decoding and cannot make a workload speedup
claim.  Every token block is a serially derived perfect draft.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, sha256
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config, load_fixture_seeds
from .manifest import validate_manifest
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
from .verifier import TargetHostIdentity
from .verifier_runner import observe_target_host


PROFILE_SCHEMA_VERSION = "pcss.llama-bridge-profile/v1"
PENDING_PROFILE_SCHEMA_VERSION = "pcss.llama-bridge-pending-profile/v1"
REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
_RESPONSE_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "measurement_clock",
    "measurement_scope",
    "claim_boundary",
    "seed_id",
    "warmup_repetitions",
    "measurement_repetitions",
    "prompt_state_sha256",
    "prompt_state_bytes",
    "profiles",
    "configuration",
    "observed_context",
    "state_export",
    "state_api",
    "generation_mode",
    "greedy_tie_break",
    "tokenization_boundary",
    "chat_template_applied",
    "thinking_mode_applied",
}
_PROFILE_FIELDS = {"k", "perfect_draft_tokens", "measurements"}
_MEASUREMENT_FIELDS = {
    "iteration",
    "restore_before_ns",
    "serial_before_ns",
    "restore_batch_ns",
    "batch_verify_ns",
    "restore_after_ns",
    "serial_after_ns",
    "snapshot_ns",
}
_OBSERVED_CONTEXT_FIELDS = {"actual_n_ctx", "actual_n_batch", "actual_n_threads", "actual_n_threads_batch"}


def _strict_json_object(text: str, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"{field} must be one strict JSON object") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{field} must be one strict JSON object")
    return payload


def _request(
    config: BridgeConfig,
    *,
    seed_id: str,
    context: tuple[int, ...],
    k_values: tuple[int, ...],
    warmup_repetitions: int,
    measurement_repetitions: int,
    commit_protocol: str = "serial",
) -> dict[str, Any]:
    if commit_protocol not in {"serial", "pending_native"}:
        raise ContractError("unsupported verifier profile commit protocol")
    if not k_values or len(set(k_values)) != len(k_values) or any(k < 1 for k in k_values):
        raise ContractError("profile k values must be positive and unique")
    required_batch = max(k_values) + (1 if commit_protocol == "pending_native" else 0)
    if required_batch > config.runtime["n_batch"]:
        raise ContractError("profile k exceeds configured n_batch")
    if warmup_repetitions < 0 or measurement_repetitions < 2:
        raise ContractError("profile requires non-negative warmups and at least two measurements")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "profile" if commit_protocol == "serial" else "profile_pending",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "seed": {"seed_id": seed_id, "initial_context_tokens": list(context)},
        "profile": {
            "k_values": list(k_values),
            "warmup_repetitions": warmup_repetitions,
            "measurement_repetitions": measurement_repetitions,
        },
    }


def _run_bridge(config: BridgeConfig, request: Mapping[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    if operation == "profile":
        argument = "--pcss-profile"
    elif operation == "profile_pending":
        argument = "--pcss-profile-pending"
    else:
        raise ContractError("unsupported pcss_llama_bridge profile operation")
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), argument),
            input=canonical_json(request) + "\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("pcss_llama_bridge profile process did not complete") from error
    response = _strict_json_object(completed.stdout, "pcss_llama_bridge profile response")
    if completed.returncode != 0:
        error = response.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str):
            raise ContractError(f"pcss_llama_bridge profile failed: {error['code']}: {error['message']}")
        raise ContractError("pcss_llama_bridge profile process returned a non-zero exit status")
    return response


def _sha256_value(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_ns(value: object, field: str) -> int:
    return require_int(value, field, minimum=1)


def _validate_response(
    response: Mapping[str, Any],
    *,
    config: BridgeConfig,
    seed_id: str,
    k_values: tuple[int, ...],
    warmup_repetitions: int,
    measurement_repetitions: int,
    commit_protocol: str = "serial",
) -> tuple[dict[str, Any], ...]:
    if commit_protocol == "serial":
        expected_schema = PROFILE_SCHEMA_VERSION
        expected_operation = "profile"
        expected_scope = "decode-and-synchronize/no-context-creation/v1"
        expected_boundary = "isolated-perfect-draft-verifier-profile/not-live-pcss/v1"
    elif commit_protocol == "pending_native":
        expected_schema = PENDING_PROFILE_SCHEMA_VERSION
        expected_operation = "profile_pending"
        expected_scope = "pending-anchor-decode-and-synchronize/no-context-creation/v1"
        expected_boundary = "isolated-exact-state-pending-token-profile/not-live-pcss/v1"
    else:
        raise ContractError("unsupported verifier profile commit protocol")
    require_exact_keys(response, _RESPONSE_FIELDS, "profile response")
    if (
        response["schema_version"] != expected_schema
        or response["operation"] != expected_operation
        or response["status"] != "ok"
        or response["backend"] != "llama.cpp"
    ):
        raise ContractError("profile response is not a successful supported llama.cpp profile")
    expected_literals = {
        "measurement_clock": "std::chrono::steady_clock/nanoseconds/v1",
        "measurement_scope": expected_scope,
        "claim_boundary": expected_boundary,
        "state_export": "model-architecture+memory-state/v2",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
    }
    if any(response[field] != value for field, value in expected_literals.items()):
        raise ContractError("profile response has an unsupported measurement or generation contract")
    if response["chat_template_applied"] is not False or response["thinking_mode_applied"] is not False:
        raise ContractError("profile response unexpectedly applied template or thinking policy")
    if response["seed_id"] != seed_id:
        raise ContractError("profile response seed identity differs from request")
    if response["warmup_repetitions"] != warmup_repetitions or response["measurement_repetitions"] != measurement_repetitions:
        raise ContractError("profile response repetition counts differ from request")
    _sha256_value(response["prompt_state_sha256"], "profile response.prompt_state_sha256")
    require_int(response["prompt_state_bytes"], "profile response.prompt_state_bytes", minimum=1)
    expected_configuration = {
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
    }
    if response["configuration"] != expected_configuration:
        raise ContractError("profile response configuration differs from request")
    observed = require_mapping(response["observed_context"], "profile response.observed_context")
    require_exact_keys(observed, _OBSERVED_CONTEXT_FIELDS, "profile response.observed_context")
    if dict(observed) != {
        "actual_n_ctx": config.runtime["n_ctx"],
        "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }:
        raise ContractError("profile response observed context differs from request")

    values = response["profiles"]
    if not isinstance(values, list) or len(values) != len(k_values):
        raise ContractError("profile response must contain exactly one row per requested k")
    validated: list[dict[str, Any]] = []
    for expected_k, item in zip(k_values, values):
        row = require_mapping(item, "profile response row")
        require_exact_keys(row, _PROFILE_FIELDS, "profile response row")
        if row["k"] != expected_k:
            raise ContractError("profile response k order differs from request")
        tokens = row["perfect_draft_tokens"]
        if not isinstance(tokens, list) or len(tokens) != expected_k or any(type(token) is not int or token < 0 for token in tokens):
            raise ContractError("profile perfect draft tokens are invalid")
        measurements = row["measurements"]
        if not isinstance(measurements, list) or len(measurements) != measurement_repetitions:
            raise ContractError("profile response measurement count differs from request")
        normalized: list[dict[str, int]] = []
        for iteration, measurement in enumerate(measurements):
            measured = require_mapping(measurement, "profile measurement")
            require_exact_keys(measured, _MEASUREMENT_FIELDS, "profile measurement")
            if measured["iteration"] != iteration:
                raise ContractError("profile measurement iterations must be chronological and contiguous")
            normalized.append(
                {
                    "iteration": iteration,
                    **{
                        field: _positive_ns(measured[field], f"profile measurement.{field}")
                        for field in _MEASUREMENT_FIELDS - {"iteration"}
                    },
                }
            )
        validated.append({"k": expected_k, "perfect_draft_tokens": list(tokens), "measurements": normalized})
    return tuple(validated)


def _analyze(profiles: tuple[dict[str, Any], ...], *, baseline_tolerance: float) -> dict[str, Any]:
    if not 0 < baseline_tolerance < 1:
        raise ContractError("baseline tolerance must be between zero and one")
    rows: list[dict[str, Any]] = []
    all_repeatable = True
    for profile in profiles:
        measurements = profile["measurements"]
        serial_before = [row["serial_before_ns"] for row in measurements]
        serial_after = [row["serial_after_ns"] for row in measurements]
        serial_paired = [(before + after) / 2.0 for before, after in zip(serial_before, serial_after)]
        batch = [row["batch_verify_ns"] for row in measurements]
        paired_ratios = {
            f"iteration-{index}": (serial / batch_value,)
            for index, (serial, batch_value) in enumerate(zip(serial_paired, batch))
        }
        interval = clustered_bootstrap_ci(paired_ratios, bootstrap_replicates=2_000, seed=profile["k"])
        before_mean = fmean(serial_before)
        after_mean = fmean(serial_after)
        repeat_delta = abs(after_mean - before_mean) / fmean((before_mean, after_mean))
        repeatable = repeat_delta <= baseline_tolerance
        all_repeatable = all_repeatable and repeatable
        rows.append(
            {
                "k": profile["k"],
                "measurement_repetitions": len(measurements),
                "serial_before_mean_ns": before_mean,
                "serial_after_mean_ns": after_mean,
                "batch_verify_mean_ns": fmean(batch),
                "restore_mean_ns": fmean(
                    row[field]
                    for row in measurements
                    for field in ("restore_before_ns", "restore_batch_ns", "restore_after_ns")
                ),
                "snapshot_mean_ns": fmean(row["snapshot_ns"] for row in measurements),
                "baseline_repeat_relative_delta": repeat_delta,
                "baseline_repeatable": repeatable,
                "paired_decode_ratio_ci": {
                    "estimate": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                    "confidence_level": interval.confidence_level,
                    "bootstrap_replicates": interval.bootstrap_replicates,
                    "paired_iteration_clusters": interval.request_clusters,
                },
            }
        )
    return {
        "baseline_relative_tolerance": baseline_tolerance,
        "all_baselines_repeatable": all_repeatable,
        "rows": rows,
        "interpretation": (
            "decode-only paired ratio; excludes provider, canonical commit, end-to-end scheduling, "
            "memory pressure, and thermal qualification"
        ),
        "decision": "PROFILE_VALID" if all_repeatable else "BASELINE_REPEATABILITY_FAILED",
    }


def run_profile(
    *,
    config_path: Path,
    seeds_path: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    k_values: tuple[int, ...],
    warmup_repetitions: int,
    measurement_repetitions: int,
    baseline_tolerance: float,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
    commit_protocol: str = "serial",
) -> ResultReceipt:
    config = load_bridge_config(config_path)
    seeds = load_fixture_seeds(seeds_path)
    if len(seeds.contexts) != 1:
        raise ContractError("profile requires exactly one capability-derived seed context")
    seed_id, context = seeds.contexts[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest["evidence_class"] != "LOCAL_MEASURED":
        raise ContractError("profile requires a LOCAL_MEASURED observed manifest")
    host = target_observer()
    if manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("profile target host differs from observed manifest")
    model_hash = file_sha256(config.model_artifact)
    if manifest["model_artifact"]["sha256"] != model_hash:
        raise ContractError("profile model hash differs from observed manifest")
    binding = manifest["verifier_gate"]["adapter_binding"]
    bridge_hash = file_sha256(config.bridge_executable)
    if binding["backend_binary_sha256"] != bridge_hash:
        raise ContractError("profile bridge binary differs from observed manifest")
    if binding["model_artifact_sha256"] != model_hash:
        raise ContractError("profile model binding differs from observed manifest")
    if binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint:
        raise ContractError("profile runtime configuration differs from observed manifest")

    request = _request(
        config,
        seed_id=seed_id,
        context=context,
        k_values=k_values,
        warmup_repetitions=warmup_repetitions,
        measurement_repetitions=measurement_repetitions,
        commit_protocol=commit_protocol,
    )
    response = bridge_runner(config, request)
    profiles = _validate_response(
        response,
        config=config,
        seed_id=seed_id,
        k_values=k_values,
        warmup_repetitions=warmup_repetitions,
        measurement_repetitions=measurement_repetitions,
        commit_protocol=commit_protocol,
    )
    analysis = _analyze(profiles, baseline_tolerance=baseline_tolerance)
    revision = resolve_git_revision(repository_root)
    command = recorded_module_command("pcss.llamacpp_profiler", command_arguments, repository_root)
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=command,
        git_revision=revision,
        input_fingerprints={
            "bridge_binary": bridge_hash,
            "config": file_sha256(config.config_path),
            "fixture_seeds": file_sha256(seeds.seed_path),
            "llama_cpp_library": file_sha256(config.llama_cpp_library),
            "manifest": file_sha256(manifest_path),
            "model_artifact": model_hash,
            "profile_request": sha256(request),
        },
        status=analysis["decision"],
        payload={
            "kind": (
                "isolated_perfect_draft_verifier_profile"
                if commit_protocol == "serial"
                else "isolated_exact_state_pending_token_profile"
            ),
            "host_identity": host.as_dict(),
            "raw_profile": dict(response),
            "analysis": analysis,
            "runtime_claim": "isolated decode profile only; no live PCSS or speedup claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS isolated verifier profile")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, action="append", required=True)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--measurement-repetitions", type=int, default=8)
    parser.add_argument("--baseline-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--commit-protocol", choices=("serial", "pending_native"), default="serial")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    run_profile(
        config_path=args.config,
        seeds_path=args.seeds,
        manifest_path=args.manifest,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root,
        command_arguments=arguments,
        k_values=tuple(args.k),
        warmup_repetitions=args.warmup_repetitions,
        measurement_repetitions=args.measurement_repetitions,
        baseline_tolerance=args.baseline_relative_tolerance,
        commit_protocol=args.commit_protocol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
