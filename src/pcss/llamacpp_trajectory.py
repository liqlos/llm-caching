"""Run a persistent exact-state pending-token trajectory on a target llama.cpp backend."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, require_string, sha256
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config, load_fixture_seeds
from .manifest import validate_manifest
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


PLAN_SCHEMA_VERSION = "pcss.pending-trajectory-plan/v1"
RESPONSE_SCHEMA_VERSION = "pcss.llama-bridge-pending-trajectory/v1"
_PLAN_FIELDS = {"schema_version", "case_id", "k", "rounds"}
_ROUND_FIELDS = {"mode", "rejection_position"}
_RESPONSE_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "case_id",
    "seed_id",
    "k",
    "rounds_requested",
    "rounds_completed",
    "rounds",
    "all_rounds_match",
    "final_state_matches",
    "continuation_matches",
    "final_next_token",
    "state_export",
    "state_api",
    "configuration",
    "observed_context",
    "claim_boundary",
}
_RESPONSE_ROUND_FIELDS = {
    "round",
    "mode",
    "rejection_position",
    "accepted_draft_tokens",
    "pending_token",
    "tail_trimmed",
    "state_matches",
    "verifier_state_sha256",
    "reference_state_sha256",
    "state_bytes",
}


def _strict_object(text: str, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ContractError(f"{field} contains a non-finite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, TypeError) as error:
        raise ContractError(f"unable to parse {field}") from error
    return dict(require_mapping(value, field))


def load_plan(path: Path) -> dict[str, Any]:
    try:
        plan = _strict_object(path.read_text(encoding="utf-8"), "pending trajectory plan")
    except OSError as error:
        raise ContractError("unable to read pending trajectory plan") from error
    require_exact_keys(plan, _PLAN_FIELDS, "pending trajectory plan")
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ContractError("unsupported pending trajectory plan schema")
    require_string(plan["case_id"], "pending trajectory plan.case_id")
    k = require_int(plan["k"], "pending trajectory plan.k", minimum=1)
    rounds = plan["rounds"]
    if not isinstance(rounds, list) or not 1 <= len(rounds) <= 128:
        raise ContractError("pending trajectory plan.rounds must contain between 1 and 128 rounds")
    normalized_rounds: list[dict[str, Any]] = []
    for index, value in enumerate(rounds):
        round_plan = require_mapping(value, f"pending trajectory plan.rounds[{index}]")
        require_exact_keys(round_plan, _ROUND_FIELDS, f"pending trajectory plan.rounds[{index}]")
        mode = require_string(round_plan["mode"], f"pending trajectory plan.rounds[{index}].mode")
        position = round_plan["rejection_position"]
        if type(position) is not int:
            raise ContractError("pending trajectory rejection positions must be integers")
        if (mode == "full_accept" and position != -1) or (
            mode == "reject" and not 0 <= position < k
        ):
            raise ContractError("pending trajectory round mode/rejection_position is invalid")
        if mode not in {"full_accept", "reject"}:
            raise ContractError("pending trajectory round mode is unsupported")
        normalized_rounds.append({"mode": mode, "rejection_position": position})
    return {"schema_version": PLAN_SCHEMA_VERSION, "case_id": plan["case_id"], "k": k, "rounds": normalized_rounds}


def _request(config: BridgeConfig, *, seed_id: str, context: tuple[int, ...], plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "pcss.llama-bridge-request/v1",
        "operation": "trajectory_pending",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": config.runtime,
        "seed": {"seed_id": seed_id, "initial_context_tokens": list(context)},
        "trajectory": {
            "case_id": plan["case_id"],
            "k": plan["k"],
            "rounds": plan["rounds"],
        },
    }


def _run_bridge(config: BridgeConfig, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-trajectory-pending"),
            input=canonical_json(request) + "\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("pending trajectory bridge did not complete") from error
    if completed.returncode not in {0, 1}:
        raise ContractError("pending trajectory bridge returned an unexpected exit status")
    response = _strict_object(completed.stdout, "pending trajectory bridge response")
    if (completed.returncode == 0) != (response.get("status") == "ok"):
        raise ContractError("pending trajectory exit status contradicts its structured response")
    return response


def _sha256(value: object, field: str) -> str:
    text = require_string(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_response(
    response: Mapping[str, Any],
    *,
    config: BridgeConfig,
    seed_id: str,
    plan: Mapping[str, Any],
) -> bool:
    require_exact_keys(response, _RESPONSE_FIELDS, "pending trajectory bridge response")
    if response["schema_version"] != RESPONSE_SCHEMA_VERSION or response["operation"] != "trajectory_pending":
        raise ContractError("pending trajectory response schema or operation is invalid")
    if response["status"] not in {"ok", "mismatch"} or response["backend"] != "llama.cpp":
        raise ContractError("pending trajectory response status or backend is invalid")
    if response["case_id"] != plan["case_id"] or response["seed_id"] != seed_id:
        raise ContractError("pending trajectory response identity differs from its request")
    if response["k"] != plan["k"] or response["rounds_requested"] != len(plan["rounds"]):
        raise ContractError("pending trajectory response dimensions differ from its request")
    rounds = response["rounds"]
    if not isinstance(rounds, list) or response["rounds_completed"] != len(rounds):
        raise ContractError("pending trajectory completed-round count is invalid")
    if not 1 <= len(rounds) <= len(plan["rounds"]):
        raise ContractError("pending trajectory response contains an invalid number of rounds")
    all_rounds_match = True
    for index, value in enumerate(rounds):
        observed = require_mapping(value, f"pending trajectory response.rounds[{index}]")
        require_exact_keys(observed, _RESPONSE_ROUND_FIELDS, f"pending trajectory response.rounds[{index}]")
        expected = plan["rounds"][index]
        expected_accepted = plan["k"] if expected["mode"] == "full_accept" else expected["rejection_position"]
        if (
            observed["round"] != index
            or observed["mode"] != expected["mode"]
            or observed["rejection_position"] != expected["rejection_position"]
            or observed["accepted_draft_tokens"] != expected_accepted
        ):
            raise ContractError("pending trajectory round outcome contradicts its schedule")
        require_int(observed["pending_token"], "pending trajectory pending token", minimum=0)
        require_int(observed["state_bytes"], "pending trajectory state bytes", minimum=1)
        verifier_digest = _sha256(observed["verifier_state_sha256"], "pending trajectory verifier digest")
        reference_digest = _sha256(observed["reference_state_sha256"], "pending trajectory reference digest")
        state_matches = observed["state_matches"]
        tail_trimmed = observed["tail_trimmed"]
        if type(state_matches) is not bool or type(tail_trimmed) is not bool:
            raise ContractError("pending trajectory match and trim claims must be booleans")
        if state_matches != (verifier_digest == reference_digest):
            raise ContractError("pending trajectory state-match claim is inconsistent")
        if not tail_trimmed:
            raise ContractError("pending trajectory did not trim a verification tail")
        all_rounds_match = all_rounds_match and state_matches
    for field in ("all_rounds_match", "final_state_matches", "continuation_matches"):
        if type(response[field]) is not bool:
            raise ContractError(f"pending trajectory response.{field} must be a boolean")
    if response["all_rounds_match"] != all_rounds_match:
        raise ContractError("pending trajectory aggregate round-match claim is inconsistent")
    successful = (
        len(rounds) == len(plan["rounds"])
        and all_rounds_match
        and response["final_state_matches"]
        and response["continuation_matches"]
    )
    if (response["status"] == "ok") != successful:
        raise ContractError("pending trajectory status contradicts exact-state results")
    if successful:
        require_int(response["final_next_token"], "pending trajectory final next token", minimum=0)
    elif response["final_next_token"] is not None:
        require_int(response["final_next_token"], "pending trajectory final next token", minimum=0)
    if response["state_export"] != "model-architecture+memory-state/v2":
        raise ContractError("pending trajectory state export is incomplete")
    if response["state_api"] != "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2":
        raise ContractError("pending trajectory state API is unsupported")
    if response["claim_boundary"] != "exact-state correctness trajectory only; no performance measurement":
        raise ContractError("pending trajectory claim boundary is invalid")
    expected_configuration = {
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": config.runtime,
    }
    if response["configuration"] != expected_configuration:
        raise ContractError("pending trajectory configuration echo differs from its request")
    observed_context = require_mapping(response["observed_context"], "pending trajectory observed context")
    expected_context = {
        "actual_n_ctx": config.runtime["n_ctx"],
        "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }
    if dict(observed_context) != expected_context:
        raise ContractError("pending trajectory observed context differs from its request")
    return successful


def run_trajectory(
    *,
    config_path: Path,
    seeds_path: Path,
    manifest_path: Path,
    plan_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
) -> ResultReceipt:
    config = load_bridge_config(config_path)
    seeds = load_fixture_seeds(seeds_path)
    if len(seeds.contexts) != 1:
        raise ContractError("pending trajectory requires exactly one capability-derived seed context")
    seed_id, context = seeds.contexts[0]
    plan = load_plan(plan_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("unable to read pending trajectory manifest") from error
    validate_manifest(manifest)
    if manifest["evidence_class"] != "LOCAL_MEASURED":
        raise ContractError("pending trajectory requires a LOCAL_MEASURED manifest")
    host = target_observer()
    if manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("pending trajectory host differs from observed manifest")
    model_hash = file_sha256(config.model_artifact)
    bridge_hash = file_sha256(config.bridge_executable)
    binding = manifest["verifier_gate"]["adapter_binding"]
    if (
        manifest["model_artifact"]["sha256"] != model_hash
        or binding["model_artifact_sha256"] != model_hash
        or binding["backend_binary_sha256"] != bridge_hash
        or binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint
    ):
        raise ContractError("pending trajectory inputs differ from observed manifest binding")
    request = _request(config, seed_id=seed_id, context=context, plan=plan)
    response = dict(bridge_runner(config, request))
    successful = _validate_response(response, config=config, seed_id=seed_id, plan=plan)
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.llamacpp_trajectory", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "bridge_binary": bridge_hash,
            "config": file_sha256(config.config_path),
            "fixture_seeds": file_sha256(seeds.seed_path),
            "llama_cpp_library": file_sha256(config.llama_cpp_library),
            "manifest": file_sha256(manifest_path),
            "model_artifact": model_hash,
            "plan": file_sha256(plan_path),
            "trajectory_request": sha256(request),
        },
        status="VERIFIED" if successful else "FAILED",
        payload={
            "kind": "persistent_pending_exact_state_trajectory",
            "execution_class": "TARGET_BACKEND",
            "host_identity": host.as_dict(),
            "raw_trajectory": response,
            "runtime_claim": "exact-state correctness trajectory only; no performance or live PCSS claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS persistent pending-token trajectory")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    receipt = run_trajectory(
        config_path=args.config,
        seeds_path=args.seeds,
        manifest_path=args.manifest,
        plan_path=args.plan,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root.resolve(),
        command_arguments=arguments,
    )
    return 0 if receipt.status == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
