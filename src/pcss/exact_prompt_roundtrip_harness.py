"""Closed-loop sealed archive provider plus native pending-verifier harness.

Each provider K-token proposal is verified by a fresh isolated native bridge
round. The bridge's authoritative committed tokens are fed back to the provider
before the next proposal. This is bounded research harness work, not a
long-lived llama.cpp provider hook.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_int
from .exact_prompt_archive import open_exact_prompt_archive
from .exact_prompt_live_harness import _fingerprints, _scope
from .exact_prompt_provider import ExactPromptArchiveShadowProvider, ExactPromptDescriptor
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config
from .llamacpp_live_benchmark import _pair
from .manifest import validate_manifest
from .offline_runner import load_trace_fixture
from .provider import Abstention, Feedback, Proposal, RequestContext
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


def _request(config: BridgeConfig, target: Mapping[str, Any], *, step: int, context_tokens: Sequence[int], draft_tokens: Sequence[int]) -> dict[str, Any]:
    return {
        "schema_version": "pcss.llama-bridge-request/v1",
        "operation": "verify_pending",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "seed": {"seed_id": f"{target['record_id']}:round:{step}", "initial_context_tokens": list(context_tokens)},
        "case": {
            "case_id": f"{target['record_id']}:round:{step}", "draft_tokens": list(draft_tokens),
            "include_bonus_on_full_accept": True,
        },
    }


def _run_bridge(config: BridgeConfig, request: Mapping[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-verify-pending-case"),
            input=canonical_json(request) + "\n", capture_output=True, text=True, check=False,
            timeout=config.request_timeout_seconds,
        )
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ContractError("native pending verifier did not return one JSON response") from error
    if completed.returncode != 0 or not isinstance(response, dict):
        raise ContractError("native pending verifier failed")
    return response


def _validate_round(response: Mapping[str, Any], proposal: Proposal) -> tuple[tuple[int, ...], int, dict[str, Any]]:
    if response.get("operation") != "verify_pending" or response.get("status") != "ok":
        raise ContractError("native pending verifier round is not successful")
    for field in ("verification_targets_match", "pending_state_matches", "state_matches", "continuation_matches"):
        if response.get(field) is not True:
            raise ContractError(f"native pending verifier round failed {field}")
    committed = response.get("committed_tokens")
    reference = response.get("reference_committed_tokens")
    if not isinstance(committed, list) or committed != reference or not committed:
        raise ContractError("native pending verifier committed token stream is invalid")
    accepted = require_int(response.get("accepted_draft_tokens"), "accepted_draft_tokens", minimum=0)
    if accepted > len(proposal.tokens) or tuple(committed[:accepted]) != proposal.tokens[:accepted]:
        raise ContractError("native pending verifier acceptance disagrees with provider proposal")
    return tuple(require_int(token, "committed_tokens", minimum=0) for token in committed), accepted, {
        "state_sha256": response.get("committed_state_sha256"),
        "state_bytes": response.get("committed_state_bytes"),
        "pending_state_sha256": response.get("pending_state_sha256"),
        "pending_state_bytes": response.get("pending_state_bytes"),
        "accepted_draft_tokens": accepted,
        "correction_token": response.get("correction_token"),
        "bonus_token": response.get("bonus_token"),
    }


def _run_pair(config: BridgeConfig, archive, source: Mapping[str, Any], target: Mapping[str, Any], *, k: int, output_tokens: int, bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
    prompt = tuple(target["content"]["prompt_token_ids"])
    expected = tuple(target["content"]["committed_token_ids"][:output_tokens])
    fingerprints = _fingerprints(target)
    scope = _scope(target, widening_level=1)
    descriptor = ExactPromptDescriptor(
        tail_tokens=prompt, source_record_id=source["record_id"], source_class="same_repository",
        scope=scope, fingerprints=fingerprints, provenance_family=source["source"]["provenance_family"],
        matched_suffix_tokens=len(prompt),
    )
    provider = ExactPromptArchiveShadowProvider(archive, (descriptor,))
    request_context = RequestContext(
        request_id=target["request_id"], fingerprints=fingerprints, scope=scope,
        source_tiers=("same_repository",), deadline_ns=10**30, probe_budget=1,
    )
    provider.begin_request(request_context)
    committed: list[int] = []
    rounds: list[dict[str, Any]] = []
    while len(committed) < output_tokens:
        proposal = provider.propose(
            request_context.request_id, tail_tokens=prompt if not committed else (), max_tokens=k, deadline_ns=10**30
        )
        if isinstance(proposal, Abstention):
            raise ContractError(f"exact-prompt provider abstained at round {len(rounds)}: {proposal.reason}")
        if not isinstance(proposal, Proposal):
            raise ContractError("exact-prompt provider returned an unsupported result")
        response = bridge_runner(
            config,
            _request(config, target, step=len(rounds), context_tokens=prompt + tuple(committed), draft_tokens=proposal.tokens),
        )
        round_committed, accepted, state = _validate_round(response, proposal)
        if len(committed) + len(round_committed) > output_tokens:
            raise ContractError("native pending verifier committed beyond the requested output boundary")
        archive_expected = archive.archive.read_tokens(
            source["record_id"], len(prompt) + len(committed), len(round_committed)
        )
        if archive_expected != round_committed:
            raise ContractError(
                f"native committed tokens diverge from archive at round {len(rounds)}: "
                f"archive={archive_expected!r} native={round_committed!r}"
            )
        feedback_status = "all_accepted" if accepted == len(proposal.tokens) else "rejected"
        provider.feedback(
            request_context.request_id,
            Feedback(proposal.proposal_id, feedback_status, accepted, round_committed, None, "same_repository"),
        )
        rounds.append({
            "round": len(rounds), "proposal_tokens": list(proposal.tokens), "committed_tokens": list(round_committed),
            "accepted_draft_tokens": accepted, "feedback_status": feedback_status, **state,
        })
        committed.extend(round_committed)
    provider.end_request(request_context.request_id)
    if tuple(committed) != expected:
        raise ContractError("closed-loop native provider output differs from recorded authoritative trace")
    return {
        "source_record_id": source["record_id"], "target_record_id": target["record_id"],
        "request_id": target["request_id"], "rounds": rounds, "output_tokens": committed,
        "exact_output": True,
        "telemetry": [
            {"event": event.event, "request_id": event.request_id, "at_ns": event.at_ns, **event.payload}
            for event in provider.telemetry
        ],
    }


def run_roundtrip_harness(
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
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
) -> ResultReceipt:
    if k < 1 or output_tokens < 1 or not pairs or len(set(pairs)) != len(pairs):
        raise ContractError("roundtrip harness requires positive k/output and unique pairs")
    config = load_bridge_config(config_path)
    records = load_trace_fixture(trace_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    host = target_observer()
    if manifest["evidence_class"] != "LOCAL_MEASURED" or manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("roundtrip harness requires matching LOCAL_MEASURED target manifest")
    archive = open_exact_prompt_archive(archive_root)
    cases = []
    for pair_spec in pairs:
        source, target = _pair(records, pair_spec, pair_mode="exact_repeat")
        cases.append(_run_pair(config, archive, source, target, k=k, output_tokens=output_tokens, bridge_runner=bridge_runner))
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.exact_prompt_roundtrip_harness", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "manifest": file_sha256(manifest_path), "trace": file_sha256(trace_path),
            "archive_manifest": file_sha256(archive_root / "archive-manifest.json"),
            "config": file_sha256(config.config_path), "bridge_binary": file_sha256(config.bridge_executable),
            "model_artifact": file_sha256(config.model_artifact), "llama_cpp_library": file_sha256(config.llama_cpp_library),
        },
        status="EXACT_PROMPT_PROVIDER_ROUNDTRIP_VERIFIED",
        payload={
            "kind": "sealed_archive_provider_native_pending_roundtrip",
            "host_identity": host.as_dict(), "k": k, "case_count": len(cases), "cases": cases,
            "round_count": sum(len(case["rounds"]) for case in cases),
            "runtime_claim": "closed-loop provider plus isolated native verifier correctness harness; no throughput or production integration claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS exact-prompt provider native roundtrip")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_roundtrip_harness(
        config_path=parsed.config, manifest_path=parsed.manifest, trace_path=parsed.trace,
        archive_root=parsed.archive_root, pairs=parsed.pair, output_path=parsed.output,
        report_path=parsed.report, repository_root=parsed.repository_root, command_arguments=arguments,
        k=parsed.k, output_tokens=parsed.output_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
