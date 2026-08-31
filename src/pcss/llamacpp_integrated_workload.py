"""Fail-closed, plan-bound llama.cpp PCSS integrated-workload executor.

This module deliberately has no fallback to ``llamacpp_live_benchmark``.  A
measurement is possible only when the native bridge exposes the integrated
pending protocol; until then the preflight is useful, but execution fails
closed before any target request is made.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Any, Mapping, Sequence

from .archive import ImmutableTokenArchive, open_immutable_archive
from .canonical import ContractError, canonical_json, require_mapping, sha256
from .integrated_run_plan import (
    QualificationPolicy,
    _record_identity,
    build_integrated_run_plan,
    validate_integrated_run_plan,
)
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config
from .manifest import validate_manifest
from .offline_runner import load_trace_fixture
from .replay import replay_chronologically
from .results import file_sha256
from .results import make_receipt, recorded_module_command, resolve_git_revision, ResultReceipt
from .metrics import clustered_bootstrap_ci
from .verifier_runner import observe_target_host


INTEGRATED_OPERATION = "integrated_pending"


@dataclass(frozen=True)
class PreparedIntegratedWorkload:
    """Inputs that have passed every non-measuring binding check."""

    plan: dict[str, Any]
    records: tuple[Any, ...]
    archive: ImmutableTokenArchive
    config: BridgeConfig
    fingerprints: dict[str, str]


@dataclass
class _PlanBoundRequest:
    """Minimal provider lifecycle with exactly one sealed cursor, never a search."""
    archive: ImmutableTokenArchive
    source_record_id: str
    source_cursor: int
    proposed_k: int
    open: bool = False
    feedback_received: bool = False

    def begin_request(self) -> None:
        if self.open:
            raise ContractError("plan-bound request begun twice")
        self.open = True

    def propose_or_abstain(self) -> tuple[int, ...] | None:
        if not self.open:
            raise ContractError("proposal before begin_request")
        tokens = self.archive.open_cursor(self.source_record_id, position=self.source_cursor).read(self.proposed_k)
        return tokens or None

    def feedback(self, committed: tuple[int, ...]) -> None:
        if not self.open or not committed:
            raise ContractError("invalid plan-bound feedback")
        self.feedback_received = True

    def end_request(self) -> None:
        if not self.open or not self.feedback_received:
            raise ContractError("end_request before feedback")
        self.open = False


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to load {field}") from error
    return dict(require_mapping(value, field))


def _trace_identity(records: Sequence[Any]) -> str:
    return f"sha256:{sha256([_record_identity(record) for record in records])}"


def _bind_plan_to_trace(plan: Mapping[str, Any], records: tuple[Any, ...]) -> None:
    """Rebuild the sealed decision list; self-consistent plan JSON is insufficient."""

    if plan["trace_sha256"] != _trace_identity(records):
        raise ContractError("integrated plan trace fingerprint differs from supplied trace")
    rebuilt = build_integrated_run_plan(replay_chronologically(records), policy=QualificationPolicy.from_mapping(plan["policy"]))
    if canonical_json(rebuilt) != canonical_json(plan):
        raise ContractError("integrated plan decisions do not exactly bind to supplied trace")


def _bind_archive(plan: Mapping[str, Any], archive: ImmutableTokenArchive, records: tuple[Any, ...]) -> None:
    by_id = {record.record_id: record for record in records}
    for case in plan["selected_cases"]:
        decision = case["decision"]
        source = by_id.get(decision["source_record_id"])
        target = by_id.get(decision["target_record_id"])
        if source is None or target is None:
            raise ContractError("integrated plan names a record absent from supplied trace")
        try:
            cursor = archive.open_cursor(source.record_id, position=decision["source_cursor"])
        except Exception as error:
            raise ContractError("sealed archive cannot open planned source cursor") from error
        proposed = cursor.read(decision["proposed_k"])
        expected = source.committed_tokens[decision["source_cursor"] : decision["source_cursor"] + decision["proposed_k"]]
        if proposed != expected or not proposed:
            raise ContractError("sealed archive source cursor differs from planned trace continuation")
        if target.request_id != decision["target_request_id"] or decision["target_position"] >= len(target.committed_tokens):
            raise ContractError("integrated plan target position differs from supplied trace")


def _bind_target(manifest: Mapping[str, Any], config: BridgeConfig) -> None:
    gate = manifest["verifier_gate"]
    binding = gate["adapter_binding"]
    model_hash = file_sha256(config.model_artifact)
    bridge_hash = file_sha256(config.bridge_executable)
    if (
        manifest["evidence_class"] != "LOCAL_MEASURED"
        or gate["host_identity"] != observe_target_host().as_dict()
        or manifest["model_artifact"]["sha256"] != model_hash
        or binding["backend_binary_sha256"] != bridge_hash
        or binding["model_artifact_sha256"] != model_hash
        or binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint
    ):
        raise ContractError("manifest, model, bridge, runtime, or host binding mismatch")


def prepare_integrated_workload(*, plan_path: Path, trace_path: Path, archive_path: Path, manifest_path: Path, config_path: Path) -> PreparedIntegratedWorkload:
    """Validate exactly the five immutable executor inputs, without measuring."""

    plan = _load_json(plan_path, "integrated run plan")
    validate_integrated_run_plan(plan)
    if plan["status"] != "QUALIFIED" or plan["future_tokens_used_for_retrieval"] or not plan["selected_cases"]:
        raise ContractError("only a qualified future-token-free sealed plan may be executed")
    records = tuple(load_trace_fixture(trace_path))
    _bind_plan_to_trace(plan, records)
    archive = open_immutable_archive(archive_path)
    _bind_archive(plan, archive, records)
    manifest = _load_json(manifest_path, "target manifest")
    validate_manifest(manifest)
    config = load_bridge_config(config_path)
    _bind_target(manifest, config)
    return PreparedIntegratedWorkload(
        plan=plan, records=records, archive=archive, config=config,
        fingerprints={
            "plan": file_sha256(plan_path), "trace": file_sha256(trace_path),
            "archive_manifest": file_sha256(archive.root / "archive-manifest.json"), "manifest": file_sha256(manifest_path),
            "model": file_sha256(config.model_artifact), "bridge": file_sha256(config.bridge_executable),
            "llama_cpp_library": file_sha256(config.llama_cpp_library), "runtime": config.runtime_configuration_fingerprint,
            "host": observe_target_host().as_dict()["fingerprint"],
        },
    )


def _bridge_verify_pending(config: BridgeConfig, *, case_id: str, context: tuple[int, ...], draft: tuple[int, ...]) -> dict[str, Any]:
    """One native pending verification; no selection logic is sent to the bridge."""
    if not context or not draft:
        raise ContractError("integrated native verification needs a pending anchor and non-empty draft")
    request = {
        "schema_version": "pcss.llama-bridge/v1", "operation": "verify_pending",
        "model_artifact": str(config.model_artifact), "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision, "runtime": config.runtime,
        "seed": {"seed_id": case_id, "initial_context_tokens": list(context)},
        "case": {"case_id": case_id, "draft_tokens": list(draft), "include_bonus_on_full_accept": True},
    }
    try:
        completed = subprocess.run((str(config.bridge_executable), "--pcss-verify-pending-case"), input=canonical_json(request) + "\n", capture_output=True, text=True, check=False, timeout=config.request_timeout_seconds)
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ContractError("native pending verification did not return strict JSON") from error
    if completed.returncode != 0 or not isinstance(response, dict) or response.get("operation") != "verify_pending":
        raise ContractError("native pending verification failed")
    required = {"status", "verification_targets_match", "state_matches", "continuation_matches", "committed_tokens", "accepted_draft_tokens", "mutable_state_digest", "configuration", "observed_context"}
    if not required.issubset(response):
        raise ContractError("native pending verification response is incomplete")
    if response["configuration"].get("model_artifact") != str(config.model_artifact) or response["configuration"].get("llama_cpp_library") != str(config.llama_cpp_library) or response["configuration"].get("runtime") != config.runtime:
        raise ContractError("native pending verification echoed a mismatched target configuration")
    if response["status"] != "ok" or not all(response[key] is True for key in ("verification_targets_match", "state_matches", "continuation_matches")):
        raise ContractError("native pending verification reported output or state mismatch")
    return response


def _serial_reference(config: BridgeConfig, *, case_id: str, context: tuple[int, ...], steps: int) -> tuple[int, ...]:
    """Same context/anchor and generated token count as the pending arm."""
    request = {
        "schema_version": "pcss.llama-bridge/v1", "operation": "reference_pending",
        "model_artifact": str(config.model_artifact), "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision, "runtime": config.runtime,
        "seed": {"seed_id": case_id, "initial_context_tokens": list(context), "steps": steps},
        "reference": {"method": "pending-token-serial-greedy-memory-state/v1"},
    }
    try:
        completed = subprocess.run((str(config.bridge_executable), "--pcss-reference-pending"), input=canonical_json(request) + "\n", capture_output=True, text=True, check=False, timeout=config.request_timeout_seconds)
        response = json.loads(completed.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ContractError("serial reference did not return strict JSON") from error
    tokens = response.get("target_tokens") if isinstance(response, dict) else None
    if completed.returncode != 0 or response.get("status") != "ok" or not isinstance(tokens, list) or len(tokens) != steps:
        raise ContractError("serial reference failed")
    return tuple(tokens)


def run_integrated_workload(*, plan_path: Path, trace_path: Path, archive_path: Path, manifest_path: Path, config_path: Path) -> ResultReceipt:
    """Execute only sealed decisions; no archive search, ranking, or static benchmark.

    Callers must separately authorize target execution.  This function performs
    real target work when called, so tests replace the two private bridge calls.
    """
    prepared = prepare_integrated_workload(plan_path=plan_path, trace_path=trace_path, archive_path=archive_path, manifest_path=manifest_path, config_path=config_path)
    by_id = {record.record_id: record for record in prepared.records}
    samples: list[dict[str, Any]] = []
    clusters: dict[str, list[float]] = {}
    for ordinal, case in enumerate(prepared.plan["selected_cases"]):
        decision = case["decision"]
        target = by_id[decision["target_record_id"]]
        source = by_id[decision["source_record_id"]]
        position, k = decision["target_position"], decision["proposed_k"]
        context = target.prompt_tokens + target.committed_tokens[:position]
        if not context:
            raise ContractError("plan position has no pending anchor")
        request_id = f"{decision['target_request_id']}:{position}:{ordinal}"
        provider = _PlanBoundRequest(prepared.archive, source.record_id, decision["source_cursor"], k)
        provider.begin_request()
        # Timer starts before provider/cursor lookup and ends after feedback/end.
        candidate_start = monotonic_ns()
        draft = provider.propose_or_abstain()
        if draft is None:
            raise ContractError("planned cursor unexpectedly abstained")
        response = _bridge_verify_pending(prepared.config, case_id=request_id, context=context, draft=draft)
        committed = tuple(response["committed_tokens"])
        provider.feedback(committed)
        provider.end_request()
        # Native verification, correction, feedback, and state synchronization are complete.
        candidate_ns = monotonic_ns() - candidate_start
        serial_start = monotonic_ns()
        serial = _serial_reference(prepared.config, case_id=request_id, context=context, steps=len(committed))
        serial_ns = monotonic_ns() - serial_start
        if serial != committed:
            raise ContractError("serial baseline and native pending committed sequence differ")
        ratio = serial_ns / candidate_ns if candidate_ns else 0.0
        clusters.setdefault(decision["target_request_id"], []).append(ratio)
        samples.append({"request_id": decision["target_request_id"], "target_record_id": target.record_id, "target_position": position, "source_record_id": source.record_id, "source_cursor": decision["source_cursor"], "serial_ns": serial_ns, "pcss_ns": candidate_ns, "serial_over_pcss": ratio, "accepted_draft_tokens": response["accepted_draft_tokens"], "committed_tokens": list(committed), "output_exact": True, "state_exact": True, "lifecycle": ["begin_request", "exact_source_cursor_lookup", "native_pending_verification", "feedback", "end_request"]})
    ci = clustered_bootstrap_ci(clusters)
    positive = all(sample["output_exact"] and sample["state_exact"] for sample in samples) and ci.lower > 1.0
    repository_root = Path(__file__).resolve().parents[2]
    return make_receipt(evidence_class="LOCAL_MEASURED", command=recorded_module_command("pcss.llamacpp_integrated_workload", (), repository_root), git_revision=resolve_git_revision(repository_root), input_fingerprints=prepared.fingerprints, status="CONTROLLED_INTEGRATED_POSITIVE" if positive else "CONTROLLED_INTEGRATED_NOT_POSITIVE", payload={"kind": "plan_bound_lookup_prefill_restore_integrated_workload", "runtime_claim": "positive only when exact output/state holds for every case and clustered serial-over-PCSS lower 95% CI is greater than 1.0", "positive_claim": positive, "clustered_serial_over_pcss_ci": ci.__dict__, "paired_samples": samples, "selection": "sealed plan only; no runtime reranking or candidate search"})
