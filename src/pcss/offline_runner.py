"""Reproducible, model-free PCSS replay receipt command.

This command deliberately runs only chronological oracle, immutable-archive,
and logical-capacity work.  Its result is a simulator receipt, never a target
decode, verifier, timing, or speedup claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .archive import TokenRecord, build_immutable_archive
from .capacity import (
    ActivationBundle,
    ArchiveSpanReference,
    CapacityObservation,
    CapacityPolicy,
    CapacitySimulator,
)
from .canonical import ContractError
from .experiments import (
    ConsensusObservation,
    NegativeEvidenceEvent,
    ScopeExperimentResult,
    calibrate_consensus,
    compare_scope_policies,
    consensus_by_provenance,
    simulate_decayed_negative_evidence,
)
from .importer import import_jsonl
from .metrics import (
    CapacityValue,
    accepted_prefix_value,
    active_capture_ratio,
    capacity_curve,
    chain_survival,
    clustered_bootstrap_ci,
    opportunity,
    oracle_reuse_ratio,
    request_cluster_values,
    source_attribution,
    viable_opportunity_recovery_rate,
)
from .replay import Candidate, ReplayResult, SourceTier, continuation_prefix_length, replay_chronologically
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .trace import normalize_trace_record, trace_stream_fingerprint, validate_trace_sequence


PCSS_MASTER_RFC_FILENAME = "PCSS_MASTER_RFC_EN.md"
PCSS_MASTER_RFC_SHA256 = "1dfdc69b055af5059021d130e216ed702a9636fb585cc5c597be783382d5e357"


def _json_value(value: Any) -> Any:
    """Convert contract dataclasses and enums without changing their values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def load_trace_fixture(path: Path) -> list[dict[str, Any]]:
    """Load a visible trace fixture through the same fail-closed trace contract."""

    if path.suffix == ".jsonl":
        result = import_jsonl(path, require_uniform_fingerprints=True)
        if not result.records:
            raise ContractError("offline trace fixture must not be empty")
        return [dict(record) for record in result.records]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read trace fixture: {path}") from error
    if not isinstance(value, list) or not value:
        raise ContractError("offline trace fixture must be a non-empty JSON array")
    if not all(isinstance(record, dict) for record in value):
        raise ContractError("offline trace fixture must contain JSON objects")
    records = [normalize_trace_record(record) for record in value]
    validate_trace_sequence(records, require_uniform_fingerprints=True)
    return records


def _validate_rfc_source(path: Path, *, expected_sha256: str) -> str:
    """Bind a receipt to the versioned master RFC rather than an arbitrary file."""

    if path.name != PCSS_MASTER_RFC_FILENAME:
        raise ContractError(f"RFC source must be named {PCSS_MASTER_RFC_FILENAME}: {path}")
    if not path.is_file():
        raise ContractError(f"RFC source file is required for the receipt: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ContractError("RFC source fingerprint does not match the pinned PCSS master RFC")
    return actual_sha256


def _archive_validation(result: ReplayResult) -> dict[str, Any]:
    """Build and read one temporary sealed archive; no storage timing is collected."""

    token_records = tuple(
        TokenRecord(
            record_id=record.record_id,
            tokens=record.committed_tokens,
            metadata={
                "timestamp_ns": record.timestamp_ns,
                "chronological_index": record.chronological_index,
                "source_class": record.source_class,
                "provenance_family": record.provenance_family,
            },
        )
        for record in result.records
    )
    with TemporaryDirectory(prefix="pcss-offline-archive-") as temporary_root:
        archive = build_immutable_archive(Path(temporary_root) / "archive", token_records)
        cursor_reads = {
            record.record_id: list(archive.open_cursor(record.record_id).read(len(record.committed_tokens)))
            for record in result.records
        }
        return {
            "kind": "sealed_temporary_archive_validation_no_storage_timing",
            "record_count": len(archive.records),
            "token_count": archive.token_count,
            "reference_count": archive.reference_count,
            "posting_limit": archive.posting_limit,
            "byte_accounting": archive.byte_accounting(),
            "cursor_reads_match_input": all(
                tuple(cursor_reads[record.record_id]) == record.committed_tokens for record in result.records
            ),
        }


def _capacity_simulation(result: ReplayResult, common_archive_bytes: int) -> dict[str, Any]:
    """Run two causal logical views over the fixed replay stream, without timing."""

    bundles = tuple(
        ActivationBundle(
            bundle_id=record.record_id,
            archive_span=ArchiveSpanReference(record.record_id, 0, len(record.committed_tokens)),
            active_bytes=max(8, 4 * len(record.committed_tokens)),
            ghost_bytes=4,
        )
        for record in result.records
    )
    record_epochs = {record.record_id: index for index, record in enumerate(result.records)}
    observations: list[CapacityObservation] = []
    ordinal_by_epoch: dict[int, int] = {}
    for item in result.opportunities:
        epoch = record_epochs[item.record_id]
        ordinal = ordinal_by_epoch.get(epoch, 0)
        ordinal_by_epoch[epoch] = ordinal + 1
        persistent = _persistent_candidates(item.candidates, known_record_ids=record_epochs)
        # Membership observations must be selected with pre-target information.
        # ``accepted_prefix_length`` is retained strictly as later reward, never
        # as an admission/ranking feature.
        selected = min(persistent, key=_persistent_candidate_rank, default=None)
        source_record_id = selected.source_record_id if selected is not None else item.record_id
        cheaper_prefix = max(item.context_prefix, item.current_source_prefix)
        observations.append(
            CapacityObservation(
                observation_id=f"{item.record_id}:{item.position}",
                epoch=epoch,
                ordinal=ordinal,
                bundle_id=source_record_id,
                unique_value=(
                    max(0, selected.accepted_prefix_length - cheaper_prefix)
                    if selected is not None
                    else 0
                ),
            )
        )
    simulator = CapacitySimulator(bundles, common_archive_bytes=common_archive_bytes)
    # The first ladder point can admit the smallest active bundle in main:
    # main has half of each logical view.  The final point can admit every
    # bundle, while retaining the fixed main/frontier/ghost partition.
    first_capacity_bytes = 2 * min(bundle.active_bytes for bundle in bundles)
    full_active_view_bytes = (4 * sum(bundle.active_bytes for bundle in bundles) + 2) // 3
    max_capacity_bytes = first_capacity_bytes
    while max_capacity_bytes < full_active_view_bytes:
        max_capacity_bytes *= 2
    simulations = {
        policy.value: simulator.run_geometric(
            observations,
            first_capacity_bytes=first_capacity_bytes,
            max_capacity_bytes=max_capacity_bytes,
            policy=policy,
        ).as_dict()
        for policy in (CapacityPolicy.HISTORICAL_VALUE_DENSITY, CapacityPolicy.RECENCY_WINDOW_CONTROL)
    }
    return {
        "kind": "offline_causal_capacity_simulator_no_runtime_timing",
        "fixed_stream_observation_count": len(observations),
        "capacity_ladder_bytes": {
            "first": first_capacity_bytes,
            "maximum": max_capacity_bytes,
            "selection_rule": "first admits the smallest bundle in main; maximum is a geometric point that admits every bundle across main+frontier",
        },
        "simulations": simulations,
        "strategy_summary": _capacity_strategy_summary(
            simulations,
            request_id_by_epoch={index: record.request_id for index, record in enumerate(result.records)},
        ),
    }


def _persistent_candidate_rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    """Stable pre-target rank shared by capacity-observation extraction only."""

    tier_rank = {
        SourceTier.SAME_SESSION: 0,
        SourceTier.SAME_REPOSITORY: 1,
        SourceTier.PERSONAL_PERSISTENT_HISTORY: 2,
        SourceTier.IMPORTED_CORPUS: 3,
    }
    return (
        -candidate.matched_suffix_length,
        tier_rank[candidate.tier],
        candidate.source_token_bytes,
        candidate.source_record_id,
        candidate.source_cursor,
    )


def _persistent_candidates(
    candidates: Sequence[Candidate], *, known_record_ids: Mapping[str, int] | None = None
) -> tuple[Candidate, ...]:
    """Keep only legal persistent tiers, optionally bound to this immutable archive."""

    persistent_tiers = {
        SourceTier.SAME_SESSION,
        SourceTier.SAME_REPOSITORY,
        SourceTier.PERSONAL_PERSISTENT_HISTORY,
        SourceTier.IMPORTED_CORPUS,
    }
    return tuple(
        candidate
        for candidate in candidates
        if candidate.tier in persistent_tiers
        and (known_record_ids is None or candidate.source_record_id in known_record_ids)
    )


def _capacity_strategy_summary(
    simulations: Mapping[str, Mapping[str, Any]], *, request_id_by_epoch: Mapping[int, str]
) -> dict[str, list[dict[str, Any]]]:
    """Compact value accounting for logical policies, without a runtime claim."""

    summary: dict[str, list[dict[str, Any]]] = {}
    for policy, simulation in sorted(simulations.items()):
        points: list[dict[str, Any]] = []
        for run in simulation["runs"]:
            totals = {
                field: sum(outcome[field] for outcome in run["outcomes"])
                for field in (
                    "main_unique_value",
                    "frontier_unique_value",
                    "ghost_unique_value",
                    "inactive_unique_value",
                )
            }
            total = sum(totals.values())
            active = totals["main_unique_value"] + totals["frontier_unique_value"]
            request_capture: dict[str, list[float]] = {}
            for outcome in run["outcomes"]:
                outcome_total = sum(
                    outcome[field]
                    for field in (
                        "main_unique_value",
                        "frontier_unique_value",
                        "ghost_unique_value",
                        "inactive_unique_value",
                    )
                )
                if outcome_total == 0:
                    continue
                request_id = request_id_by_epoch[outcome["epoch"]]
                request_capture.setdefault(request_id, []).append(
                    (outcome["main_unique_value"] + outcome["frontier_unique_value"]) / outcome_total
                )
            points.append(
                {
                    "capacity_bytes": run["capacity_bytes"],
                    **totals,
                    "total_unique_value": total,
                    "active_capture_ratio": active / total if total else 0.0,
                    "clustered_request_mean_active_capture_ci": (
                        _json_value(clustered_bootstrap_ci(request_capture, bootstrap_replicates=200, seed=0))
                        if request_capture
                        else None
                    ),
                }
            )
        curve = capacity_curve(
            CapacityValue(
                capacity_bytes=point["capacity_bytes"],
                practical_marginal_value=point["main_unique_value"] + point["frontier_unique_value"],
                full_archive_oracle_marginal_value=point["total_unique_value"],
            )
            for point in points
        )
        for point, curve_point in zip(points, curve, strict=True):
            point["marginal_value_rate"] = curve_point.marginal_value_rate
        summary[policy] = points
    return summary


def _scope_strategy_summary(experiments: Sequence[ScopeExperimentResult]) -> list[dict[str, Any]]:
    """Expose comparable, post-hoc scope outcomes without hiding the ranker gap."""

    global_result = next(result for result in experiments if result.policy.name == "global")
    summary: list[dict[str, Any]] = []
    for result in experiments:
        summary.append(
            {
                "name": result.policy.name,
                "byte_budget": result.policy.byte_budget,
                "probe_budget": result.policy.probe_budget,
                "selected_prefix_value": result.selected_prefix_value,
                "candidate_oracle_prefix_value": result.candidate_oracle_prefix_value,
                "ranking_gap": result.candidate_oracle_prefix_value - result.selected_prefix_value,
                "selected_prefix_delta_vs_global": result.selected_prefix_value - global_result.selected_prefix_value,
                "selected_positive_positions": result.selected_positive_positions,
                "zero_accept_positions": result.zero_accept_positions,
                "selected_source_tiers": {
                    tier.value: count for tier, count in result.selected_source_tiers
                },
            }
        )
    return summary


def _consensus_and_negative_evidence(result: ReplayResult) -> dict[str, Any]:
    """Score offline-only consensus and a recovering anti-cache over one stream."""

    records = {record.record_id: record for record in result.records}
    observations: list[ConsensusObservation] = []
    events: list[NegativeEvidenceEvent] = []
    independent_support_at_least_two = 0
    for ordinal, replay_opportunity in enumerate(result.opportunities):
        persistent = _persistent_candidates(replay_opportunity.candidates)
        if not persistent:
            continue
        consensus = consensus_by_provenance(persistent)
        target_tokens = records[replay_opportunity.record_id].committed_tokens[replay_opportunity.position :]
        accepted = continuation_prefix_length(target_tokens, consensus.consensus_prefix)
        observations.append(
            ConsensusObservation(
                score=consensus.support_shrunk_score,
                independent_support=consensus.independent_support,
                accepted_prefix_length=accepted,
            )
        )
        independent_support_at_least_two += consensus.independent_support >= 2
        selected = min(persistent, key=_persistent_candidate_rank)
        events.append(
            NegativeEvidenceEvent(
                key=f"{selected.source_record_id}:{selected.source_cursor}",
                chronological_index=ordinal,
                accepted_prefix_length=selected.accepted_prefix_length,
            )
        )
    points = simulate_decayed_negative_evidence(events)
    return {
        "kind": "offline_consensus_and_negative_evidence_no_runtime_control",
        "consensus": {
            "persistent_candidate_opportunities": len(observations),
            "independent_support_at_least_two": independent_support_at_least_two,
            "accepted_prefix_value": sum(item.accepted_prefix_length for item in observations),
            "calibration": _json_value(calibrate_consensus(observations, bins=5)),
            "boundary": "consensus is scored from prior continuations and target tokens only after the chronological position closes",
        },
        "negative_evidence": {
            "events": len(events),
            "zero_accept_events": sum(event.accepted_prefix_length == 0 for event in events),
            "suppressed_before_events": sum(point.suppressed_before for point in points),
            "configuration": {
                "half_life_events": 10.0,
                "suppress_at": 2.0,
                "max_entries": 128,
                "success_recovery_multiplier": 0.5,
            },
            "boundary": "offline recovering anti-cache simulation; no proposal was suppressed during target decoding",
        },
    }


def _payload(result: ReplayResult) -> dict[str, Any]:
    archive_validation = _archive_validation(result)
    opportunities = result.opportunities
    cluster_values = request_cluster_values(opportunities)
    # No provider/verifier outcome is supplied.  These zero-recovery values
    # retain the denominators while explicitly avoiding a fictional live run.
    empty_outcomes = ()
    scope_experiments = compare_scope_policies(result, byte_budget=4096, probe_budget=2)
    return {
        "kind": "model_free_chronological_replay",
        "runtime_claim": "none",
        "future_tokens_used_for_retrieval": result.future_tokens_used_for_retrieval,
        "trace_records": len(result.records),
        "proposal_positions": len(opportunities),
        "replay_census": _json_value(result.census),
        "oracle_metrics": {
            "opportunity": _json_value(opportunity(opportunities)),
            "accepted_prefix_value": accepted_prefix_value(opportunities),
            "chain_survival": _json_value(chain_survival(opportunities)),
            "source_attribution": _json_value(source_attribution(opportunities)),
            "orr": oracle_reuse_ratio(opportunities),
            "persistent_unique_opportunity": _json_value(
                opportunity(opportunities, persistent_unique=True)
            ),
            "persistent_unique_orr": oracle_reuse_ratio(opportunities, persistent_unique=True),
        },
        "clustered_confidence_interval": _json_value(
            clustered_bootstrap_ci(cluster_values, bootstrap_replicates=200, seed=0)
        ),
        "active_metrics": {
            "status": "NOT_RUN",
            "reason": "no live target decode, provider deadline, or verifier outcome was collected",
            "vorr_with_no_outcomes": _json_value(
                viable_opportunity_recovery_rate(opportunities, empty_outcomes)
            ),
            "acr_with_no_outcomes": _json_value(active_capture_ratio(opportunities, empty_outcomes)),
        },
        "scope_experiments": _json_value(scope_experiments),
        "scope_strategy_summary": _scope_strategy_summary(scope_experiments),
        "consensus_and_negative_evidence": _consensus_and_negative_evidence(result),
        "archive_validation": archive_validation,
        "capacity_simulator": _capacity_simulation(
            result, archive_validation["byte_accounting"]["total"]
        ),
    }


def execute_offline_fixture(
    *,
    fixture_path: Path,
    rfc_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    expected_rfc_sha256: str = PCSS_MASTER_RFC_SHA256,
) -> ResultReceipt:
    """Execute the host-independent replay and make one simulator receipt."""

    resolved_root = repository_root.resolve()
    rfc_fingerprint = _validate_rfc_source(rfc_path, expected_sha256=expected_rfc_sha256)
    records = load_trace_fixture(fixture_path)
    result = replay_chronologically(records)
    if result.future_tokens_used_for_retrieval:
        raise ContractError("offline replay future-leakage invariant failed")
    return make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.offline_runner", command_arguments, resolved_root),
        git_revision=resolve_git_revision(resolved_root),
        input_fingerprints={
            "rfc": rfc_fingerprint,
            "trace_fixture": file_sha256(fixture_path),
        },
        status="VERIFIED_OFFLINE",
        payload={
            "trace_stream_fingerprint": trace_stream_fingerprint(records),
            **_payload(result),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a model-free chronological PCSS replay and write a simulator receipt."
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--rfc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    receipt = execute_offline_fixture(
        fixture_path=args.fixture,
        rfc_path=args.rfc,
        repository_root=args.repository_root,
        command_arguments=command_arguments,
    )
    write_raw_result(args.output, receipt)
    if args.report is not None:
        write_compact_report(args.report, raw_result=args.output, title="PCSS offline chronological replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
