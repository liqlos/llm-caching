from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

from .adaptive_datastore import (
    K_VALUES,
    AdaptiveDatastore,
    PolicyName,
    Proposal,
    SequenceMode,
    SourceCondition,
    policy_density,
)
from .canonical import ContractError, canonical_json
from .metrics import clustered_bootstrap_ci
from .offline_runner import load_trace_fixture
from .replay import (
    Candidate,
    ReplayRecord,
    SOURCE_TIER_PRIORITY,
    SourceTier,
    group_opportunities_by_request,
    record_from_mapping,
    replay_chronologically,
)
from .results import file_sha256


STUDY_SCHEMA = "pcss.adaptive-datastore-study/v1"
PERSISTENT_TIERS = (
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
)
REDESIGNED_POLICIES = (
    PolicyName.DENSE_RECENT,
    PolicyName.DENSE_MODAL,
    PolicyName.DENSE_CONFIDENT,
    PolicyName.STRIDE_4_CONFIDENT,
    PolicyName.STRIDE_16_CONFIDENT,
)


@dataclass(frozen=True)
class SelectedProposal:
    condition: SourceCondition
    continuation: tuple[int, ...]
    suffix_length: int
    source_tier: SourceTier | None
    proposal: Proposal | None


def _json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _dataset_class(path: Path) -> str:
    name = str(path).casefold()
    if "chronological-oracle" in name:
        return "UNIT_SANITY"
    if "chronological" in name:
        return "UNIT_SANITY"
    if "cross-repo" in name:
        return "SYNTHETIC_RETRIEVAL_CONTROL"
    if "e1-synthetic" in name or "synthetic-prompts" in name or "repeat" in name:
        return "SYNTHETIC_MECHANISM_CONTROL"
    if "e2-cross-repo" in name or "cross-repository" in name:
        return "SYNTHETIC_RETRIEVAL_CONTROL"
    if "source-edit" in name or "diverse-edit" in name:
        return "SYNTHETIC_WORKLOAD_CALIBRATION"
    if "project" in name or "naturalistic" in name:
        return "AGENT_ASSISTED_CALIBRATION"
    return "IMPORTED_CORPUS"


def _git_head(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=repository_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_metadata(path: Path, records: Sequence[ReplayRecord]) -> dict[str, Any]:
    result = replay_chronologically(records, min_suffix=1, max_continuation=32)
    clusters = group_opportunities_by_request(result)
    fingerprints = {
        field: sorted({getattr(record.fingerprints, field) for record in records})
        for field in ("model", "tokenizer", "chat_template", "generation_protocol")
    }
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "evidence_class": _dataset_class(path),
        "schema_version": "pcss.trace/v1",
        "record_count": len(records),
        "total_prompt_tokens": sum(len(record.prompt_tokens) for record in records),
        "total_committed_output_tokens": sum(len(record.committed_tokens) for record in records),
        "independent_cluster_count": len(clusters),
        "fingerprints": fingerprints,
        "chronological": all(first.order_key < second.order_key for first, second in zip(records, records[1:])),
        "current_source_present": any(record.current_source_tokens is not None for record in records),
        "source_tokens_available_before_target": True,
        "persistent_eligible_records": sum(record.eligible_for_persistent_store for record in records),
        "synthetic": _dataset_class(path).startswith("SYNTHETIC"),
    }


def _candidate_rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    tier_rank = {
        SourceTier.CURRENT_CONTEXT: 0,
        SourceTier.CURRENT_SOURCE: 1,
        SourceTier.SAME_SESSION: 2,
        SourceTier.SAME_REPOSITORY: 3,
        SourceTier.PERSONAL_PERSISTENT_HISTORY: 4,
        SourceTier.IMPORTED_CORPUS: 5,
    }
    return (
        -candidate.matched_suffix_length,
        tier_rank[candidate.tier],
        candidate.source_token_bytes,
        candidate.source_record_id,
        candidate.source_cursor,
    )


def _current_proposal(
    condition: SourceCondition,
    candidates: Iterable[Candidate],
    k: int,
) -> SelectedProposal | None:
    tiers = {
        SourceCondition.CURRENT_CONTEXT_ONLY: (SourceTier.CURRENT_CONTEXT,),
        SourceCondition.CURRENT_SOURCE_ONLY: (SourceTier.CURRENT_SOURCE,),
    }.get(condition)
    if tiers is None:
        return None
    usable = tuple(candidate for candidate in candidates if candidate.tier in tiers)
    if not usable:
        return None
    selected = min(usable, key=_candidate_rank)
    return SelectedProposal(
        condition=condition,
        continuation=selected.continuation[:k],
        suffix_length=selected.matched_suffix_length,
        source_tier=selected.tier,
        proposal=None,
    )


def _persistent_tier(proposal: Proposal) -> SourceTier | None:
    if not proposal.matching_postings:
        return None
    return min(
        (posting.source_tier for posting in proposal.matching_postings),
        key=lambda tier: PERSISTENT_TIERS.index(tier) if tier in PERSISTENT_TIERS else len(PERSISTENT_TIERS),
    )


def _condition_proposal(
    condition: SourceCondition,
    current_context: SelectedProposal | None,
    current_source: SelectedProposal | None,
    persistent: SelectedProposal | None,
) -> SelectedProposal | None:
    by_tier = {
        SourceTier.CURRENT_CONTEXT: current_context,
        SourceTier.CURRENT_SOURCE: current_source,
        SourceTier.SAME_SESSION: persistent,
        SourceTier.SAME_REPOSITORY: persistent,
        SourceTier.PERSONAL_PERSISTENT_HISTORY: persistent,
        SourceTier.IMPORTED_CORPUS: persistent,
    }
    allowed = {
        SourceCondition.CURRENT_CONTEXT_ONLY: {SourceTier.CURRENT_CONTEXT},
        SourceCondition.CURRENT_SOURCE_ONLY: {SourceTier.CURRENT_SOURCE},
        SourceCondition.PERSISTENT_ONLY: set(PERSISTENT_TIERS),
        SourceCondition.CURRENT_CONTEXT_PLUS_PERSISTENT: {SourceTier.CURRENT_CONTEXT, *PERSISTENT_TIERS},
        SourceCondition.CURRENT_SOURCE_PLUS_PERSISTENT: {SourceTier.CURRENT_SOURCE, *PERSISTENT_TIERS},
        SourceCondition.ALL_LEGAL_TIERS: set(SOURCE_TIER_PRIORITY),
    }[condition]
    for tier in SOURCE_TIER_PRIORITY:
        if tier in allowed and by_tier.get(tier) is not None:
            return by_tier[tier]
    return None


def _accepted(future: Sequence[int], continuation: Sequence[int]) -> int:
    result = 0
    for expected, proposed in zip(future, continuation):
        if expected != proposed:
            break
        result += 1
    return result


def _empty_metrics() -> dict[str, Any]:
    return {
        "lookup_count": 0,
        "hit_count": 0,
        "hits_by_suffix_length": Counter(),
        "abstentions": Counter(),
        "proposals": 0,
        "proposed_tokens": 0,
        "accepted_zero_proposals": 0,
        "first_token_accepted_proposals": 0,
        "fully_accepted_proposals": 0,
        "accepted_token_count": 0,
        "persistent_unique_accepted_tokens": 0,
        "all_position_persistent_unique_accepted_tokens": 0,
        "online_persistent_unique_accepted_tokens": 0,
        "accepted_lengths": [],
        "match_lengths": [],
        "support": [],
        "all_position_accepted": 0,
        "all_position_proposals": 0,
        "online_steps": 0,
        "online_accepted_tokens": 0,
        "online_proposals": 0,
        "online_accepted_zero": 0,
        "persistent_tiers": Counter(),
        "clusters": defaultdict(lambda: {"all_position_accepted": 0, "all_position_proposals": 0, "online_accepted_tokens": 0, "online_steps": 0, "online_proposals": 0, "online_zero_accept_proposals": 0}),
    }


def _finalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    proposals = metrics["proposals"]
    hits = metrics["hit_count"]
    result = dict(metrics)
    result["hits_by_suffix_length"] = dict(sorted(metrics["hits_by_suffix_length"].items()))
    result["abstentions"] = dict(sorted(metrics["abstentions"].items()))
    result["persistent_tiers"] = {key.value: value for key, value in sorted(metrics["persistent_tiers"].items(), key=lambda item: item[0].value)}
    result["clusters"] = {key: value for key, value in sorted(metrics["clusters"].items())}
    result["accepted_tokens_per_proposal"] = metrics["accepted_token_count"] / proposals if proposals else 0.0
    result["accepted_tokens_per_hit"] = metrics["accepted_token_count"] / hits if hits else 0.0
    result["accepted_output_token_coverage"] = metrics["accepted_token_count"] / metrics["output_tokens"] if metrics.get("output_tokens") else 0.0
    result["persistent_unique_accepted_tokens"] = metrics["online_persistent_unique_accepted_tokens"]
    result["persistent_unique_coverage"] = metrics["online_persistent_unique_accepted_tokens"] / metrics["output_tokens"] if metrics.get("output_tokens") else 0.0
    result["all_position_persistent_unique_coverage"] = metrics["all_position_persistent_unique_accepted_tokens"] / metrics["output_tokens"] if metrics.get("output_tokens") else 0.0
    result["zero_accept_proposal_rate"] = metrics["accepted_zero_proposals"] / proposals if proposals else 0.0
    result["match_length_distribution"] = dict(Counter(metrics["match_lengths"]))
    result["support_distribution"] = metrics["support"]
    result["accepted_lengths"] = metrics["accepted_lengths"]
    return result


def _record_event(
    metrics: dict[str, Any],
    selected: SelectedProposal | None,
    record: ReplayRecord,
    position: int,
    *,
    context_accepted: int,
    source_accepted: int,
    persistent: SelectedProposal | None,
    online: bool,
) -> int:
    future = record.committed_tokens[position:]
    accepted = _accepted(future, selected.continuation) if selected else 0
    cluster = record.request_id or record.record_id
    if selected is not None and not online:
        metrics["proposals"] += 1
        metrics["proposed_tokens"] += len(selected.continuation)
        metrics["accepted_lengths"].append(accepted)
        metrics["all_position_proposals"] += 1
        if selected.suffix_length:
            metrics["match_lengths"].append(selected.suffix_length)
        if accepted == 0:
            metrics["accepted_zero_proposals"] += 1
            metrics["abstentions"]["posthoc_zero_accept"] += 1
        else:
            metrics["accepted_token_count"] += accepted
            metrics["all_position_accepted"] += accepted
            metrics["first_token_accepted_proposals"] += 1
            if accepted >= len(selected.continuation):
                metrics["fully_accepted_proposals"] += 1
        metrics["clusters"][cluster]["all_position_proposals"] += 1
        metrics["clusters"][cluster]["all_position_accepted"] += accepted
    if persistent is not None and selected is persistent and not online:
        metrics["persistent_tiers"][persistent.source_tier] += accepted
        metrics["all_position_persistent_unique_accepted_tokens"] += max(0, accepted - max(context_accepted, source_accepted))
    if online:
        metrics["online_steps"] += 1
        metrics["clusters"][cluster]["online_steps"] += 1
        if selected is not None:
            metrics["online_proposals"] += 1
            metrics["clusters"][cluster]["online_proposals"] += 1
            if accepted == 0:
                metrics["online_accepted_zero"] += 1
                metrics["clusters"][cluster]["online_zero_accept_proposals"] += 1
            metrics["online_accepted_tokens"] += accepted
            metrics["clusters"][cluster]["online_accepted_tokens"] += accepted
            if persistent is not None and selected is persistent:
                metrics["online_persistent_unique_accepted_tokens"] += max(0, accepted - max(context_accepted, source_accepted))
    return max(1, accepted)


def _lookup_latency(store: AdaptiveDatastore, target: ReplayRecord, query: tuple[int, ...], policy: PolicyName, k: int) -> tuple[Proposal | None, int]:
    started = perf_counter_ns()
    proposal = store.lookup(target, query, k=k, policy=policy)
    return proposal, perf_counter_ns() - started


def run_configuration(
    records: Sequence[ReplayRecord],
    policy: PolicyName,
    sequence_mode: SequenceMode,
    k: int,
    opportunities: Mapping[tuple[str, int], Any],
    verifier_costs: Mapping[str, Any] | None = None,
    economic_bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    density = policy_density(policy)
    store = AdaptiveDatastore(sequence_mode, density)
    conditions = tuple(SourceCondition)
    events: dict[str, dict[int, dict[SourceCondition, tuple[SelectedProposal | None, int]]]] = defaultdict(dict)
    metrics = {condition: _empty_metrics() for condition in conditions}
    latencies: list[int] = []
    growth: list[dict[str, Any]] = []
    cumulative_source_tokens = 0
    for record_index, record in enumerate(records):
        for position in range(len(record.committed_tokens)):
            query = record.prompt_tokens + record.committed_tokens[:position]
            persistent, latency = _lookup_latency(store, record, query, policy, 2 if policy == PolicyName.LEGACY_SPARSE else k)
            latencies.append(latency)
            opportunity = opportunities[(record.record_id, position)]
            context = _current_proposal(SourceCondition.CURRENT_CONTEXT_ONLY, opportunity.candidates, k)
            source = _current_proposal(SourceCondition.CURRENT_SOURCE_ONLY, opportunity.candidates, k)
            persistent_selected = None
            if persistent is not None:
                persistent_selected = SelectedProposal(
                    condition=SourceCondition.PERSISTENT_ONLY,
                    continuation=persistent.continuation,
                    suffix_length=persistent.suffix_length,
                    source_tier=_persistent_tier(persistent),
                    proposal=persistent,
                )
            row: dict[SourceCondition, tuple[SelectedProposal | None, int]] = {}
            context_accepted = _accepted(record.committed_tokens[position:], context.continuation) if context else 0
            source_accepted = _accepted(record.committed_tokens[position:], source.continuation) if source else 0
            for condition in conditions:
                selected = _condition_proposal(condition, context, source, persistent_selected)
                row[condition] = (selected, 0)
                metrics[condition]["output_tokens"] = metrics[condition].get("output_tokens", 0) + 1
                if condition in (SourceCondition.PERSISTENT_ONLY, SourceCondition.CURRENT_CONTEXT_PLUS_PERSISTENT, SourceCondition.CURRENT_SOURCE_PLUS_PERSISTENT, SourceCondition.ALL_LEGAL_TIERS):
                    metrics[condition]["lookup_count"] += 1
                    if persistent is not None:
                        metrics[condition]["hit_count"] += 1
                        metrics[condition]["hits_by_suffix_length"][persistent.suffix_length] += 1
                        if persistent.abstention_reason:
                            metrics[condition]["abstentions"][persistent.abstention_reason] += 1
                    elif selected is None:
                        metrics[condition]["abstentions"]["miss"] += 1
                if selected is not None and selected.proposal is not None:
                    for support in selected.proposal.supports:
                        metrics[condition]["support"].append(_json(support))
                _record_event(
                    metrics[condition],
                    selected,
                    record,
                    position,
                    context_accepted=context_accepted,
                    source_accepted=source_accepted,
                    persistent=persistent_selected,
                    online=False,
                )
            events[record.record_id][position] = row
        ingest = store.ingest(record)
        cumulative_source_tokens += len(store._sequence(record))
        persistent_metrics = metrics[SourceCondition.PERSISTENT_ONLY]
        growth.append({
            "records_seen": record_index + 1,
            "record_id": record.record_id,
            "cumulative_source_tokens": cumulative_source_tokens,
            "cumulative_persistent_hits": persistent_metrics["hit_count"],
            "cumulative_persistent_accepted_tokens": persistent_metrics["accepted_token_count"],
            "cumulative_all_position_persistent_unique_accepted_tokens": persistent_metrics["all_position_persistent_unique_accepted_tokens"],
            "accounting": _json(store.accounting()),
            "ingest": _json(ingest),
        })
    for record in records:
        for condition in conditions:
            metrics[condition]["output_tokens"] = len(record.committed_tokens) + metrics[condition].get("output_tokens", 0) - len(record.committed_tokens)
            position = 0
            while position < len(record.committed_tokens):
                selected = events[record.record_id][position][condition][0]
                context = events[record.record_id][position][SourceCondition.CURRENT_CONTEXT_ONLY][0]
                source = events[record.record_id][position][SourceCondition.CURRENT_SOURCE_ONLY][0]
                context_accepted = _accepted(record.committed_tokens[position:], context.continuation) if context else 0
                source_accepted = _accepted(record.committed_tokens[position:], source.continuation) if source else 0
                progress = _record_event(
                    metrics[condition],
                    selected,
                    record,
                    position,
                    context_accepted=context_accepted,
                    source_accepted=source_accepted,
                    persistent=events[record.record_id][position][SourceCondition.PERSISTENT_ONLY][0],
                    online=True,
                )
                position += progress
    for condition in conditions:
        metrics[condition]["output_tokens"] = sum(len(record.committed_tokens) for record in records)
    accounting = store.accounting()
    boundary_targets = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
    growth_boundaries = []
    for fraction in boundary_targets:
        target_count = round(fraction * len(records))
        target_count = max(0, min(len(records), target_count))
        selected_growth = growth[target_count - 1] if target_count else {"records_seen": 0, "accounting": _json(AdaptiveDatastore(sequence_mode, density).accounting())}
        growth_boundaries.append({"requested_fraction": fraction, "records_seen": selected_growth["records_seen"], "actual_fraction": selected_growth["records_seen"] / len(records) if records else 0.0, "accounting": selected_growth["accounting"]})
    warm = latencies[5:] if len(latencies) > 5 else latencies
    def percentile(values: Sequence[int], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
        return ordered[index] / 1000.0
    condition_output = {}
    for condition, value in metrics.items():
        finalized = _finalize_metrics(value)
        finalized["condition"] = condition.value
        finalized["all_position_metrics"] = {"accepted_tokens": value["all_position_accepted"], "proposals": value["all_position_proposals"]}
        finalized["online_step_metrics"] = {"accepted_tokens": value["online_accepted_tokens"], "proposals": value["online_proposals"], "steps": value["online_steps"], "zero_accept_proposals": value["online_accepted_zero"]}
        condition_output[condition.value] = finalized
    tree = {"tree_hit_positions": 0, "target_path_tokens_found": 0, "persistent_unique_tree_coverage": 0, "positive_clusters": 0, "linear_vs_tree_accepted_token_delta": 0, "branching_loss_tokens": 0, "tree_nodes": 0}
    positive_clusters: set[str] = set()
    for record in records:
        for position in range(len(record.committed_tokens)):
            selected = events[record.record_id][position][SourceCondition.PERSISTENT_ONLY][0]
            if selected is None or selected.proposal is None or selected.proposal.policy != PolicyName.DENSE_MODAL:
                continue
            diagnostic = AdaptiveDatastore(sequence_mode, density)
            tree_root = diagnostic.build_multibranch_tree(selected.proposal)
            score = diagnostic.score_multibranch(selected.proposal, tree_root, record.committed_tokens[position:], linear_accepted_tokens=_accepted(record.committed_tokens[position:], selected.continuation))
            tree["tree_hit_positions"] += score.tree_hit
            tree["target_path_tokens_found"] += score.target_path_tokens_found
            tree["persistent_unique_tree_coverage"] += score.persistent_unique_tree_coverage
            tree["linear_vs_tree_accepted_token_delta"] += score.tree_accepted_tokens - score.linear_accepted_tokens
            tree["branching_loss_tokens"] += score.branching_loss_tokens
            tree["tree_nodes"] += score.tree_nodes
            if score.tree_hit:
                positive_clusters.add(record.request_id or record.record_id)
    tree["positive_clusters"] = len(positive_clusters)
    return {
        "policy": policy.value,
        "sequence_mode": sequence_mode.value,
        "density": density.value,
        "k": 2 if policy == PolicyName.LEGACY_SPARSE else k,
        "storage": _json(accounting),
        "growth": growth,
        "growth_boundaries": growth_boundaries,
        "source_conditions": condition_output,
        "python_prototype_lookup_latency_us": {"label": "PYTHON_PROTOTYPE_ONLY", "warmup_calls": len(latencies) - len(warm), "p50": percentile(warm, 0.50), "p95": percentile(warm, 0.95), "p99": percentile(warm, 0.99)},
        "multibranch_diagnostic_only": tree,
        "future_tokens_used_for_retrieval": False,
        "economic_sensitivity": _economic_sensitivity(
            metrics[SourceCondition.PERSISTENT_ONLY],
            2 if policy == PolicyName.LEGACY_SPARSE else k,
            verifier_costs,
            bootstrap_resamples=economic_bootstrap_resamples,
        ),
    }


def configurations() -> tuple[tuple[PolicyName, SequenceMode, int], ...]:
    values: list[tuple[PolicyName, SequenceMode, int]] = [(PolicyName.LEGACY_SPARSE, SequenceMode.PROMPT_PLUS_COMPLETION, 2)]
    for policy in REDESIGNED_POLICIES:
        for mode in SequenceMode:
            for k in K_VALUES:
                values.append((policy, mode, k))
    return tuple(values)


def _config_key(value: Mapping[str, Any]) -> tuple[str, str, int]:
    return (str(value["policy"]), str(value["sequence_mode"]), int(value["k"]))


def _aggregate_point(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [result["source_conditions"][SourceCondition.PERSISTENT_ONLY.value] for result in results]
    accepted = sum(float(value["accepted_token_count"]) for value in values)
    output = sum(float(value["output_tokens"]) for value in values)
    unique = sum(float(value["persistent_unique_accepted_tokens"]) for value in values)
    speeds = []
    speed_lcbs = []
    for result in results:
        sensitivity = result.get("economic_sensitivity", {})
        row = sensitivity.get("lookup_budgets_us", {}).get("1000") if isinstance(sensitivity, Mapping) else None
        if isinstance(row, Mapping) and isinstance(row.get("point_estimate"), (int, float)):
            speeds.append(float(row["point_estimate"]))
        interval = row.get("cluster_bootstrap_95") if isinstance(row, Mapping) else None
        if isinstance(interval, Mapping) and isinstance(interval.get("lower"), (int, float)):
            speed_lcbs.append(float(interval["lower"]))
    median_speed = median(speeds) if speeds else 0.0
    return {
        "accepted_persistent_tokens": accepted,
        "persistent_unique_accepted_tokens": unique,
        "persistent_unique_coverage": unique / output if output else 0.0,
        "zero_accept_rate": sum(float(value["zero_accept_proposal_rate"]) for value in values) / len(values) if values else 0.0,
        "median_conservative_speed_ratio": median(speed_lcbs) if speed_lcbs else median_speed,
        "median_speed_ratio": median_speed,
    }


def _candidate_order(results_by_config: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]]) -> list[tuple[str, str, int]]:
    ranked: list[tuple[tuple[Any, ...], tuple[str, str, int]]] = []
    for key, results in results_by_config.items():
        if key[0] == PolicyName.LEGACY_SPARSE.value:
            continue
        non_control = [result for result in results if result["dataset"]["evidence_class"] not in {"UNIT_SANITY", "SYNTHETIC_MECHANISM_CONTROL", "SYNTHETIC_RETRIEVAL_CONTROL"}]
        point = _aggregate_point(non_control or results)
        storage = sum(int(result["storage"]["packed_ram_bytes"]) for result in results)
        disk = sum(int(result["storage"]["total_logical_bytes"]) for result in results)
        ranked.append(((-point["median_conservative_speed_ratio"], -point["persistent_unique_coverage"], point["zero_accept_rate"], storage, disk, key[2], key[0]), key))
    ranked.sort()
    return [key for _, key in ranked]


def _selection_summary(
    results_by_config: Mapping[tuple[str, str, int], Sequence[Mapping[str, Any]]],
    datasets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    control_paths = {"e1": [], "e2": []}
    for dataset in datasets:
        path = str(dataset["path"]).casefold()
        if "e1-synthetic" in path:
            control_paths["e1"].append(dataset["path"])
        if "e2-cross-repo" in path:
            control_paths["e2"].append(dataset["path"])
    control_gate = {"E1_persistent_positive": False, "E2_persistent_unique_positive": False}
    eligible: list[tuple[str, str, int]] = []
    rejected: dict[str, list[str]] = {}
    for key, results in results_by_config.items():
        if key[0] == PolicyName.LEGACY_SPARSE.value:
            continue
        by_path = {str(result["dataset"]["path"]): result for result in results}
        e1 = [by_path[path] for path in control_paths["e1"] if path in by_path]
        e2 = [by_path[path] for path in control_paths["e2"] if path in by_path]
        e1_positive = any(result["source_conditions"][SourceCondition.PERSISTENT_ONLY.value]["accepted_token_count"] > 0 for result in e1)
        e2_positive = any(result["source_conditions"][SourceCondition.PERSISTENT_ONLY.value]["persistent_unique_accepted_tokens"] > 0 for result in e2)
        if e1_positive:
            control_gate["E1_persistent_positive"] = True
        if e2_positive:
            control_gate["E2_persistent_unique_positive"] = True
        if not e1_positive:
            rejected[str(key)] = ["E1_positive_control"]
            continue
        if not e2_positive:
            rejected[str(key)] = ["E2_positive_control"]
            continue
        non_control = [result for result in results if result["dataset"]["evidence_class"] not in {"UNIT_SANITY", "SYNTHETIC_MECHANISM_CONTROL", "SYNTHETIC_RETRIEVAL_CONTROL"}]
        if non_control and not any(result["source_conditions"][SourceCondition.PERSISTENT_ONLY.value]["persistent_unique_accepted_tokens"] > 0 for result in non_control):
            rejected[str(key)] = ["zero_non_control_persistent_unique"]
            continue
        ram = max((int(result["storage"]["packed_ram_bytes"]) for result in results), default=0)
        bounded_ram = max((int(result["storage"]["bounded_ram_index_bytes"]) for result in results), default=0)
        disk = max((int(result["storage"]["total_logical_bytes"]) for result in results), default=0)
        projected_ram = (ram * 200 * 1024**3 / disk) if disk else 0
        bounded_on_disk = all(int(result["storage"]["on_disk_index_bytes"]) >= 0 for result in results)
        if (ram > 256 * 1024**2 or projected_ram > 256 * 1024**2) and (not bounded_on_disk or bounded_ram > 256 * 1024**2):
            rejected[str(key)] = ["storage_projection_over_256MiB"]
            continue
        eligible.append(key)
    ordered = _candidate_order({key: results_by_config[key] for key in eligible})
    return {
        "positive_controls": control_gate,
        "eligible_candidates": [_json(key) for key in ordered],
        "rejected_candidates": rejected,
        "selected_candidate": _json(ordered[0]) if ordered else None,
        "selection_rule": "positive controls, non-control persistent-unique signal, 256 MiB RAM and linear 200 GiB SSD projection, then median speed if available, coverage, zero-accept rate, RAM, SSD, K, lexical policy",
        "selection_status": "SELECTED" if ordered else "NO_ELIGIBLE_CANDIDATE",
    }


def load_verifier_costs(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PROFILE_VALID" or receipt.get("evidence_class") != "LOCAL_MEASURED":
        raise ContractError("verifier profile is not a valid local measurement")
    raw = receipt.get("payload", {}).get("raw_profile", {})
    rows = raw.get("profiles")
    if not isinstance(rows, list) or not rows:
        raise ContractError("verifier profile has no raw K rows")
    result: dict[str, Any] = {}
    for row in rows:
        k = row.get("k")
        measurements = row.get("measurements")
        if not isinstance(k, int) or not isinstance(measurements, list) or not measurements:
            raise ContractError("verifier profile row is invalid")
        serial_per_token = [((float(item["serial_before_ns"]) + float(item["serial_after_ns"])) / 2.0) / k for item in measurements]
        verification = [float(item["batch_verify_ns"]) for item in measurements]
        result[str(k)] = {"k": k, "D_ns_per_token": sum(serial_per_token) / len(serial_per_token), "V_ns": sum(verification) / len(verification), "measurements": len(measurements)}
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "host_identity": receipt.get("payload", {}).get("host_identity"),
        "model_artifact_sha256": receipt.get("input_fingerprints", {}).get("model_artifact"),
        "input_fingerprints": receipt.get("input_fingerprints", {}),
        "backend": raw.get("backend"),
        "llama_cpp_revision": raw.get("configuration", {}).get("llama_cpp_revision"),
        "runtime": raw.get("configuration", {}).get("runtime"),
        "rows": result,
    }


def _verifier_compatibility(records: Sequence[ReplayRecord], costs: Mapping[str, Any]) -> dict[str, Any]:
    trace_models = sorted({record.fingerprints.model.removeprefix("sha256:") for record in records})
    profile_model = str(costs.get("model_artifact_sha256") or "").removeprefix("sha256:")
    if not profile_model or trace_models != [profile_model]:
        raise ContractError("verifier profile model fingerprint does not match every trace")
    return {
        "status": "COMPATIBLE_MODEL_HASH",
        "trace_model_sha256": profile_model,
        "profile_model_sha256": profile_model,
        "backend": costs.get("backend"),
        "llama_cpp_revision": costs.get("llama_cpp_revision"),
        "host_identity": costs.get("host_identity"),
        "input_fingerprints": costs.get("input_fingerprints"),
    }


def _economic_sensitivity(
    metrics: Mapping[str, Any],
    k: int,
    costs: Mapping[str, Any] | None,
    *,
    bootstrap_resamples: int = 0,
) -> dict[str, Any]:
    if costs is None or str(k) not in costs["rows"]:
        return {"status": "NOT_COMPUTED", "reason": "no compatible verifier row for this K"}
    target = costs["rows"][str(k)]
    d_ns = float(target["D_ns_per_token"])
    verification_ns = float(target["V_ns"])
    lookup_costs_us = (0, 10, 50, 100, 250, 500, 1000)
    cluster_rows = metrics["clusters"]
    output: dict[str, Any] = {"status": "MEASURED_COMPATIBLE_TARGET_COSTS", "D_ns_per_token": d_ns, "V_ns": verification_ns, "lookup_budgets_us": {}}
    for lookup_us in lookup_costs_us:
        lookup_ns = lookup_us * 1000.0
        cost_components: dict[str, tuple[float, ...]] = {}
        baseline_total = 0.0
        pcss_total = 0.0
        for cluster, row in cluster_rows.items():
            steps = float(row["online_steps"])
            proposals = float(row["online_proposals"])
            accepted = float(row["online_accepted_tokens"])
            zero = float(row["online_zero_accept_proposals"])
            progress = steps - proposals + accepted + zero
            baseline = progress * d_ns
            pcss = (steps - proposals) * d_ns + proposals * (lookup_ns + verification_ns)
            baseline_total += baseline
            pcss_total += pcss
            cost_components[cluster] = (baseline, pcss)
        interval = (
            clustered_bootstrap_ci(
                cost_components,
                statistic=lambda values: sum(values[0::2]) / sum(values[1::2]) if sum(values[1::2]) else 0.0,
                bootstrap_replicates=bootstrap_resamples,
                seed=lookup_us + k,
            )
            if cost_components and bootstrap_resamples > 0
            else None
        )
        break_even = max(1, int((lookup_ns + verification_ns + d_ns - 1) // d_ns))
        output["lookup_budgets_us"][str(lookup_us)] = {"point_estimate": baseline_total / pcss_total if pcss_total else 0.0, "cluster_bootstrap_95": _json(interval) if interval else None, "break_even_accepted_prefix_length": break_even, "rounding": "ceil((L+V(K))/D), minimum 1"}
    return output


def run_study(
    trace_paths: Sequence[Path],
    *,
    seed: int,
    bootstrap_resamples: int,
    phase: str,
    repository_root: Path,
    verifier_profile: Path | None = None,
) -> dict[str, Any]:
    verifier_costs = load_verifier_costs(verifier_profile) if verifier_profile is not None else None
    datasets = []
    for path in trace_paths:
        records = tuple(record_from_mapping(item) for item in load_trace_fixture(path))
        datasets.append((path, records, dataset_metadata(path, records)))
    verifier_compatibility = (
        _verifier_compatibility(
            tuple(
                record
                for _, records, metadata in datasets
                if metadata["evidence_class"] != "UNIT_SANITY"
                for record in records
            ),
            verifier_costs,
        )
        if verifier_costs is not None
        else {"status": "NOT_COMPUTED"}
    )
    grid_results: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for path, records, metadata in datasets:
        replay = replay_chronologically(records, min_suffix=1, max_continuation=32)
        opportunities = {(item.record_id, item.position): item for item in replay.opportunities}
        for policy, mode, k in configurations():
            result = run_configuration(records, policy, mode, k, opportunities, verifier_costs, 0)
            result["dataset"] = metadata
            grid_results[(policy.value, mode.value, int(result["k"]))].append(result)
    all_keys = list(grid_results)
    selection_details = _selection_summary(grid_results, [metadata for _, _, metadata in datasets])
    ordered = [tuple(value) for value in selection_details["eligible_candidates"]]
    selected_keys = all_keys
    if phase in {"SHORTLIST_PASS", "FINAL_PASS"}:
        if phase == "SHORTLIST_PASS":
            selected_keys = [key for key in ordered[:3]] + [key for key in all_keys if key[0] == PolicyName.LEGACY_SPARSE.value]
        else:
            selected = ordered[0] if ordered else None
            dense = next((key for key in ordered if key[0] in {PolicyName.DENSE_MODAL.value, PolicyName.DENSE_CONFIDENT.value}), None)
            selected_keys = [key for key in ([selected, dense] if selected is not None else []) if key is not None]
            selected_keys.extend(key for key in all_keys if key[0] == PolicyName.LEGACY_SPARSE.value)
            selected_keys = list(dict.fromkeys(selected_keys))
        selected_keys = list(dict.fromkeys(selected_keys))
        for key in selected_keys:
            policy = PolicyName(key[0])
            mode = SequenceMode(key[1])
            k = int(key[2])
            for dataset_index, (path, records, metadata) in enumerate(datasets):
                replay = replay_chronologically(records, min_suffix=1, max_continuation=32)
                opportunities = {(item.record_id, item.position): item for item in replay.opportunities}
                result = run_configuration(
                    records,
                    policy,
                    mode,
                    k,
                    opportunities,
                    verifier_costs,
                    bootstrap_resamples,
                )
                result["dataset"] = metadata
                grid_results[key][dataset_index] = result
    selected_results = [result for key in selected_keys for result in grid_results.get(key, ())]
    return {
        "schema_version": STUDY_SCHEMA,
        "status": "OBSERVATION_DIRTY_WORKTREE",
        "phase": phase,
        "fixed_seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "repository_root": str(repository_root.resolve()),
        "git_revision": _git_head(repository_root),
        "datasets": [metadata for _, _, metadata in datasets],
        "configuration_count": len(configurations()),
        "source_tier_priority": [tier.value for tier in SOURCE_TIER_PRIORITY],
        "persistent_tiers": [tier.value for tier in PERSISTENT_TIERS],
        "policy_matrix": [_json({"policy": policy.value, "sequence_mode": mode.value, "k": k}) for policy, mode, k in configurations()],
        "future_tokens_used_for_retrieval": False,
        "selection": {**selection_details, "candidate_keys": [_json(key) for key in selected_keys], "post_result_tuning": False, "economic_measurement_status": "MEASURED_COMPATIBLE_TARGET_COSTS" if verifier_costs else "NOT_COMPUTED"},
        "verifier_costs": verifier_costs,
        "verifier_compatibility": verifier_compatibility,
        "results": selected_results,
        "cluster_bootstrap": {
            "status": "NOT_COMPUTED",
            "reason": "heterogeneous configurations, source conditions, and datasets have no single request-cluster estimand",
        },
        "claim_boundary": "Offline legal retrieval and packed-storage opportunity study; no native SSD timing or general speedup claim.",
    }


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise ContractError(f"output path already exists; choose a fresh path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fixed adaptive PCSS datastore study matrix.")
    parser.add_argument("--trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--bootstrap-resamples", type=int, default=0)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("GRID_PASS", "SHORTLIST_PASS", "FINAL_PASS"), default="GRID_PASS")
    parser.add_argument("--verifier-profile", type=Path)
    args = parser.parse_args(argv)
    payload = run_study(
        args.trace,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        phase=args.phase,
        repository_root=args.repository_root,
        verifier_profile=args.verifier_profile,
    )
    raw = canonical_json(payload) + "\n"
    _write_new(args.output, raw)
    digest = file_sha256(args.output)
    summaries = []
    for result in payload["results"]:
        persistent = result["source_conditions"][SourceCondition.PERSISTENT_ONLY.value]
        summaries.append(
            f"| `{result['dataset']['path']}` | `{result['policy']}` | `{result['sequence_mode']}` | {result['k']} | {persistent['accepted_token_count']} | {persistent['persistent_unique_accepted_tokens']} | {result['storage']['packed_ram_bytes']} | {result['storage']['total_logical_bytes']} |"
        )
    report = "\n".join(
        [
            "# Adaptive datastore study",
            "",
            f"- Phase: `{payload['phase']}`",
            f"- Status: `{payload['status']}`",
            f"- Raw result: `{args.output.name}` (`sha256:{digest}`)",
            f"- Configurations in fixed matrix: `{payload['configuration_count']}`; evaluated result rows: `{len(payload['results'])}`",
            "- Target future tokens were used only for post-hoc acceptance scoring.",
            f"- Economic sensitivity: `{payload['selection']['economic_measurement_status']}`; bootstrap resamples are applied only to shortlisted/final candidates.",
            "",
            "| Dataset | Policy | Mode | K | Persistent accepted | Persistent-unique accepted | Packed RAM bytes | Logical bytes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            *summaries,
            "",
            "The raw result contains all source conditions, all-position and non-overlapping online-step metrics, growth boundaries, fanout/storage accounting, and the multibranch diagnostic-only fields.",
        ]
    ) + "\n"
    _write_new(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
