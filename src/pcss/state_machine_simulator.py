"""Fast deterministic PCSS protocol simulator over a chronological replay.

The simulator executes the operational state machine without loading a model:
``idle -> lookup -> verify -> feedback -> lookup``.  Candidate selection uses
only completed records and suffix/provenance metadata.  Recorded target tokens
are consulted only by the verification transition to score the selected draft.
Costs are declared abstract units, never measured latency.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import ContractError, canonical_json
from .offline_runner import load_trace_fixture
from .replay import Candidate, Opportunity, ReplayRecord, ReplayResult, SourceTier, longest_exact_suffix, replay_chronologically


class MachineState(StrEnum):
    IDLE = "idle"
    LOOKUP = "lookup"
    VERIFY = "verify"
    FEEDBACK = "feedback"
    ADMIT = "admit"
    EVICT = "evict"


_PERSISTENT_TIERS = frozenset({
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
})
_TIER_RANK = {
    SourceTier.SAME_SESSION: 0,
    SourceTier.SAME_REPOSITORY: 1,
    SourceTier.PERSONAL_PERSISTENT_HISTORY: 2,
    SourceTier.IMPORTED_CORPUS: 3,
}


@dataclass(frozen=True)
class SimulationCosts:
    """Explicit abstract costs for comparing protocol choices, not timings."""

    serial_token: int = 10
    lookup: int = 1
    verify_batch: int = 4
    verify_token: int = 1
    correction: int = 2

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()) or self.serial_token < 1:
            raise ContractError("simulation costs must be non-negative and serial_token positive")


@dataclass(frozen=True)
class SimulationConfig:
    k: int = 2
    max_history_bytes: int | None = None
    costs: SimulationCosts = SimulationCosts()

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ContractError("simulation K must be positive")
        if self.max_history_bytes is not None and self.max_history_bytes < 0:
            raise ContractError("simulation max_history_bytes must be non-negative")


def _candidate_rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    """Rank pre-target evidence only; acceptance is deliberately absent."""

    return (
        -candidate.matched_suffix_length,
        _TIER_RANK[candidate.tier],
        candidate.source_token_bytes,
        candidate.source_record_id,
        candidate.source_cursor,
    )


def _legal_completed_candidate(
    candidate: Candidate,
    *,
    target: ReplayRecord,
    source: ReplayRecord | None,
    position: int,
) -> bool:
    """Re-bind derived opportunity metadata to a completed immutable source."""

    if source is None or source.order_key >= target.order_key:
        return False
    if not source.eligible_for_persistent_store or source.fingerprints != target.fingerprints:
        return False
    if (
        source.scope.principal_namespace != target.scope.principal_namespace
        or source.scope.application_id != target.scope.application_id
        or candidate.source_scope != source.scope
        or candidate.provenance_family != source.provenance_family
        or candidate.source_class != source.source_class
        or candidate.source_token_bytes != source.logical_source_bytes
    ):
        return False
    if candidate.source_cursor < 0 or candidate.source_cursor >= len(source.committed_tokens):
        return False
    if candidate.continuation != source.committed_tokens[
        candidate.source_cursor : candidate.source_cursor + len(candidate.continuation)
    ] or not candidate.continuation:
        return False
    if candidate.matched_suffix_length != longest_exact_suffix(
        target.prompt_tokens + target.committed_tokens[:position],
        source.committed_tokens,
        candidate.source_cursor,
    ):
        return False
    expected_tier = (
        SourceTier.IMPORTED_CORPUS if "import" in source.source_class.casefold()
        else SourceTier.CURRENT_SOURCE if target.scope.same_current_source(source.scope)
        else SourceTier.SAME_SESSION if target.scope.same_session(source.scope)
        else SourceTier.SAME_REPOSITORY if target.scope.same_repository(source.scope)
        else SourceTier.PERSONAL_PERSISTENT_HISTORY
    )
    return candidate.tier == expected_tier


def _select(
    opportunity: Opportunity,
    *,
    target: ReplayRecord,
    records: Mapping[str, ReplayRecord],
    active_record_ids: set[str],
) -> Candidate | None:
    legal = tuple(
        candidate
        for candidate in opportunity.candidates
        if (
            candidate.tier in _PERSISTENT_TIERS
            and candidate.source_record_id in active_record_ids
            and _legal_completed_candidate(
                candidate,
                target=target,
                source=records.get(candidate.source_record_id),
                position=opportunity.position,
            )
        )
    )
    return min(legal, key=_candidate_rank, default=None)


def simulate(result: ReplayResult, *, config: SimulationConfig = SimulationConfig()) -> dict[str, Any]:
    """Execute one deterministic, future-token-free protocol trace.

    A target record is admitted only after all of its output positions have
    completed.  Under a capacity limit, whole records are evicted FIFO; this is
    intentionally mechanical rather than a post-hoc value policy.
    """

    if result.future_tokens_used_for_retrieval:
        raise ContractError("simulator refuses replay evidence that used future tokens for retrieval")
    records = {record.record_id: record for record in result.records}
    opportunities = {(item.record_id, item.position): item for item in result.opportunities}
    active: set[str] = set()
    admission_order: deque[tuple[str, int]] = deque()
    active_bytes = 0
    transitions: Counter[MachineState] = Counter()
    counters: Counter[str] = Counter()
    serial_cost = 0
    machine_cost = 0
    request_results: list[dict[str, Any]] = []

    for record in result.records:
        position = 0
        request_serial_cost = len(record.committed_tokens) * config.costs.serial_token
        request_machine_cost = 0
        request_accepted = 0
        request_proposed = 0
        request_abstentions = 0
        while position < len(record.committed_tokens):
            transitions[MachineState.LOOKUP] += 1
            counters["lookup_calls"] += 1
            candidate = _select(
                opportunities[(record.record_id, position)],
                target=record,
                records=records,
                active_record_ids=active,
            )
            if candidate is None:
                counters["abstentions"] += 1
                request_abstentions += 1
                machine_cost += config.costs.lookup + config.costs.serial_token
                request_machine_cost += config.costs.lookup + config.costs.serial_token
                position += 1
                continue

            proposed = min(config.k, len(candidate.continuation), len(record.committed_tokens) - position)
            if proposed < 1:
                raise ContractError("selected candidate has no usable continuation")
            transitions[MachineState.VERIFY] += 1
            counters["verification_batches"] += 1
            counters["proposed_tokens"] += proposed
            accepted = min(candidate.accepted_prefix_length, proposed)
            counters["accepted_tokens"] += accepted
            request_proposed += proposed
            request_accepted += accepted
            batch_cost = config.costs.lookup + config.costs.verify_batch + config.costs.verify_token * (proposed + 1)
            if accepted < proposed:
                transitions[MachineState.FEEDBACK] += 1
                counters["corrections"] += 1
                batch_cost += config.costs.correction
            else:
                counters["full_accepts"] += 1
            machine_cost += batch_cost
            request_machine_cost += batch_cost
            position += accepted + 1

        serial_cost += request_serial_cost
        request_results.append({
            "record_id": record.record_id,
            "request_id": record.request_id or record.record_id,
            "output_tokens": len(record.committed_tokens),
            "serial_cost": request_serial_cost,
            "machine_cost": request_machine_cost,
            "accepted_tokens": request_accepted,
            "proposed_tokens": request_proposed,
            "abstentions": request_abstentions,
        })
        transitions[MachineState.ADMIT] += 1
        counters["admissions"] += 1
        active.add(record.record_id)
        admission_order.append((record.record_id, record.logical_source_bytes))
        active_bytes += record.logical_source_bytes
        while config.max_history_bytes is not None and active_bytes > config.max_history_bytes:
            evicted_id, evicted_bytes = admission_order.popleft()
            active.remove(evicted_id)
            active_bytes -= evicted_bytes
            transitions[MachineState.EVICT] += 1
            counters["evictions"] += 1

    transitions[MachineState.IDLE] += 1
    return {
        "schema_version": "pcss.protocol-state-machine-simulator/v1",
        "kind": "deterministic_future_token_free_protocol_cost_simulation",
        "runtime_claim": "none; abstract costs are not measured latency or speedup",
        "future_tokens_used_for_retrieval": result.future_tokens_used_for_retrieval,
        "config": {
            "k": config.k,
            "max_history_bytes": config.max_history_bytes,
            "cost_units": asdict(config.costs),
            "eviction": "whole-record FIFO after completed-record admission",
        },
        "state_transitions": {state.value: transitions[state] for state in MachineState},
        "counters": dict(sorted(counters.items())),
        "serial_cost": serial_cost,
        "machine_cost": machine_cost,
        "abstract_serial_over_machine_ratio": serial_cost / machine_cost if machine_cost else None,
        "requests": request_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-history-bytes", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_trace_fixture(args.trace)
    result = replay_chronologically(records)
    simulation = simulate(result, config=SimulationConfig(k=args.k, max_history_bytes=args.max_history_bytes))
    print(canonical_json(simulation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
