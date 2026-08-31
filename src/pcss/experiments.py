"""Host-independent logical experiments built on chronological replay.

These experiments compare legal information surfaces and offline post-hoc
prefixes.  They do not connect a provider to target decoding and do not make a
runtime-performance claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum
from math import fsum
from typing import Any

from .replay import (
    Candidate,
    Opportunity,
    ReplayResult,
    ScopeIdentity,
    SourceTier,
    continuation_prefix_length,
    dense_exact_suffix_matches,
)


class ExperimentError(ValueError):
    """Raised when an offline experiment would violate its declared budget."""


class ScopeLevel(StrEnum):
    BLOCK = "same_repository_and_block"
    REPOSITORY = "same_repository"
    GLOBAL = "global"


@dataclass(frozen=True)
class ScopePolicy:
    """A logical scope route with identical source-byte/probe budgets."""

    name: str
    levels: tuple[ScopeLevel, ...]
    byte_budget: int
    probe_budget: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ExperimentError("scope policy name is required")
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ExperimentError("scope policy levels must be non-empty and unique")
        if isinstance(self.byte_budget, bool) or self.byte_budget < 0:
            raise ExperimentError("byte_budget must be non-negative")
        if isinstance(self.probe_budget, bool) or self.probe_budget < 1:
            raise ExperimentError("probe_budget must be positive")


def standard_scope_policies(*, byte_budget: int, probe_budget: int) -> tuple[ScopePolicy, ...]:
    """Global, isolated, and bounded-widening policies on the same budgets."""

    return (
        ScopePolicy("global", (ScopeLevel.GLOBAL,), byte_budget, probe_budget),
        ScopePolicy("repository_isolation", (ScopeLevel.REPOSITORY,), byte_budget, probe_budget),
        ScopePolicy("block_isolation", (ScopeLevel.BLOCK,), byte_budget, probe_budget),
        ScopePolicy(
            "bounded_hierarchical_widening",
            (ScopeLevel.BLOCK, ScopeLevel.REPOSITORY, ScopeLevel.GLOBAL),
            byte_budget,
            probe_budget,
        ),
    )


def _persistent_candidate(candidate: Candidate) -> bool:
    return candidate.tier in {
        SourceTier.SAME_SESSION,
        SourceTier.SAME_REPOSITORY,
        SourceTier.PERSONAL_PERSISTENT_HISTORY,
        SourceTier.IMPORTED_CORPUS,
    }


def _matches_level(target: ScopeIdentity, candidate: Candidate, level: ScopeLevel) -> bool:
    if level == ScopeLevel.GLOBAL:
        return True
    if level == ScopeLevel.REPOSITORY:
        return target.same_repository(candidate.source_scope)
    if level == ScopeLevel.BLOCK:
        return target.same_block(candidate.source_scope)
    raise ExperimentError(f"unsupported scope level: {level}")


def _retrieval_rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    """Pre-target rank: exact suffix, local tier, small source, stable cursor."""

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


@dataclass(frozen=True)
class ScopedPosition:
    record_id: str
    position: int
    probes_used: int
    source_bytes_used: int
    levels_probed: tuple[ScopeLevel, ...]
    selected: Candidate | None
    selected_accepted_prefix: int
    candidate_oracle_prefix: int


@dataclass(frozen=True)
class ScopeExperimentResult:
    policy: ScopePolicy
    positions: tuple[ScopedPosition, ...]
    selected_prefix_value: int
    candidate_oracle_prefix_value: int
    selected_positive_positions: int
    zero_accept_positions: int
    selected_source_tiers: tuple[tuple[SourceTier, int], ...]


def _budgeted_lookup(
    opportunity: Opportunity,
    target_scope: ScopeIdentity,
    policy: ScopePolicy,
) -> ScopedPosition:
    """Probe candidates under fixed budgets without using target acceptance to rank."""

    candidates = tuple(candidate for candidate in opportunity.candidates if _persistent_candidate(candidate))
    seen: set[tuple[str, int, SourceTier]] = set()
    charged_sources: set[str] = set()
    admitted: list[Candidate] = []
    levels_probed: list[ScopeLevel] = []
    probes = 0
    byte_cost = 0
    for level in policy.levels:
        levels_probed.append(level)
        eligible = sorted(
            (candidate for candidate in candidates if _matches_level(target_scope, candidate, level)),
            key=_retrieval_rank,
        )
        for candidate in eligible:
            if candidate.candidate_key in seen:
                continue
            seen.add(candidate.candidate_key)
            if probes >= policy.probe_budget:
                break
            candidate_cost = (
                candidate.source_token_bytes if candidate.source_record_id not in charged_sources else 0
            )
            # A source that does not fit is not exposed; it cannot consume a
            # continuation probe and make another policy look worse by accident.
            if byte_cost + candidate_cost > policy.byte_budget:
                continue
            probes += 1
            byte_cost += candidate_cost
            charged_sources.add(candidate.source_record_id)
            admitted.append(candidate)
        if probes >= policy.probe_budget:
            break
    selected = min(admitted, key=_retrieval_rank) if admitted else None
    return ScopedPosition(
        record_id=opportunity.record_id,
        position=opportunity.position,
        probes_used=probes,
        source_bytes_used=byte_cost,
        levels_probed=tuple(levels_probed),
        selected=selected,
        selected_accepted_prefix=selected.accepted_prefix_length if selected else 0,
        candidate_oracle_prefix=max((item.accepted_prefix_length for item in admitted), default=0),
    )


def run_scope_experiment(result: ReplayResult, policy: ScopePolicy) -> ScopeExperimentResult:
    """Run one policy over the identical chronological opportunities and budgets."""

    scopes = {record.record_id: record.scope for record in result.records}
    positions = tuple(
        _budgeted_lookup(opportunity, scopes[opportunity.record_id], policy)
        for opportunity in result.opportunities
    )
    selected_tiers: Counter[SourceTier] = Counter(
        position.selected.tier for position in positions if position.selected is not None
    )
    return ScopeExperimentResult(
        policy=policy,
        positions=positions,
        selected_prefix_value=sum(position.selected_accepted_prefix for position in positions),
        candidate_oracle_prefix_value=sum(position.candidate_oracle_prefix for position in positions),
        selected_positive_positions=sum(position.selected_accepted_prefix > 0 for position in positions),
        zero_accept_positions=sum(
            position.selected is not None and position.selected_accepted_prefix == 0 for position in positions
        ),
        selected_source_tiers=tuple(sorted(selected_tiers.items(), key=lambda item: item[0].value)),
    )


def compare_scope_policies(
    result: ReplayResult, *, byte_budget: int, probe_budget: int
) -> tuple[ScopeExperimentResult, ...]:
    """Compare all four logical routes with exactly the same declared budgets."""

    return tuple(
        run_scope_experiment(result, policy)
        for policy in standard_scope_policies(byte_budget=byte_budget, probe_budget=probe_budget)
    )


@dataclass(frozen=True)
class Reanchor:
    output_position: int
    source_cursor: int
    matched_suffix_length: int
    accepted_prefix_length: int


@dataclass(frozen=True)
class CurrentSourceReuse:
    positions: int
    accepted_prefix_value: int
    positive_positions: int
    reanchors: tuple[Reanchor, ...]


def _as_tokens(value: Sequence[int], field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ExperimentError(f"{field} must be token ids")
    result = tuple(value)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in result):
        raise ExperimentError(f"{field} must be non-negative integer token ids")
    return result


def current_source_reanchor_oracle(
    before_tokens: Sequence[int], after_tokens: Sequence[int], *, max_continuation: int | None = 32
) -> CurrentSourceReuse:
    """Offline current-source reuse/re-anchor oracle for an edit trajectory.

    The full source is legal input.  Future ``after_tokens`` score a candidate
    after its cursor is selected by suffix length and stable cursor order.
    """

    before = _as_tokens(before_tokens, "before_tokens")
    after = _as_tokens(after_tokens, "after_tokens")
    if max_continuation is not None and max_continuation < 1:
        raise ExperimentError("max_continuation must be positive when set")
    prefixes: list[int] = []
    anchors: list[Reanchor] = []
    for position in range(len(after)):
        if position == 0:
            continuation = before[:max_continuation]
            accepted = continuation_prefix_length(after, continuation)
            matched = 0
            cursor = 0
        else:
            matches = dense_exact_suffix_matches(
                after[:position],
                after[position:],
                before,
                min_suffix=1,
                max_continuation=max_continuation,
            )
            selected = min(
                matches,
                key=lambda match: (-match.matched_suffix_length, match.source_cursor),
                default=None,
            )
            if selected is None:
                accepted = 0
                matched = 0
                cursor = 0
            else:
                accepted = selected.accepted_prefix_length
                matched = selected.matched_suffix_length
                cursor = selected.source_cursor
        prefixes.append(accepted)
        if position > 0 and accepted > 0:
            anchors.append(Reanchor(position, cursor, matched, accepted))
    return CurrentSourceReuse(
        positions=len(after),
        accepted_prefix_value=sum(prefixes),
        positive_positions=sum(prefix > 0 for prefix in prefixes),
        reanchors=tuple(anchors),
    )


@dataclass(frozen=True)
class PatchOperation:
    operation: str
    source_start: int
    source_length: int
    output_start: int
    output_length: int


@dataclass(frozen=True)
class PatchEditRepresentation:
    operations: tuple[PatchOperation, ...]
    copied_output_tokens: int
    edited_output_tokens: int


def patch_edit_representation(
    before_tokens: Sequence[int], after_tokens: Sequence[int]
) -> PatchEditRepresentation:
    """Exact token-level patch representation for an offline protocol comparison."""

    before = _as_tokens(before_tokens, "before_tokens")
    after = _as_tokens(after_tokens, "after_tokens")
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    operations: list[PatchOperation] = []
    copied = 0
    edited = 0
    for tag, source_start, source_end, output_start, output_end in matcher.get_opcodes():
        operation = "copy" if tag == "equal" else tag
        output_length = output_end - output_start
        operations.append(
            PatchOperation(
                operation=operation,
                source_start=source_start,
                source_length=source_end - source_start,
                output_start=output_start,
                output_length=output_length,
            )
        )
        if tag == "equal":
            copied += output_length
        else:
            edited += output_length
    return PatchEditRepresentation(tuple(operations), copied, edited)


@dataclass(frozen=True)
class ProtocolValue:
    protocol: str
    accepted_prefix_value: int
    directly_covered_output_tokens: int


@dataclass(frozen=True)
class EditProtocolComparison:
    output_tokens: int
    current_source: CurrentSourceReuse
    patch: PatchEditRepresentation
    values: tuple[ProtocolValue, ...]


def compare_edit_protocols(
    before_tokens: Sequence[int],
    after_tokens: Sequence[int],
    *,
    persistent_opportunities: Iterable[Opportunity] = (),
    max_continuation: int | None = 32,
) -> EditProtocolComparison:
    """Compare regeneration, source reuse, patch representation, and PCSS credit.

    Persistent credit is taken only from ``persistent_unique_prefix`` so it
    cannot claim copy/re-anchor value already available from the current source.
    """

    after = _as_tokens(after_tokens, "after_tokens")
    reuse = current_source_reanchor_oracle(before_tokens, after, max_continuation=max_continuation)
    patch = patch_edit_representation(before_tokens, after)
    persistent = tuple(persistent_opportunities)
    persistent_value = sum(item.persistent_unique_prefix for item in persistent)
    persistent_covered = sum(item.persistent_unique_prefix > 0 for item in persistent)
    return EditProtocolComparison(
        output_tokens=len(after),
        current_source=reuse,
        patch=patch,
        values=(
            ProtocolValue("full_regeneration", 0, 0),
            ProtocolValue(
                "current_source_reuse_reanchor",
                reuse.accepted_prefix_value,
                reuse.positive_positions,
            ),
            ProtocolValue("patch_edit_representation", patch.copied_output_tokens, patch.copied_output_tokens),
            ProtocolValue("persistent_cross_session", persistent_value, persistent_covered),
        ),
    )


@dataclass(frozen=True)
class ConsensusResult:
    raw_support: int
    independent_support: int
    consensus_prefix: tuple[int, ...]
    support_shrunk_score: float
    representative_candidates: tuple[Candidate, ...]


def _common_prefix(values: Sequence[Sequence[int]]) -> tuple[int, ...]:
    if not values:
        return ()
    common: list[int] = []
    for tokens in zip(*values):
        if len(set(tokens)) != 1:
            break
        common.append(tokens[0])
    return tuple(common)


def consensus_by_provenance(
    candidates: Iterable[Candidate], *, shrinkage: float = 2.0
) -> ConsensusResult:
    """Choose a modal continuation supported by independent provenance families."""

    if shrinkage <= 0:
        raise ExperimentError("shrinkage must be positive")
    usable = tuple(candidate for candidate in candidates if candidate.continuation)
    if not usable:
        return ConsensusResult(0, 0, (), 0.0, ())
    raw_by_first: Counter[int] = Counter(candidate.continuation[0] for candidate in usable)
    families_by_first: defaultdict[int, set[str]] = defaultdict(set)
    for candidate in usable:
        families_by_first[candidate.continuation[0]].add(candidate.provenance_family)
    chosen_first = min(
        raw_by_first,
        key=lambda token: (-len(families_by_first[token]), -raw_by_first[token], token),
    )
    selected = tuple(candidate for candidate in usable if candidate.continuation[0] == chosen_first)
    by_family: defaultdict[str, list[Candidate]] = defaultdict(list)
    for candidate in selected:
        by_family[candidate.provenance_family].append(candidate)
    representatives = tuple(
        min(
            family_candidates,
            key=lambda candidate: (
                -candidate.matched_suffix_length,
                candidate.source_record_id,
                candidate.source_cursor,
            ),
        )
        for _, family_candidates in sorted(by_family.items())
    )
    raw_support = len(selected)
    independent_support = len(representatives)
    duplicate_penalty = independent_support / raw_support
    score = (independent_support / (independent_support + shrinkage)) * duplicate_penalty
    return ConsensusResult(
        raw_support=raw_support,
        independent_support=independent_support,
        consensus_prefix=_common_prefix([item.continuation for item in representatives]),
        support_shrunk_score=score,
        representative_candidates=representatives,
    )


@dataclass(frozen=True)
class ConsensusObservation:
    score: float
    independent_support: int
    accepted_prefix_length: int


@dataclass(frozen=True)
class ConsensusCalibrationBin:
    lower: float
    upper: float
    observations: int
    positive_rate: float
    mean_accepted_prefix: float
    mean_independent_support: float


def calibrate_consensus(
    observations: Iterable[ConsensusObservation], *, bins: int = 5, minimum_prefix: int = 1
) -> tuple[ConsensusCalibrationBin, ...]:
    """Deterministic score calibration table, preserving provenance-family support."""

    if isinstance(bins, bool) or bins < 1:
        raise ExperimentError("bins must be positive")
    if isinstance(minimum_prefix, bool) or minimum_prefix < 1:
        raise ExperimentError("minimum_prefix must be positive")
    values = tuple(observations)
    for item in values:
        if not 0 <= item.score <= 1 or item.independent_support < 0 or item.accepted_prefix_length < 0:
            raise ExperimentError("invalid consensus observation")
    bucketed: list[list[ConsensusObservation]] = [[] for _ in range(bins)]
    for item in values:
        index = min(bins - 1, int(item.score * bins))
        bucketed[index].append(item)
    result: list[ConsensusCalibrationBin] = []
    for index, bucket in enumerate(bucketed):
        lower = index / bins
        upper = (index + 1) / bins
        result.append(
            ConsensusCalibrationBin(
                lower=lower,
                upper=upper,
                observations=len(bucket),
                positive_rate=(
                    sum(item.accepted_prefix_length >= minimum_prefix for item in bucket) / len(bucket)
                    if bucket
                    else 0.0
                ),
                mean_accepted_prefix=(fsum(item.accepted_prefix_length for item in bucket) / len(bucket))
                if bucket
                else 0.0,
                mean_independent_support=(fsum(item.independent_support for item in bucket) / len(bucket))
                if bucket
                else 0.0,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class NegativeEvidenceEvent:
    key: str
    chronological_index: int
    accepted_prefix_length: int
    failure_weight: float = 1.0


@dataclass(frozen=True)
class NegativeEvidencePoint:
    key: str
    chronological_index: int
    score_before: float
    score_after: float
    suppressed_before: bool


class DecayedNegativeEvidence:
    """Bounded, recovering offline anti-cache; not a production controller."""

    def __init__(
        self,
        *,
        half_life_events: float = 10.0,
        suppress_at: float = 2.0,
        max_entries: int = 128,
        success_recovery_multiplier: float = 0.5,
    ) -> None:
        if half_life_events <= 0 or suppress_at < 0 or max_entries < 1:
            raise ExperimentError("invalid negative-evidence configuration")
        if not 0 <= success_recovery_multiplier <= 1:
            raise ExperimentError("success_recovery_multiplier must be between zero and one")
        self._half_life_events = half_life_events
        self._suppress_at = suppress_at
        self._max_entries = max_entries
        self._success_recovery_multiplier = success_recovery_multiplier
        self._entries: dict[str, tuple[float, int]] = {}
        self._last_chronological_index = -1

    def score(self, key: str, chronological_index: int) -> float:
        if not key:
            raise ExperimentError("negative-evidence key is required")
        if chronological_index < 0:
            raise ExperimentError("chronological_index must be non-negative")
        state = self._entries.get(key)
        if state is None:
            return 0.0
        weight, last_index = state
        if chronological_index < last_index:
            raise ExperimentError("negative-evidence lookup moves backwards in chronology")
        return weight * 0.5 ** ((chronological_index - last_index) / self._half_life_events)

    def should_suppress(self, key: str, chronological_index: int) -> bool:
        return self.score(key, chronological_index) >= self._suppress_at

    def observe(self, event: NegativeEvidenceEvent) -> NegativeEvidencePoint:
        if not event.key:
            raise ExperimentError("negative-evidence key is required")
        if event.chronological_index < self._last_chronological_index:
            raise ExperimentError("negative evidence must be observed chronologically")
        if event.accepted_prefix_length < 0 or event.failure_weight < 0:
            raise ExperimentError("negative-evidence event values must be non-negative")
        before = self.score(event.key, event.chronological_index)
        suppressed = before >= self._suppress_at
        if event.accepted_prefix_length == 0:
            after = before + event.failure_weight
        else:
            after = before * self._success_recovery_multiplier
        self._entries[event.key] = (after, event.chronological_index)
        self._last_chronological_index = event.chronological_index
        while len(self._entries) > self._max_entries:
            evicted = min(self._entries, key=lambda key: (self._entries[key][1], key))
            del self._entries[evicted]
        return NegativeEvidencePoint(
            key=event.key,
            chronological_index=event.chronological_index,
            score_before=before,
            score_after=after,
            suppressed_before=suppressed,
        )

    def snapshot(self, chronological_index: int) -> tuple[tuple[str, float], ...]:
        """Stable decayed state for machine-readable offline reports."""

        return tuple(sorted((key, self.score(key, chronological_index)) for key in self._entries))


def simulate_decayed_negative_evidence(
    events: Iterable[NegativeEvidenceEvent], **configuration: Any
) -> tuple[NegativeEvidencePoint, ...]:
    """Run a compact chronological anti-cache simulation with explicit decay."""

    simulator = DecayedNegativeEvidence(**configuration)
    return tuple(simulator.observe(event) for event in events)
