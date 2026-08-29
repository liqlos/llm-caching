"""Offline opportunity metrics for PCSS replay.

The metrics intentionally stop before runtime economics: they quantify exact
post-hoc reuse and recovery under declared conditions, never a decode speedup.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import TypeVar

from .replay import Opportunity, SourceTier


class MetricError(ValueError):
    """Raised when metric inputs cannot support an auditable denominator."""


def _validate_minimum(minimum_prefix: int) -> None:
    if isinstance(minimum_prefix, bool) or not isinstance(minimum_prefix, int) or minimum_prefix < 1:
        raise MetricError("minimum_prefix must be a positive integer")


def _opportunity_value(
    opportunity: Opportunity,
    *,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> int:
    if tier is not None and persistent_unique:
        raise MetricError("tier and persistent_unique cannot be combined")
    if persistent_unique:
        return opportunity.persistent_unique_prefix
    if tier is not None:
        return opportunity.prefix_for(tier)
    return opportunity.all_prefix


@dataclass(frozen=True)
class OpportunityMetric:
    positions: int
    positive_positions: int
    minimum_prefix: int
    rate: float
    accepted_prefix_value: int


def opportunity(
    opportunities: Iterable[Opportunity],
    *,
    minimum_prefix: int = 1,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> OpportunityMetric:
    """Opportunity rate and summed accepted-prefix proxy for a declared tier."""

    _validate_minimum(minimum_prefix)
    items = tuple(opportunities)
    values = tuple(
        _opportunity_value(item, tier=tier, persistent_unique=persistent_unique) for item in items
    )
    positive = sum(value >= minimum_prefix for value in values)
    return OpportunityMetric(
        positions=len(items),
        positive_positions=positive,
        minimum_prefix=minimum_prefix,
        rate=(positive / len(items)) if items else 0.0,
        accepted_prefix_value=sum(values),
    )


def accepted_prefix_value(
    opportunities: Iterable[Opportunity],
    *,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> int:
    """Sum exact post-hoc prefix lengths; proposal positions remain distinct."""

    return opportunity(
        opportunities, minimum_prefix=1, tier=tier, persistent_unique=persistent_unique
    ).accepted_prefix_value


@dataclass(frozen=True)
class ChainSurvivalPoint:
    prefix_length: int
    surviving_positions: int
    total_positions: int
    survival_rate: float


def chain_survival(
    opportunities: Iterable[Opportunity],
    *,
    max_prefix: int | None = None,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> tuple[ChainSurvivalPoint, ...]:
    """Survival curve for exact chains at increasing requested lengths."""

    values = tuple(
        _opportunity_value(item, tier=tier, persistent_unique=persistent_unique)
        for item in opportunities
    )
    if max_prefix is None:
        max_prefix = max(values, default=0)
    if isinstance(max_prefix, bool) or not isinstance(max_prefix, int) or max_prefix < 0:
        raise MetricError("max_prefix must be a non-negative integer")
    total = len(values)
    return tuple(
        ChainSurvivalPoint(
            prefix_length=length,
            surviving_positions=sum(value >= length for value in values),
            total_positions=total,
            survival_rate=(sum(value >= length for value in values) / total) if total else 0.0,
        )
        for length in range(1, max_prefix + 1)
    )


@dataclass(frozen=True)
class SourceAttribution:
    accepted_prefix_value: int
    tokens_by_tier: tuple[tuple[SourceTier, int], ...]
    fractions_by_tier: tuple[tuple[SourceTier, float], ...]


def source_attribution(opportunities: Iterable[Opportunity]) -> SourceAttribution:
    """Aggregate span-level cheapest-legal-source credit from replay records."""

    counts: Counter[SourceTier] = Counter()
    for item in opportunities:
        counts.update(item.source_attribution())
    total = sum(counts.values())
    ordered = tuple(sorted(counts.items(), key=lambda item: item[0].value))
    return SourceAttribution(
        accepted_prefix_value=total,
        tokens_by_tier=ordered,
        fractions_by_tier=tuple(
            (tier, (count / total) if total else 0.0) for tier, count in ordered
        ),
    )


def oracle_reuse_ratio(
    opportunities: Iterable[Opportunity],
    *,
    minimum_prefix: int = 1,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> float:
    """ORR: fraction of legal positions with a post-hoc exact opportunity."""

    return opportunity(
        opportunities,
        minimum_prefix=minimum_prefix,
        tier=tier,
        persistent_unique=persistent_unique,
    ).rate


@dataclass(frozen=True)
class PracticalOutcome:
    """Telemetry-shaped outcome from an offline active-system simulation.

    It is deliberately generic: no instance implies that a live target decoded
    or verified the candidate.
    """

    record_id: str
    position: int
    accepted_prefix_length: int
    source_tier: SourceTier | None = None
    authorized: bool = True
    on_time: bool = True
    router_profitable: bool = True

    @property
    def key(self) -> tuple[str, int]:
        return (self.record_id, self.position)


def _outcome_map(
    opportunities: Sequence[Opportunity], outcomes: Iterable[PracticalOutcome]
) -> dict[tuple[str, int], PracticalOutcome]:
    known = {(item.record_id, item.position): item for item in opportunities}
    result: dict[tuple[str, int], PracticalOutcome] = {}
    for outcome in outcomes:
        if isinstance(outcome.accepted_prefix_length, bool) or outcome.accepted_prefix_length < 0:
            raise MetricError("accepted_prefix_length must be non-negative")
        if outcome.key not in known:
            raise MetricError(f"outcome is not in evaluation window: {outcome.key}")
        if outcome.key in result:
            raise MetricError(f"duplicate practical outcome: {outcome.key}")
        if outcome.accepted_prefix_length > known[outcome.key].target_remaining_tokens:
            raise MetricError(f"outcome exceeds target remainder: {outcome.key}")
        result[outcome.key] = outcome
    return result


@dataclass(frozen=True)
class ViableRecovery:
    oracle_positions: int
    recovered_positions: int
    minimum_prefix: int
    rate: float


def viable_opportunity_recovery_rate(
    opportunities: Iterable[Opportunity],
    outcomes: Iterable[PracticalOutcome],
    *,
    minimum_prefix: int = 1,
) -> ViableRecovery:
    """VORR: practical authorized/on-time/profitable recovery over ORR positions."""

    _validate_minimum(minimum_prefix)
    items = tuple(opportunities)
    indexed = _outcome_map(items, outcomes)
    oracle_items = tuple(item for item in items if item.all_prefix >= minimum_prefix)
    recovered = sum(
        1
        for item in oracle_items
        if (outcome := indexed.get((item.record_id, item.position))) is not None
        and outcome.authorized
        and outcome.on_time
        and outcome.router_profitable
        and outcome.accepted_prefix_length >= minimum_prefix
    )
    return ViableRecovery(
        oracle_positions=len(oracle_items),
        recovered_positions=recovered,
        minimum_prefix=minimum_prefix,
        rate=(recovered / len(oracle_items)) if oracle_items else 0.0,
    )


@dataclass(frozen=True)
class ActiveCapture:
    practical_marginal_value: int
    full_archive_oracle_marginal_value: int
    ratio: float


def active_capture_ratio(
    opportunities: Iterable[Opportunity], outcomes: Iterable[PracticalOutcome]
) -> ActiveCapture:
    """ACR after removing current-context/current-source credit from both sides."""

    items = tuple(opportunities)
    indexed = _outcome_map(items, outcomes)
    full_oracle = sum(item.persistent_unique_prefix for item in items)
    practical = 0
    for item in items:
        outcome = indexed.get((item.record_id, item.position))
        if outcome is None or outcome.source_tier not in {
            SourceTier.SAME_SESSION,
            SourceTier.SAME_REPOSITORY,
            SourceTier.PERSONAL_PERSISTENT_HISTORY,
            SourceTier.IMPORTED_CORPUS,
        }:
            continue
        if not (outcome.authorized and outcome.on_time and outcome.router_profitable):
            continue
        baseline = max(item.context_prefix, item.current_source_prefix)
        practical += max(0, min(outcome.accepted_prefix_length, item.persistent_prefix) - baseline)
    return ActiveCapture(
        practical_marginal_value=practical,
        full_archive_oracle_marginal_value=full_oracle,
        ratio=(practical / full_oracle) if full_oracle else 0.0,
    )


@dataclass(frozen=True)
class CapacityValue:
    """One fixed-capacity offline value observation with exact byte accounting."""

    capacity_bytes: int
    practical_marginal_value: float
    full_archive_oracle_marginal_value: float


@dataclass(frozen=True)
class CapacityCurvePoint:
    capacity_bytes: int
    practical_marginal_value: float
    full_archive_oracle_marginal_value: float
    active_capture_ratio: float
    marginal_value_rate: float | None


def capacity_curve(values: Iterable[CapacityValue]) -> tuple[CapacityCurvePoint, ...]:
    """SVCC-style offline accepted-prefix curve; values are not performance."""

    ordered = tuple(sorted(values, key=lambda value: value.capacity_bytes))
    previous: CapacityValue | None = None
    result: list[CapacityCurvePoint] = []
    for value in ordered:
        if isinstance(value.capacity_bytes, bool) or value.capacity_bytes < 0:
            raise MetricError("capacity_bytes must be non-negative")
        if not math.isfinite(value.practical_marginal_value) or not math.isfinite(
            value.full_archive_oracle_marginal_value
        ):
            raise MetricError("capacity values must be finite")
        if previous is not None and value.capacity_bytes == previous.capacity_bytes:
            raise MetricError("capacity bytes must be unique")
        mvr = None
        if previous is not None:
            mvr = (value.practical_marginal_value - previous.practical_marginal_value) / (
                value.capacity_bytes - previous.capacity_bytes
            )
        result.append(
            CapacityCurvePoint(
                capacity_bytes=value.capacity_bytes,
                practical_marginal_value=value.practical_marginal_value,
                full_archive_oracle_marginal_value=value.full_archive_oracle_marginal_value,
                active_capture_ratio=(
                    value.practical_marginal_value / value.full_archive_oracle_marginal_value
                    if value.full_archive_oracle_marginal_value
                    else 0.0
                ),
                marginal_value_rate=mvr,
            )
        )
        previous = value
    return tuple(result)


T = TypeVar("T")


@dataclass(frozen=True)
class ClusteredConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    bootstrap_replicates: int
    request_clusters: int


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise MetricError("cannot calculate a quantile of no bootstrap values")
    if probability <= 0:
        return values[0]
    if probability >= 1:
        return values[-1]
    location = (len(values) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (location - lower)


def clustered_bootstrap_ci(
    values_by_request: Mapping[str, Sequence[float]],
    *,
    statistic: Callable[[Sequence[float]], float] | None = None,
    bootstrap_replicates: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ClusteredConfidenceInterval:
    """Percentile CI that resamples whole requests, never individual tokens.

    ``statistic`` receives the flattened token-position values from sampled
    request clusters.  The default is their mean, matching a per-position rate
    while retaining within-request dependence during resampling.
    """

    if not values_by_request:
        raise MetricError("at least one request cluster is required")
    if bootstrap_replicates < 1:
        raise MetricError("bootstrap_replicates must be positive")
    if not 0 < confidence_level < 1:
        raise MetricError("confidence_level must be between zero and one")
    clusters = tuple((request_id, tuple(values)) for request_id, values in sorted(values_by_request.items()))
    if any(not request_id for request_id, _ in clusters):
        raise MetricError("request cluster ids must be non-empty")
    if any(not values for _, values in clusters):
        raise MetricError("each request cluster must contain at least one value")
    if any(not math.isfinite(value) for _, values in clusters for value in values):
        raise MetricError("clustered values must be finite")
    reducer = statistic or fmean

    def evaluate(sample: Sequence[tuple[str, tuple[float, ...]]]) -> float:
        value = reducer(tuple(item for _, cluster in sample for item in cluster))
        if not math.isfinite(value):
            raise MetricError("statistic must return a finite value")
        return value

    estimate = evaluate(clusters)
    rng = random.Random(seed)
    sampled = sorted(
        evaluate(tuple(clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))))
        for _ in range(bootstrap_replicates)
    )
    alpha = (1 - confidence_level) / 2
    return ClusteredConfidenceInterval(
        estimate=estimate,
        lower=_quantile(sampled, alpha),
        upper=_quantile(sampled, 1 - alpha),
        confidence_level=confidence_level,
        bootstrap_replicates=bootstrap_replicates,
        request_clusters=len(clusters),
    )


def request_cluster_values(
    opportunities: Iterable[Opportunity],
    *,
    tier: SourceTier | None = None,
    persistent_unique: bool = False,
) -> dict[str, tuple[float, ...]]:
    """Build request-level groups for a metric CI without token IID assumptions."""

    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for item in opportunities:
        grouped[item.request_cluster_id].append(
            float(_opportunity_value(item, tier=tier, persistent_unique=persistent_unique))
        )
    return {request_id: tuple(values) for request_id, values in grouped.items()}
