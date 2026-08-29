"""Causal, byte-accounted logical active views over one immutable archive.

This is an offline simulator.  It has no decode-path mutation, online archive
append, eviction controller, or storage-performance claim.  It only makes the
capacity/policy question reproducible on one chronological opportunity stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Iterable, Mapping

from .canonical import ContractError, canonical_json, require_int, require_string, sha256


CAPACITY_SCHEMA_VERSION = "pcss.capacity-simulator/v1"
RECENCY_WINDOW_EPOCHS = 1


class CapacityContractError(ContractError):
    """Raised when a capacity replay would not be reproducible or causal."""


class CapacityPolicy(str, Enum):
    """The simulator's two deliberately small, inspectable offline policies."""

    HISTORICAL_VALUE_DENSITY = "historical_value_density"
    RECENCY_WINDOW_CONTROL = "fixed_one_epoch_recency_window_control"


@dataclass(frozen=True)
class ArchiveSpanReference:
    """Logical reference to tokens already held in the one common archive."""

    record_id: str
    start_token: int
    token_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", require_string(self.record_id, "archive_span.record_id"))
        object.__setattr__(
            self, "start_token", require_int(self.start_token, "archive_span.start_token", minimum=0)
        )
        object.__setattr__(
            self, "token_count", require_int(self.token_count, "archive_span.token_count", minimum=1)
        )


@dataclass(frozen=True)
class ActivationBundle:
    """One activation candidate with no copied token payload.

    ``active_bytes`` accounts for the logical fast-path representation of this
    bundle (keys, postings, metadata).  ``ghost_bytes`` is the metadata-only
    reference cost when the bundle is outside the active view.
    """

    bundle_id: str
    archive_span: ArchiveSpanReference
    active_bytes: int
    ghost_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", require_string(self.bundle_id, "bundle_id"))
        if not isinstance(self.archive_span, ArchiveSpanReference):
            raise CapacityContractError("archive_span must be an ArchiveSpanReference")
        object.__setattr__(
            self, "active_bytes", require_int(self.active_bytes, "active_bytes", minimum=1)
        )
        object.__setattr__(self, "ghost_bytes", require_int(self.ghost_bytes, "ghost_bytes", minimum=1))


@dataclass(frozen=True)
class CapacityObservation:
    """One pre-computed, model-free unique opportunity on the fixed stream."""

    observation_id: str
    epoch: int
    ordinal: int
    bundle_id: str
    unique_value: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation_id", require_string(self.observation_id, "observation_id")
        )
        object.__setattr__(self, "epoch", require_int(self.epoch, "epoch", minimum=0))
        object.__setattr__(self, "ordinal", require_int(self.ordinal, "ordinal", minimum=0))
        object.__setattr__(self, "bundle_id", require_string(self.bundle_id, "bundle_id"))
        object.__setattr__(
            self, "unique_value", require_int(self.unique_value, "unique_value", minimum=0)
        )


@dataclass(frozen=True)
class BundleReference:
    """Membership is only a bundle ID; span/token data remains shared once."""

    bundle_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", require_string(self.bundle_id, "bundle_reference.bundle_id"))


@dataclass(frozen=True)
class ByteAccounting:
    """Exact bytes for one logical view, with common archive bytes separate."""

    common_archive_bytes: int
    main_active_bytes: int
    frontier_active_bytes: int
    ghost_reference_bytes: int
    logical_view_bytes: int
    capacity_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "common_archive_bytes",
            "main_active_bytes",
            "frontier_active_bytes",
            "ghost_reference_bytes",
            "logical_view_bytes",
            "capacity_bytes",
        ):
            object.__setattr__(self, field, require_int(getattr(self, field), field, minimum=0))
        calculated = self.main_active_bytes + self.frontier_active_bytes + self.ghost_reference_bytes
        if self.logical_view_bytes != calculated:
            raise CapacityContractError("logical_view_bytes must equal the sum of every membership cost")
        if self.logical_view_bytes > self.capacity_bytes:
            raise CapacityContractError("logical view exceeds its declared capacity")

    def as_dict(self) -> dict[str, int]:
        return {
            "common_archive_bytes": self.common_archive_bytes,
            "main_active_bytes": self.main_active_bytes,
            "frontier_active_bytes": self.frontier_active_bytes,
            "ghost_reference_bytes": self.ghost_reference_bytes,
            "logical_view_bytes": self.logical_view_bytes,
            "capacity_bytes": self.capacity_bytes,
        }


@dataclass(frozen=True)
class EpochManifest:
    """An immutable, fingerprinted view installed for exactly one epoch."""

    schema_version: str
    epoch: int
    policy: CapacityPolicy
    history_through_epoch: int
    capacity_bytes: int
    main: tuple[BundleReference, ...]
    frontier: tuple[BundleReference, ...]
    ghost: tuple[BundleReference, ...]
    accounting: ByteAccounting
    previous_manifest_fingerprint: str | None
    fingerprint: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "policy": self.policy.value,
            "history_through_epoch": self.history_through_epoch,
            "capacity_bytes": self.capacity_bytes,
            "main": [reference.bundle_id for reference in self.main],
            "frontier": [reference.bundle_id for reference in self.frontier],
            "ghost": [reference.bundle_id for reference in self.ghost],
            "accounting": self.accounting.as_dict(),
            "previous_manifest_fingerprint": self.previous_manifest_fingerprint,
        }

    def verify(self) -> bool:
        if self.schema_version != CAPACITY_SCHEMA_VERSION:
            return False
        if self.history_through_epoch != self.epoch - 1:
            return False
        memberships = [
            *(reference.bundle_id for reference in self.main),
            *(reference.bundle_id for reference in self.frontier),
            *(reference.bundle_id for reference in self.ghost),
        ]
        if len(memberships) != len(set(memberships)):
            return False
        return self.fingerprint == sha256(self.payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True)
class EpochOutcome:
    """Value observed while serving one sealed manifest on its later epoch."""

    epoch: int
    observation_count: int
    main_unique_value: int
    frontier_unique_value: int
    ghost_unique_value: int
    inactive_unique_value: int

    @property
    def total_unique_value(self) -> int:
        return (
            self.main_unique_value
            + self.frontier_unique_value
            + self.ghost_unique_value
            + self.inactive_unique_value
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "observation_count": self.observation_count,
            "main_unique_value": self.main_unique_value,
            "frontier_unique_value": self.frontier_unique_value,
            "ghost_unique_value": self.ghost_unique_value,
            "inactive_unique_value": self.inactive_unique_value,
            "total_unique_value": self.total_unique_value,
        }


@dataclass(frozen=True)
class CapacityRun:
    """One capacity point over the same chronological observation stream."""

    capacity_bytes: int
    policy: CapacityPolicy
    stream_fingerprint: str
    manifests: tuple[EpochManifest, ...]
    outcomes: tuple[EpochOutcome, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "policy": self.policy.value,
            "stream_fingerprint": self.stream_fingerprint,
            "manifests": [manifest.as_dict() for manifest in self.manifests],
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class CapacitySimulation:
    """Machine-readable output for a geometric ladder over one common archive."""

    schema_version: str
    common_archive_bytes: int
    stream_fingerprint: str
    policy: CapacityPolicy
    runs: tuple[CapacityRun, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "common_archive_bytes": self.common_archive_bytes,
            "stream_fingerprint": self.stream_fingerprint,
            "policy": self.policy.value,
            "runs": [run.as_dict() for run in self.runs],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())


@dataclass(frozen=True)
class _HistoryStat:
    total_value: int
    last_epoch: int
    last_ordinal: int


class CapacitySimulator:
    """Replay fixed chronological opportunities with causal epoch boundaries.

    The sole archive is represented by the bundle span references supplied to
    the constructor.  Every active view and ghost set only retains bundle IDs,
    so the archive payload is never duplicated across capacity points.
    """

    def __init__(self, bundles: Iterable[ActivationBundle], *, common_archive_bytes: int) -> None:
        common_archive_bytes = require_int(
            common_archive_bytes, "common_archive_bytes", minimum=0
        )
        try:
            normalized_bundles = tuple(bundles)
        except TypeError as exc:
            raise CapacityContractError("bundles must be an iterable of ActivationBundle objects") from exc
        for index, bundle in enumerate(normalized_bundles):
            if not isinstance(bundle, ActivationBundle):
                raise CapacityContractError(f"bundles[{index}] must be an ActivationBundle")
        bundle_ids = [bundle.bundle_id for bundle in normalized_bundles]
        if len(set(bundle_ids)) != len(bundle_ids):
            raise CapacityContractError("bundle_id values must be unique")
        self._bundles = {bundle.bundle_id: bundle for bundle in normalized_bundles}
        self._common_archive_bytes = common_archive_bytes

    @property
    def common_archive_bytes(self) -> int:
        return self._common_archive_bytes

    def run_geometric(
        self,
        observations: Iterable[CapacityObservation],
        *,
        first_capacity_bytes: int,
        max_capacity_bytes: int,
        factor: int = 2,
        policy: CapacityPolicy = CapacityPolicy.HISTORICAL_VALUE_DENSITY,
    ) -> CapacitySimulation:
        stream = self._normalise_stream(observations)
        capacities = geometric_capacities(first_capacity_bytes, max_capacity_bytes, factor=factor)
        return self._run(stream, capacities, policy)

    def run_capacity(
        self,
        observations: Iterable[CapacityObservation],
        *,
        capacity_bytes: int,
        policy: CapacityPolicy = CapacityPolicy.HISTORICAL_VALUE_DENSITY,
    ) -> CapacityRun:
        stream = self._normalise_stream(observations)
        return self._run(stream, (require_int(capacity_bytes, "capacity_bytes", minimum=1),), policy).runs[0]

    def _run(
        self,
        stream: tuple[CapacityObservation, ...],
        capacities: tuple[int, ...],
        policy: CapacityPolicy,
    ) -> CapacitySimulation:
        if not isinstance(policy, CapacityPolicy):
            raise CapacityContractError("policy must be a CapacityPolicy")
        stream_fingerprint = sha256([_observation_dict(observation) for observation in stream])
        runs = tuple(
            self._run_one_capacity(stream, capacity_bytes, policy, stream_fingerprint)
            for capacity_bytes in capacities
        )
        return CapacitySimulation(
            schema_version=CAPACITY_SCHEMA_VERSION,
            common_archive_bytes=self._common_archive_bytes,
            stream_fingerprint=stream_fingerprint,
            policy=policy,
            runs=runs,
        )

    def _run_one_capacity(
        self,
        stream: tuple[CapacityObservation, ...],
        capacity_bytes: int,
        policy: CapacityPolicy,
        stream_fingerprint: str,
    ) -> CapacityRun:
        grouped = _group_by_epoch(stream)
        manifest = self._build_manifest(
            epoch=0,
            history=(),
            capacity_bytes=capacity_bytes,
            policy=policy,
            previous_manifest_fingerprint=None,
        )
        manifests = [manifest]
        outcomes: list[EpochOutcome] = []
        history: list[CapacityObservation] = []
        for epoch, epoch_observations in grouped:
            if manifest.epoch != epoch:
                raise CapacityContractError("internal epoch manifest chain is discontinuous")
            outcomes.append(_evaluate_epoch(manifest, epoch_observations))
            history.extend(epoch_observations)
            manifest = self._build_manifest(
                epoch=epoch + 1,
                history=tuple(history),
                capacity_bytes=capacity_bytes,
                policy=policy,
                previous_manifest_fingerprint=manifest.fingerprint,
            )
            manifests.append(manifest)
        return CapacityRun(
            capacity_bytes=capacity_bytes,
            policy=policy,
            stream_fingerprint=stream_fingerprint,
            manifests=tuple(manifests),
            outcomes=tuple(outcomes),
        )

    def _build_manifest(
        self,
        *,
        epoch: int,
        history: tuple[CapacityObservation, ...],
        capacity_bytes: int,
        policy: CapacityPolicy,
        previous_manifest_fingerprint: str | None,
    ) -> EpochManifest:
        # This method receives only closed epochs.  Its output is used by the
        # next epoch, which is why a manifest's history boundary is explicit.
        ranked = self._rank_history(history, policy)
        main_budget, frontier_budget, ghost_budget = _membership_budgets(capacity_bytes)
        main, main_bytes = self._select(ranked, (), main_budget, active=True)
        frontier, frontier_bytes = self._select(ranked, main, frontier_budget, active=True)
        known = {observation.bundle_id for observation in history}
        ghost_order = tuple(ranked) + tuple(sorted(known - set(ranked)))
        ghost, ghost_bytes = self._select(ghost_order, (*main, *frontier), ghost_budget, active=False)
        accounting = ByteAccounting(
            common_archive_bytes=self._common_archive_bytes,
            main_active_bytes=main_bytes,
            frontier_active_bytes=frontier_bytes,
            ghost_reference_bytes=ghost_bytes,
            logical_view_bytes=main_bytes + frontier_bytes + ghost_bytes,
            capacity_bytes=capacity_bytes,
        )
        payload = {
            "schema_version": CAPACITY_SCHEMA_VERSION,
            "epoch": epoch,
            "policy": policy.value,
            "history_through_epoch": epoch - 1,
            "capacity_bytes": capacity_bytes,
            "main": list(main),
            "frontier": list(frontier),
            "ghost": list(ghost),
            "accounting": accounting.as_dict(),
            "previous_manifest_fingerprint": previous_manifest_fingerprint,
        }
        fingerprint = sha256(payload)
        manifest = EpochManifest(
            schema_version=CAPACITY_SCHEMA_VERSION,
            epoch=epoch,
            policy=policy,
            history_through_epoch=epoch - 1,
            capacity_bytes=capacity_bytes,
            main=tuple(BundleReference(bundle_id) for bundle_id in main),
            frontier=tuple(BundleReference(bundle_id) for bundle_id in frontier),
            ghost=tuple(BundleReference(bundle_id) for bundle_id in ghost),
            accounting=accounting,
            previous_manifest_fingerprint=previous_manifest_fingerprint,
            fingerprint=fingerprint,
        )
        if not manifest.verify():
            raise CapacityContractError("internal manifest sealing failed")
        return manifest

    def _rank_history(
        self, history: tuple[CapacityObservation, ...], policy: CapacityPolicy
    ) -> tuple[str, ...]:
        if not history:
            return ()
        if policy is CapacityPolicy.RECENCY_WINDOW_CONTROL:
            # Exactly one untuned control: the immediately preceding sealed
            # epoch, not an adjustable window or a runtime FIFO variant.
            recent_epoch = history[-1].epoch
            recent = [observation for observation in history if observation.epoch == recent_epoch]
            per_bundle: dict[str, _HistoryStat] = {}
            for observation in recent:
                previous = per_bundle.get(observation.bundle_id)
                total = observation.unique_value + (previous.total_value if previous else 0)
                per_bundle[observation.bundle_id] = _HistoryStat(
                    total_value=total,
                    last_epoch=observation.epoch,
                    last_ordinal=observation.ordinal,
                )
            return tuple(
                sorted(
                    per_bundle,
                    key=lambda bundle_id: (
                        -per_bundle[bundle_id].last_ordinal,
                        -per_bundle[bundle_id].total_value,
                        bundle_id,
                    ),
                )
            )

        stats: dict[str, _HistoryStat] = {}
        for observation in history:
            previous = stats.get(observation.bundle_id)
            stats[observation.bundle_id] = _HistoryStat(
                total_value=observation.unique_value + (previous.total_value if previous else 0),
                last_epoch=observation.epoch,
                last_ordinal=observation.ordinal,
            )
        return tuple(
            sorted(
                stats,
                key=lambda bundle_id: (
                    -Fraction(stats[bundle_id].total_value, self._bundles[bundle_id].active_bytes),
                    -stats[bundle_id].total_value,
                    -stats[bundle_id].last_epoch,
                    -stats[bundle_id].last_ordinal,
                    bundle_id,
                ),
            )
        )

    def _select(
        self,
        ranked_bundle_ids: Iterable[str],
        excluded: Iterable[str],
        budget: int,
        *,
        active: bool,
    ) -> tuple[tuple[str, ...], int]:
        selected: list[str] = []
        selected_ids = set(excluded)
        used_bytes = 0
        for bundle_id in ranked_bundle_ids:
            if bundle_id in selected_ids:
                continue
            bundle = self._bundles[bundle_id]
            cost = bundle.active_bytes if active else bundle.ghost_bytes
            if used_bytes + cost > budget:
                continue
            selected.append(bundle_id)
            selected_ids.add(bundle_id)
            used_bytes += cost
        return tuple(selected), used_bytes

    def _normalise_stream(
        self, observations: Iterable[CapacityObservation]
    ) -> tuple[CapacityObservation, ...]:
        try:
            stream = tuple(observations)
        except TypeError as exc:
            raise CapacityContractError("observations must be iterable") from exc
        previous_key: tuple[int, int] | None = None
        expected_epoch = 0
        seen_epochs: set[int] = set()
        seen_ids: set[str] = set()
        for index, observation in enumerate(stream):
            if not isinstance(observation, CapacityObservation):
                raise CapacityContractError(
                    f"observations[{index}] must be a CapacityObservation"
                )
            if observation.bundle_id not in self._bundles:
                raise CapacityContractError(
                    f"observations[{index}] references unknown bundle_id {observation.bundle_id}"
                )
            if observation.observation_id in seen_ids:
                raise CapacityContractError("observation_id values must be unique")
            seen_ids.add(observation.observation_id)
            key = (observation.epoch, observation.ordinal)
            if previous_key is not None and key <= previous_key:
                raise CapacityContractError("observations must be in strict chronological epoch/ordinal order")
            previous_key = key
            if observation.epoch not in seen_epochs:
                if observation.epoch != expected_epoch:
                    raise CapacityContractError("epochs must start at zero and have no gaps")
                seen_epochs.add(observation.epoch)
                expected_epoch += 1
        return stream


def geometric_capacities(first_capacity_bytes: int, max_capacity_bytes: int, *, factor: int = 2) -> tuple[int, ...]:
    """Return the unrounded geometric ladder used for every logical view."""
    first_capacity_bytes = require_int(first_capacity_bytes, "first_capacity_bytes", minimum=1)
    max_capacity_bytes = require_int(max_capacity_bytes, "max_capacity_bytes", minimum=first_capacity_bytes)
    factor = require_int(factor, "factor", minimum=2)
    capacities: list[int] = []
    capacity = first_capacity_bytes
    while capacity <= max_capacity_bytes:
        capacities.append(capacity)
        capacity *= factor
    return tuple(capacities)


def _membership_budgets(capacity_bytes: int) -> tuple[int, int, int]:
    """Fixed, reported partition; it is not an adaptive production controller."""
    main = capacity_bytes // 2
    frontier = capacity_bytes // 4
    ghost = capacity_bytes - main - frontier
    return main, frontier, ghost


def _group_by_epoch(
    stream: tuple[CapacityObservation, ...]
) -> tuple[tuple[int, tuple[CapacityObservation, ...]], ...]:
    if not stream:
        return ()
    groups: list[tuple[int, tuple[CapacityObservation, ...]]] = []
    current_epoch = stream[0].epoch
    current: list[CapacityObservation] = []
    for observation in stream:
        if observation.epoch != current_epoch:
            groups.append((current_epoch, tuple(current)))
            current_epoch = observation.epoch
            current = []
        current.append(observation)
    groups.append((current_epoch, tuple(current)))
    return tuple(groups)


def _evaluate_epoch(
    manifest: EpochManifest, observations: tuple[CapacityObservation, ...]
) -> EpochOutcome:
    if not observations:
        raise CapacityContractError("an epoch must contain at least one observation")
    if any(observation.epoch != manifest.epoch for observation in observations):
        raise CapacityContractError("outcome observations do not match their sealed manifest epoch")
    main = {reference.bundle_id for reference in manifest.main}
    frontier = {reference.bundle_id for reference in manifest.frontier}
    ghost = {reference.bundle_id for reference in manifest.ghost}
    values = {"main": 0, "frontier": 0, "ghost": 0, "inactive": 0}
    for observation in observations:
        if observation.bundle_id in main:
            values["main"] += observation.unique_value
        elif observation.bundle_id in frontier:
            values["frontier"] += observation.unique_value
        elif observation.bundle_id in ghost:
            values["ghost"] += observation.unique_value
        else:
            values["inactive"] += observation.unique_value
    return EpochOutcome(
        epoch=manifest.epoch,
        observation_count=len(observations),
        main_unique_value=values["main"],
        frontier_unique_value=values["frontier"],
        ghost_unique_value=values["ghost"],
        inactive_unique_value=values["inactive"],
    )


def _observation_dict(observation: CapacityObservation) -> dict[str, int | str]:
    return {
        "observation_id": observation.observation_id,
        "epoch": observation.epoch,
        "ordinal": observation.ordinal,
        "bundle_id": observation.bundle_id,
        "unique_value": observation.unique_value,
    }
