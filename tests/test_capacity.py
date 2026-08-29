from __future__ import annotations

import json

import pytest

from pcss.capacity import (
    ActivationBundle,
    ArchiveSpanReference,
    CapacityContractError,
    CapacityObservation,
    CapacityPolicy,
    CapacitySimulator,
    geometric_capacities,
)


def bundle(bundle_id: str, *, active_bytes: int = 10, ghost_bytes: int = 2) -> ActivationBundle:
    return ActivationBundle(
        bundle_id=bundle_id,
        archive_span=ArchiveSpanReference(record_id=f"record-{bundle_id}", start_token=0, token_count=4),
        active_bytes=active_bytes,
        ghost_bytes=ghost_bytes,
    )


def observation(epoch: int, ordinal: int, bundle_id: str, value: int) -> CapacityObservation:
    return CapacityObservation(
        observation_id=f"e{epoch}-o{ordinal}-{bundle_id}",
        epoch=epoch,
        ordinal=ordinal,
        bundle_id=bundle_id,
        unique_value=value,
    )


def member_ids(manifest, field: str) -> tuple[str, ...]:
    return tuple(reference.bundle_id for reference in getattr(manifest, field))


def test_manifests_use_only_closed_past_epochs() -> None:
    simulator = CapacitySimulator((bundle("a"), bundle("b")), common_archive_bytes=4096)
    run = simulator.run_capacity(
        (
            observation(0, 0, "a", 1),
            observation(1, 0, "b", 100),
        ),
        capacity_bytes=20,
    )

    # Epoch 1 cannot see the large future value observed in epoch 1 itself.
    assert run.manifests[1].history_through_epoch == 0
    assert member_ids(run.manifests[1], "main") == ("a",)
    assert run.manifests[2].history_through_epoch == 1
    assert member_ids(run.manifests[2], "main") == ("b",)
    assert all(manifest.verify() for manifest in run.manifests)
    assert run.manifests[2].previous_manifest_fingerprint == run.manifests[1].fingerprint


def test_membership_is_by_references_and_all_bytes_are_accounted() -> None:
    simulator = CapacitySimulator(
        (bundle("a"), bundle("b"), bundle("c"), bundle("d")), common_archive_bytes=8192
    )
    run = simulator.run_capacity(
        (
            observation(0, 0, "a", 4),
            observation(0, 1, "b", 3),
            observation(0, 2, "c", 2),
            observation(0, 3, "d", 1),
        ),
        capacity_bytes=40,
    )
    manifest = run.manifests[1]

    assert member_ids(manifest, "main") == ("a", "b")
    assert member_ids(manifest, "frontier") == ("c",)
    assert member_ids(manifest, "ghost") == ("d",)
    assert manifest.accounting.main_active_bytes == 20
    assert manifest.accounting.frontier_active_bytes == 10
    assert manifest.accounting.ghost_reference_bytes == 2
    assert manifest.accounting.logical_view_bytes == 32
    assert manifest.accounting.common_archive_bytes == 8192
    assert manifest.accounting.logical_view_bytes <= manifest.capacity_bytes
    assert not hasattr(manifest.main[0], "tokens")


def test_geometric_views_receive_the_identical_chronological_stream() -> None:
    simulator = CapacitySimulator((bundle("a"), bundle("b")), common_archive_bytes=100)
    stream = (
        observation(0, 0, "a", 2),
        observation(1, 0, "b", 3),
    )
    result = simulator.run_geometric(
        stream,
        first_capacity_bytes=8,
        max_capacity_bytes=32,
    )

    assert geometric_capacities(8, 32) == (8, 16, 32)
    assert tuple(run.capacity_bytes for run in result.runs) == (8, 16, 32)
    assert {run.stream_fingerprint for run in result.runs} == {result.stream_fingerprint}
    assert all(all(manifest.verify() for manifest in run.manifests) for run in result.runs)
    raw = json.loads(result.canonical_json())
    assert raw["schema_version"] == "pcss.capacity-simulator/v1"
    assert raw["runs"][0]["stream_fingerprint"] == result.stream_fingerprint


def test_single_fixed_recency_window_control_drops_old_winner() -> None:
    simulator = CapacitySimulator((bundle("a"), bundle("b")), common_archive_bytes=100)
    run = simulator.run_capacity(
        (
            observation(0, 0, "a", 100),
            observation(1, 0, "b", 1),
        ),
        capacity_bytes=20,
        policy=CapacityPolicy.RECENCY_WINDOW_CONTROL,
    )

    assert run.policy is CapacityPolicy.RECENCY_WINDOW_CONTROL
    assert member_ids(run.manifests[1], "main") == ("a",)
    assert member_ids(run.manifests[2], "main") == ("b",)
    assert run.manifests[2].policy.value == "fixed_one_epoch_recency_window_control"


def test_stream_validation_rejects_nonchronological_or_unknown_events() -> None:
    simulator = CapacitySimulator((bundle("a"),), common_archive_bytes=0)
    with pytest.raises(CapacityContractError, match="strict chronological"):
        simulator.run_capacity(
            (observation(0, 1, "a", 1), observation(0, 0, "a", 1)), capacity_bytes=10
        )
    with pytest.raises(CapacityContractError, match="unknown bundle_id"):
        simulator.run_capacity((observation(0, 0, "missing", 1),), capacity_bytes=10)
    with pytest.raises(CapacityContractError, match="no gaps"):
        simulator.run_capacity((observation(1, 0, "a", 1),), capacity_bytes=10)
