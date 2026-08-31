from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.persistent_store import (
    CorruptStoreError,
    IncompatibleStoreError,
    PersistentSegmentStore,
    ReadOnlyStoreError,
    StoreIdentity,
)


IDENTITY = StoreIdentity("repo-a", "model", "tokenizer", "vocab", "template", "no-thinking")


def store(path: Path, **kwargs) -> PersistentSegmentStore:
    return PersistentSegmentStore(path, IDENTITY, max_bytes=10_000_000, **kwargs)


def test_append_lookup_restart_and_longest_exact_context(tmp_path: Path) -> None:
    location = tmp_path / "store"
    first = store(location)
    first.append((1, 2), (7, 8), utility=1)
    record_id = first.append((2, 3, 4), (9, 10), utility=2)
    assert first.lookup((0, 2, 3, 4), max_tokens=1).completion == (9,)
    reopened = store(location)
    result = reopened.lookup((0, 2, 3, 4))
    assert result is not None
    assert result.record_id == record_id
    assert result.completion == (9, 10)


def test_incompatible_namespace_is_rejected_before_loading(tmp_path: Path) -> None:
    location = tmp_path / "store"
    store(location).append((1,), (2,))
    incompatible = StoreIdentity("repo-b", "model", "tokenizer", "vocab", "template", "no-thinking")
    with pytest.raises(IncompatibleStoreError, match="identity"):
        PersistentSegmentStore(location, incompatible)


def test_corrupt_referenced_segment_is_quarantined_and_rejected(tmp_path: Path) -> None:
    location = tmp_path / "store"
    active = store(location)
    active.append((1,), (2,))
    name = active._manifest["segments"][0]["name"]
    (location / name).write_text("not-json")
    reopened = store(location)
    assert reopened.lookup((1,)) is None
    assert reopened.stats().corruptions == 1
    assert list(location.glob(f"{name}.corrupt-*"))


def test_corrupt_manifest_is_rejected_and_quarantined(tmp_path: Path) -> None:
    location = tmp_path / "store"
    store(location)
    (location / "manifest.json").write_text("not-json")
    with pytest.raises(CorruptStoreError):
        store(location)
    assert list(location.glob("manifest.json.corrupt-*"))


def test_orphan_and_interrupted_temporary_writes_are_never_resurrected(tmp_path: Path) -> None:
    location = tmp_path / "store"
    active = store(location)
    active.append((1,), (2,))
    (location / "segment-interrupted.json").write_text("{}")
    (location / ".manifest.any.tmp").write_text("partial")
    reopened = store(location)
    assert reopened.lookup((1,)) is not None
    assert not (location / "segment-interrupted.json").exists()
    assert not (location / ".manifest.any.tmp").exists()


def test_deterministic_low_utility_then_oldest_eviction(tmp_path: Path) -> None:
    ticks = iter((10, 20, 30))
    location = tmp_path / "store"
    active = PersistentSegmentStore(location, IDENTITY, max_bytes=1, clock_ns=lambda: next(ticks))
    active.append((1,), (11,), utility=10)
    active.append((2,), (22,), utility=1)
    # Capacity is too small for a segment, so every append has been evicted.
    assert active.stats().record_count == 0
    assert active.stats().evictions == 2


def test_compaction_and_read_only_behavior(tmp_path: Path) -> None:
    location = tmp_path / "store"
    active = store(location)
    active.append((1,), (2,))
    active.append((3,), (4,))
    assert active.compact() is True
    assert active.stats().segment_count == 1
    readonly = store(location, read_only=True)
    assert readonly.lookup((1,)).completion == (2,)
    with pytest.raises(ReadOnlyStoreError):
        readonly.append((5,), (6,))
    with pytest.raises(ReadOnlyStoreError):
        readonly.compact()


def test_manifest_is_atomic_envelope_with_identity_and_segment_checksums(tmp_path: Path) -> None:
    location = tmp_path / "store"
    active = store(location)
    active.append((1,), (2,))
    envelope = json.loads((location / "manifest.json").read_text())
    assert set(envelope) == {"payload", "sha256"}
    assert envelope["payload"]["identity"]["namespace"] == "repo-a"
    assert envelope["payload"]["segments"][0]["sha256"]


def test_empty_context_and_nonfinite_utility_are_rejected(tmp_path: Path) -> None:
    active = store(tmp_path / "store")
    with pytest.raises(ValueError, match="context"):
        active.append((), (1,))
    with pytest.raises(ValueError, match="finite"):
        active.append((1,), (2,), utility=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        active.append((1,), (2,), utility=float("inf"))


@pytest.mark.parametrize("bad_name", ["segment-../manifest.json", "segment-not-a-segment.json"])
def test_invalid_manifest_segment_name_is_rejected_before_path_access(tmp_path: Path, bad_name: str) -> None:
    location = tmp_path / "store"
    active = store(location)
    manifest = active._empty_manifest()
    manifest["segments"] = [{"name": bad_name, "sha256": "0" * 64, "bytes": 1}]
    active._atomic_json(location / "manifest.json", active._envelope(manifest))
    with pytest.raises(CorruptStoreError, match="segment name"):
        store(location)


def test_manifest_byte_count_is_authenticated_capacity_accounting(tmp_path: Path) -> None:
    location = tmp_path / "store"
    active = store(location)
    active.append((1,), (2,))
    manifest = active._manifest.copy()
    manifest["segments"] = [active._manifest["segments"][0].copy()]
    manifest["segments"][0]["bytes"] += 1
    active._atomic_json(location / "manifest.json", active._envelope(manifest))
    reopened = store(location)
    assert reopened.lookup((1,)) is None
    assert reopened.stats().corruptions == 1


def test_eviction_manifest_commit_precedes_file_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    active = PersistentSegmentStore(tmp_path / "store", IDENTITY, max_bytes=1)
    observed: list[tuple[str, ...]] = []

    def verify(names: tuple[str, ...]) -> None:
        payload = json.loads(active.manifest_path.read_text())["payload"]
        active_names = {entry["name"] for entry in payload["segments"]}
        assert active_names.isdisjoint(names)
        observed.append(names)

    monkeypatch.setattr(active, "_remove_committed_orphans", verify)
    active.append((1,), (2,))
    assert observed and observed[0]


def test_compaction_preserves_utility_ranking(tmp_path: Path) -> None:
    active = store(tmp_path / "store")
    active.append((1,), (10,), utility=1)
    active.append((1,), (20,), utility=5)
    assert active.lookup((1,)).completion == (20,)
    assert active.compact() is True
    result = active.lookup((1,))
    assert result is not None
    assert result.completion == (20,)
    assert result.utility == 5.0
