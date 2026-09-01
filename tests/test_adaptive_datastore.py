from __future__ import annotations

import random
from dataclasses import replace

import pytest

from pcss.adaptive_datastore import (
    AdaptiveDatastore,
    IndexDensity,
    PolicyName,
    SequenceMode,
    _is_confident,
)
from pcss.replay import Fingerprints, ReplayRecord, ScopeIdentity


FINGERPRINTS = Fingerprints("model", "tokenizer", "template", "protocol")


def make_scope(
    *,
    principal: str = "principal",
    application: str = "application",
    repository: str = "repository",
    session: str = "session",
) -> ScopeIdentity:
    return ScopeIdentity(
        principal_namespace=principal,
        application_id=application,
        repository_id=repository,
        workflow_stage="edit",
        semantic_block_id="block",
        session_id=session,
        revision_family="git",
        path_hash="path",
        before_blob="before",
    )


def make_record(
    record_id: str,
    ordinal: int,
    *,
    prompt: tuple[int, ...] = (),
    committed: tuple[int, ...],
    fingerprints: Fingerprints = FINGERPRINTS,
    scope: ScopeIdentity = make_scope(),
    eligible: bool = True,
    source_class: str = "visible_code",
    current_source: tuple[int, ...] | None = None,
) -> ReplayRecord:
    return ReplayRecord(
        record_id=record_id,
        timestamp_ns=ordinal,
        chronological_index=ordinal,
        request_id=record_id,
        prompt_tokens=prompt,
        committed_tokens=committed,
        fingerprints=fingerprints,
        scope=scope,
        source_class=source_class,
        provenance_family="repository",
        eligible_for_persistent_store=eligible,
        current_source_tokens=current_source,
    )


def source_with_key(key: tuple[int, ...], continuation: tuple[int, ...], prefix: int = 1000) -> tuple[int, ...]:
    return tuple(range(prefix, prefix + 3)) + key + continuation + (999999,)


def lookup(store: AdaptiveDatastore, target: ReplayRecord, query: tuple[int, ...], policy: PolicyName, k: int = 2):
    return store.lookup(target, query, k=k, policy=policy)


def test_suffix_keys_stop_at_sequence_boundaries() -> None:
    key = tuple(range(16))
    record = make_record("source", 1, committed=key + (77, 78))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(record)
    assert set(len(item) for item in store.postings) == {4, 6, 8, 12, 16}
    assert all(posting.source_cursor < len(record.committed_tokens) for values in store.postings.values() for posting in values)
    assert not any(posting.source_cursor == len(record.committed_tokens) for values in store.postings.values() for posting in values)


def test_dense_stride_and_legacy_cursor_selection() -> None:
    for density, expected in (
        (IndexDensity.DENSE, tuple(range(16, 80))),
        (IndexDensity.STRIDE_4, (16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79)),
        (IndexDensity.STRIDE_16, (16, 32, 48, 64, 79)),
        (IndexDensity.LEGACY, (32, 48, 64, 79)),
    ):
        store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, density)
        assert store._cursors(80, 16) == expected if density != IndexDensity.LEGACY else store._cursors(80, 32) == expected


def test_longest_suffix_lookup_order() -> None:
    key = tuple(range(16))
    record = make_record("source", 1, committed=source_with_key(key, (90, 91)))
    target = make_record("target", 2, committed=(90, 91))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(record)
    proposal = lookup(store, target, key, PolicyName.DENSE_MODAL)
    assert proposal is not None
    assert proposal.suffix_length == 16


def test_posting_cap_keeps_128_most_recent() -> None:
    key = tuple(range(16))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    for ordinal in range(130):
        store.ingest(make_record(f"source-{ordinal}", ordinal + 1, committed=source_with_key(key, (ordinal + 1,))))
    assert store.dropped_postings >= 2
    assert len(store.postings[key]) == 128
    assert min(posting.chronological_ordinal for posting in store.postings[key]) == 3


def test_modal_token_selection_uses_support_then_recent_ordinal_then_token_id() -> None:
    key = tuple(range(16))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(make_record("old", 1, committed=source_with_key(key, (9, 10))))
    store.ingest(make_record("new", 2, committed=source_with_key(key, (9, 11), prefix=2000)))
    target = make_record("target", 3, committed=(9, 11))
    proposal = lookup(store, target, key, PolicyName.DENSE_MODAL, 2)
    assert proposal is not None
    assert proposal.continuation == (9, 11)
    assert proposal.supports[0].record_support == 2


def test_modal_continuation_filters_surviving_postings_at_each_depth() -> None:
    key = tuple(range(16))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(make_record("a", 1, committed=source_with_key(key, (5, 10))))
    store.ingest(make_record("b", 2, committed=source_with_key(key, (5, 11), prefix=2000)))
    store.ingest(make_record("c", 3, committed=source_with_key(key, (5, 11), prefix=3000)))
    target = make_record("target", 4, committed=(5, 11))
    proposal = lookup(store, target, key, PolicyName.DENSE_MODAL, 2)
    assert proposal is not None
    assert proposal.continuation == (5, 11)
    assert proposal.supports[1].record_support == 2
    assert len(proposal.surviving_postings) == 2


@pytest.mark.parametrize(
    ("suffix", "support", "expected"),
    ((8, 2, True), (8, 1, False), (6, 3, True), (6, 2, False), (4, 4, True), (4, 3, False)),
)
def test_confidence_boundaries(suffix: int, support: int, expected: bool) -> None:
    from pcss.adaptive_datastore import DepthSupport

    value = DepthSupport(0, 7, 3 if suffix == 4 else 2, support, 0.80 if suffix == 4 else 0.75, 0.0, 0.80 if suffix == 4 else 0.75, support, False)
    assert _is_confident(suffix, value) is expected


@pytest.mark.parametrize("k", (2, 4, 8, 16))
def test_k_truncation(k: int) -> None:
    key = tuple(range(16))
    continuation = tuple(range(100, 140))
    source = make_record("source", 1, committed=source_with_key(key, continuation))
    target = make_record("target", 2, committed=continuation)
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    proposal = lookup(store, target, key, PolicyName.DENSE_MODAL, k)
    assert proposal is not None
    assert len(proposal.continuation) == k


def test_distinct_record_support_does_not_count_repeated_occurrences_as_records() -> None:
    key = tuple(range(16))
    repeated = key + (7, 8) + key + (7, 8)
    source = make_record("one", 1, committed=repeated)
    target = make_record("target", 2, committed=(7, 8))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    proposal = lookup(store, target, key, PolicyName.DENSE_MODAL)
    assert proposal is not None
    assert proposal.supports[0].record_support == 1
    assert proposal.supports[0].occurrence_support >= 2


def test_completion_only_vs_prompt_plus_completion() -> None:
    key = tuple(range(16))
    source = make_record("source", 1, prompt=key, committed=(77, 78))
    target = make_record("target", 2, committed=(77, 78))
    completion = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    combined = AdaptiveDatastore(SequenceMode.PROMPT_PLUS_COMPLETION, IndexDensity.DENSE)
    completion.ingest(source)
    combined.ingest(source)
    assert lookup(completion, target, key, PolicyName.DENSE_MODAL) is None
    assert lookup(combined, target, key, PolicyName.DENSE_MODAL) is not None


@pytest.mark.parametrize(
    "target",
    (
        make_record("target-model", 2, committed=(1, 2), fingerprints=Fingerprints("other", "tokenizer", "template", "protocol")),
        make_record("target-tokenizer", 2, committed=(1, 2), fingerprints=Fingerprints("model", "other", "template", "protocol")),
        make_record("target-template", 2, committed=(1, 2), fingerprints=Fingerprints("model", "tokenizer", "other", "protocol")),
        make_record("target-protocol", 2, committed=(1, 2), fingerprints=Fingerprints("model", "tokenizer", "template", "other")),
    ),
)
def test_fingerprint_mismatch_abstains(target: ReplayRecord) -> None:
    key = tuple(range(16))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(make_record("source", 1, committed=source_with_key(key, (1, 2))))
    assert lookup(store, target, key, PolicyName.DENSE_MODAL) is None


@pytest.mark.parametrize("field", ("principal", "application"))
def test_namespace_or_application_mismatch_abstains(field: str) -> None:
    key = tuple(range(16))
    source = make_record("source", 1, committed=source_with_key(key, (1, 2)))
    values = {"principal": "other-principal", "application": "other-application"}
    target_scope = make_scope(**{field: values[field]})
    target = make_record("target", 2, committed=(1, 2), scope=target_scope)
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    assert lookup(store, target, key, PolicyName.DENSE_MODAL) is None


def test_current_source_is_request_scoped_and_not_indexed() -> None:
    key = tuple(range(16))
    source = make_record("source", 1, committed=(1, 2), current_source=key + (7, 8))
    target = make_record("target", 2, committed=(7, 8))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    assert lookup(store, target, key, PolicyName.DENSE_MODAL) is None
    assert all(key not in store.postings for key in (key,))


def test_persistent_eligibility_rejection() -> None:
    key = tuple(range(16))
    source = make_record("source", 1, committed=source_with_key(key, (1, 2)), eligible=False)
    target = make_record("target", 2, committed=(1, 2))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    stats = store.ingest(source)
    assert stats.indexed_sequences == 0
    assert lookup(store, target, key, PolicyName.DENSE_MODAL) is None


def test_packed_storage_accounting_arithmetic() -> None:
    key = tuple(range(16))
    source = make_record("source", 1, committed=source_with_key(key, (1, 2)))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    accounting = store.accounting()
    assert accounting.total_logical_bytes == accounting.key_bytes + accounting.posting_metadata_bytes + accounting.token_payload_bytes
    assert accounting.packed_ram_bytes == len(store.postings) * 40 + accounting.postings * 32
    assert accounting.on_disk_index_bytes == accounting.key_bytes + accounting.posting_metadata_bytes
    assert accounting.bounded_ram_index_bytes <= accounting.on_disk_index_cache_cap_bytes
    assert accounting.on_disk_index_page_bytes == 4096
    assert accounting.bytes_per_indexed_source_token == accounting.total_logical_bytes / accounting.indexed_tokens


def test_chronology_self_and_later_records_are_not_visible() -> None:
    key = tuple(range(16))
    current = make_record("current", 2, committed=key + (1, 2))
    later = make_record("later", 3, committed=source_with_key(key, (1, 2)), prompt=())
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    assert lookup(store, current, key, PolicyName.DENSE_MODAL) is None
    store.ingest(later)
    assert lookup(store, current, key, PolicyName.DENSE_MODAL) is None


def test_randomized_selection_does_not_depend_on_unseen_future() -> None:
    rng = random.Random(20260831)
    key = tuple(range(16))
    source = make_record("source", 1, committed=source_with_key(key, (7, 8, 9)))
    store = AdaptiveDatastore(SequenceMode.COMPLETION_ONLY, IndexDensity.DENSE)
    store.ingest(source)
    for ordinal in range(2, 22):
        target = make_record(f"target-{ordinal}", ordinal, committed=tuple(rng.randrange(100, 200) for _ in range(4)))
        query = key
        altered = replace(target, committed_tokens=tuple(rng.randrange(300, 400) for _ in range(4)))
        first = lookup(store, target, query, PolicyName.DENSE_MODAL)
        second = lookup(store, altered, query, PolicyName.DENSE_MODAL)
        assert (first is None) == (second is None)
        if first is not None and second is not None:
            assert first.continuation == second.continuation
