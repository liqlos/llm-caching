from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pcss.replay import (
    Fingerprints,
    ReplayError,
    ReplayRecord,
    ScopeIdentity,
    SourceTier,
    coerce_records,
    dense_exact_suffix_matches,
    footprint_accounting,
    ngram_census,
    posting_fanout,
    replay_chronologically,
)
from pcss.trace import validate_trace_sequence


FIXTURE = Path(__file__).parents[1] / "fixtures" / "replay" / "chronological-oracle-v1.json"
FINGERPRINTS = Fingerprints("model-a", "tokenizer-a", "template-a", "protocol-a")


def scope(
    *,
    repository: str = "repo-a",
    block: str = "block-a",
    session: str = "session-a",
    path: str = "path-a",
) -> ScopeIdentity:
    return ScopeIdentity(
        principal_namespace="private/local",
        application_id="app-a",
        repository_id=repository,
        workflow_stage="edit",
        semantic_block_id=block,
        session_id=session,
        revision_family="git",
        path_hash=path,
        before_blob=f"before-{path}",
    )


def record(
    record_id: str,
    index: int,
    *,
    prompt: tuple[int, ...] = (),
    committed: tuple[int, ...],
    scope_identity: ScopeIdentity | None = None,
    fingerprints: Fingerprints = FINGERPRINTS,
    source_class: str = "visible_code",
    eligible: bool = True,
    current_source: tuple[int, ...] | None = None,
) -> ReplayRecord:
    return ReplayRecord(
        record_id=record_id,
        timestamp_ns=index,
        chronological_index=index,
        prompt_tokens=prompt,
        committed_tokens=committed,
        fingerprints=fingerprints,
        scope=scope_identity or scope(),
        source_class=source_class,
        provenance_family="repository",
        eligible_for_persistent_store=eligible,
        current_source_tokens=current_source,
    )


def test_dense_suffix_reports_longest_match_and_exact_continuation() -> None:
    matches = dense_exact_suffix_matches(
        (1, 2, 3),
        (4, 5, 8),
        (9, 1, 2, 3, 4, 5, 7),
        min_suffix=1,
    )
    best = max(matches, key=lambda match: (match.matched_suffix_length, match.accepted_prefix_length))
    assert best.source_cursor == 4
    assert best.matched_suffix_length == 3
    assert best.continuation == (4, 5, 7)
    assert best.accepted_prefix_length == 2


def test_chronological_replay_never_uses_future_tokens_from_current_record() -> None:
    history = record("history", 1, committed=(1, 2, 3))
    target = record("target", 2, prompt=(1, 2), committed=(3, 4))

    result = replay_chronologically((target, history), min_suffix=1, max_continuation=8)

    first_target = next(
        opportunity
        for opportunity in result.opportunities
        if opportunity.record_id == "target" and opportunity.position == 0
    )
    assert result.future_tokens_used_for_retrieval is False
    assert first_target.all_prefix == 1
    assert {candidate.source_record_id for candidate in first_target.candidates} == {"history"}


def test_incompatible_fingerprints_are_not_mixed() -> None:
    history = record("history", 1, committed=(1, 2, 3))
    incompatible = replace(FINGERPRINTS, tokenizer="tokenizer-b")
    target = record(
        "target", 2, prompt=(1, 2), committed=(3,), fingerprints=incompatible
    )

    result = replay_chronologically((history, target))

    assert all(opportunity.all_prefix == 0 for opportunity in result.opportunities if opportunity.record_id == "target")


def test_persistent_credit_is_only_the_increment_beyond_current_source() -> None:
    history = record(
        "history",
        1,
        committed=(1, 2, 3, 4),
        scope_identity=scope(repository="repo-history", path="history"),
    )
    target = record(
        "target",
        2,
        prompt=(1, 2),
        committed=(3, 4),
        scope_identity=scope(repository="repo-target", path="target", session="session-target"),
        current_source=(1, 2, 3, 9),
    )

    result = replay_chronologically((history, target))
    opportunity = next(item for item in result.opportunities if item.record_id == "target" and item.position == 0)

    assert opportunity.prefix_for(SourceTier.CURRENT_SOURCE) == 1
    assert opportunity.prefix_for(SourceTier.PERSONAL_PERSISTENT_HISTORY) == 2
    assert opportunity.persistent_unique_prefix == 1
    assert opportunity.source_attribution() == {
        SourceTier.CURRENT_SOURCE: 1,
        SourceTier.PERSONAL_PERSISTENT_HISTORY: 1,
    }


def test_ngram_fanout_growth_and_footprint_are_declared_exactly() -> None:
    records = (
        record("one", 1, committed=(1, 2, 3, 4)),
        record("two", 2, committed=(1, 2, 3, 4)),
    )
    census = ngram_census(records, n_values=(4,))[0]
    fanout = posting_fanout(records, n_values=(4,))[0]
    footprint = footprint_accounting(records)

    assert census.total_occurrences == 2
    assert census.repeated_occurrences == 2
    assert census.chronological_reused_occurrences == 1
    assert fanout.max_fanout == 2
    assert fanout.fanout_histogram == ((2, 1),)
    assert footprint.source_tokens == 8
    assert footprint.token_bytes == 32
    assert footprint.total_bytes > footprint.token_bytes


def test_trace_v1_fixture_adapts_without_trace_import_dependency() -> None:
    records = json.loads(FIXTURE.read_text())
    validate_trace_sequence(records)

    result = replay_chronologically(records)
    target = next(
        item for item in result.opportunities if item.record_id == "target-repo-b" and item.position == 0
    )

    assert target.prefix_for(SourceTier.PERSONAL_PERSISTENT_HISTORY) == 2
    assert target.persistent_unique_prefix == 2


def test_redacted_trace_records_are_excluded_from_replay_inputs() -> None:
    records = json.loads(FIXTURE.read_text())
    redacted = copy.deepcopy(records[1])
    redacted["record_state"] = "redacted"
    redacted["content"]["prompt_token_ids"] = []
    redacted["content"]["committed_token_ids"] = []
    redacted["content"]["committed_token_status"] = "none"
    redacted["source"]["eligible_for_persistent_store"] = False
    validate_trace_sequence((records[0], redacted))

    assert [item.record_id for item in coerce_records((records[0], redacted))] == ["history-repo-a"]


def test_duplicate_chronological_key_is_rejected() -> None:
    with pytest.raises(ReplayError, match="duplicate chronological"):
        coerce_records((record("one", 1, committed=(1,)), record("two", 1, committed=(2,))))
