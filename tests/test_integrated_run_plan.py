from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.integrated_run_plan import QualificationPolicy, build_integrated_run_plan, validate_integrated_run_plan
from pcss.offline_runner import load_trace_fixture
from pcss.replay import Fingerprints, ReplayRecord, ScopeIdentity, replay_chronologically


ROOT = Path(__file__).resolve().parents[1]


def _scope(*, session: str, repository: str = "repo-a", path: str = "path-a") -> ScopeIdentity:
    return ScopeIdentity("private/local", "app-a", repository, "edit", "block-a", session, "git", path, f"before-{path}")


def _record(record_id: str, index: int, *, committed: tuple[int, ...], scope: ScopeIdentity, current_source: tuple[int, ...] | None = None) -> ReplayRecord:
    return ReplayRecord(record_id, index, index, (1, 2), committed, Fingerprints("m", "t", "c", "g"), scope, "visible_code", "repository", True, current_source)


def test_decision_is_pre_target_and_posthoc_is_separate() -> None:
    records = load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json")
    plan = build_integrated_run_plan(replay_chronologically(records))

    assert plan["status"] == "QUALIFIED"
    case = plan["selected_cases"][0]
    assert "accepted_prefix_length" not in case["decision"]
    assert set(case) == {"decision", "posthoc"}
    assert case["decision"]["cross_session"] is True
    assert plan["future_tokens_used_for_retrieval"] is False


def test_same_session_history_cannot_qualify_under_default_policy() -> None:
    shared = _scope(session="one")
    result = replay_chronologically((_record("history", 1, committed=(1, 2, 3, 4), scope=shared), _record("target", 2, committed=(3, 4), scope=shared)))
    plan = build_integrated_run_plan(result)

    assert plan["status"] == "NOT_QUALIFIED_NO_ELIGIBLE_CROSS_SESSION_DECISION"
    assert plan["selected_cases"] == []


def test_current_source_covered_persistent_has_zero_unique_credit() -> None:
    history = _record("history", 1, committed=(1, 2, 3, 4), scope=_scope(session="history", repository="repo-history", path="history"))
    target = _record("target", 2, committed=(3, 4), scope=_scope(session="target", repository="repo-target", path="target"), current_source=(1, 2, 3, 4))
    plan = build_integrated_run_plan(replay_chronologically((history, target)))

    assert plan["status"] == "NOT_QUALIFIED_ZERO_UNIQUE_PERSISTENT_CREDIT"
    assert plan["qualification"]["unique_persistent_credit"] == 0


def test_trace_or_policy_change_changes_fingerprint() -> None:
    records = load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json")
    baseline = build_integrated_run_plan(replay_chronologically(records))
    changed_policy = build_integrated_run_plan(replay_chronologically(records), policy=QualificationPolicy(k=1))
    changed_records = copy.deepcopy(records)
    changed_records[1]["content"]["committed_token_ids"][-1] = 98
    changed_trace = build_integrated_run_plan(replay_chronologically(changed_records))

    assert baseline["plan_fingerprint"] != changed_policy["plan_fingerprint"]
    assert baseline["plan_fingerprint"] != changed_trace["plan_fingerprint"]


def test_malformed_policy_and_tampered_plan_fail_closed() -> None:
    with pytest.raises(ContractError):
        QualificationPolicy.from_mapping({"schema_version": "wrong"})
    records = load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json")
    plan = build_integrated_run_plan(replay_chronologically(records))
    tampered = copy.deepcopy(plan)
    tampered["selected_cases"][0]["decision"]["matched_suffix_length"] = 999
    with pytest.raises(ContractError, match="fingerprint mismatch"):
        validate_integrated_run_plan(tampered)


def test_candidate_metadata_must_bind_to_the_completed_source_record() -> None:
    records = load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json")
    result = replay_chronologically(records)
    target_scope = result.records[1].scope
    forged_opportunities = tuple(
        replace(
            opportunity,
            candidates=tuple(
                replace(candidate, source_scope=target_scope)
                if candidate.source_record_id == "history-repo-a"
                else candidate
                for candidate in opportunity.candidates
            ),
        )
        for opportunity in result.opportunities
    )

    plan = build_integrated_run_plan(replace(result, opportunities=forged_opportunities))

    assert plan["status"] == "NOT_QUALIFIED_NO_ELIGIBLE_CROSS_SESSION_DECISION"
    assert plan["selected_cases"] == []


def test_candidate_rank_metadata_is_recomputed_from_pre_target_tokens() -> None:
    records = load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json")
    result = replay_chronologically(records)
    forged_opportunities = tuple(
        replace(
            opportunity,
            candidates=tuple(
                replace(candidate, matched_suffix_length=999)
                if candidate.source_record_id == "history-repo-a"
                else candidate
                for candidate in opportunity.candidates
            ),
        )
        for opportunity in result.opportunities
    )

    plan = build_integrated_run_plan(replace(result, opportunities=forged_opportunities))

    assert plan["status"] == "NOT_QUALIFIED_NO_ELIGIBLE_CROSS_SESSION_DECISION"
    assert plan["selected_cases"] == []
