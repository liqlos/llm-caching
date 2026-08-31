from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.offline_runner import load_trace_fixture
from pcss.replay import replay_chronologically
from pcss.state_machine_simulator import SimulationConfig, simulate


ROOT = Path(__file__).resolve().parents[1]


def _replay():
    return replay_chronologically(load_trace_fixture(ROOT / "fixtures/replay/chronological-oracle-v1.json"))


def test_machine_executes_lookup_verify_feedback_and_only_admits_closed_history() -> None:
    simulation = simulate(_replay(), config=SimulationConfig(k=2))

    assert simulation["future_tokens_used_for_retrieval"] is False
    assert simulation["state_transitions"]["admit"] == 2
    assert simulation["state_transitions"]["lookup"] == simulation["counters"]["lookup_calls"]
    assert simulation["counters"]["verification_batches"] > 0
    assert simulation["counters"]["accepted_tokens"] > 0
    assert simulation["serial_cost"] > 0
    assert simulation["machine_cost"] > 0
    assert simulation["runtime_claim"] == "none; abstract costs are not measured latency or speedup"


def test_zero_history_capacity_forces_abstention_without_future_leakage() -> None:
    simulation = simulate(_replay(), config=SimulationConfig(k=2, max_history_bytes=0))

    assert simulation["counters"]["evictions"] == 2
    assert simulation["counters"]["abstentions"] == simulation["counters"]["lookup_calls"]
    assert simulation["counters"].get("verification_batches", 0) == 0
    assert simulation["machine_cost"] > simulation["serial_cost"]


def test_machine_records_correction_after_a_partial_accept() -> None:
    records = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))
    records[1]["content"]["committed_token_ids"] = [13, 99, 14]

    simulation = simulate(replay_chronologically(records), config=SimulationConfig(k=2))

    assert simulation["state_transitions"]["verify"] == 1
    assert simulation["state_transitions"]["feedback"] == 1
    assert simulation["counters"]["accepted_tokens"] == 1
    assert simulation["counters"]["corrections"] == 1


def test_machine_rejects_candidate_metadata_that_does_not_bind_to_its_source() -> None:
    result = _replay()
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

    simulation = simulate(replace(result, opportunities=forged_opportunities), config=SimulationConfig(k=2))

    assert simulation["counters"].get("verification_batches", 0) == 0
    assert simulation["counters"]["abstentions"] == simulation["counters"]["lookup_calls"]


def test_machine_refuses_future_retrieval_evidence() -> None:
    with pytest.raises(ContractError, match="future tokens"):
        simulate(replace(_replay(), future_tokens_used_for_retrieval=True))
