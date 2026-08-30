from __future__ import annotations

from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.results import make_receipt, recorded_module_command, write_compact_report, write_raw_result


def receipt():
    return make_receipt(
        evidence_class="SIMULATOR",
        command="python -m pytest tests/test_results.py",
        git_revision="a" * 40,
        input_fingerprints={"fixture": "b" * 64},
        status="VERIFIED_OFFLINE",
        payload={"count": 1},
    )


def test_raw_receipt_and_compact_report_link_exact_inputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    report = tmp_path / "report.md"
    digest = write_raw_result(raw, receipt())
    write_compact_report(report, raw_result=raw, title="Offline result")
    assert len(digest) == 64
    assert "Exact command" in report.read_text()
    assert "VERIFIED_OFFLINE" in report.read_text()


def test_report_is_not_created_without_a_real_raw_result(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="existing raw result"):
        write_compact_report(tmp_path / "report.md", raw_result=tmp_path / "missing.json", title="No result")


def test_offline_replay_report_includes_measured_summary(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    report = tmp_path / "report.md"
    replay = make_receipt(
        evidence_class="SIMULATOR",
        command="python -m pcss.offline_runner",
        git_revision="a" * 40,
        input_fingerprints={"fixture": "b" * 64},
        status="VERIFIED_OFFLINE",
        payload={
            "kind": "model_free_chronological_replay",
            "trace_records": 2,
            "proposal_positions": 4,
            "future_tokens_used_for_retrieval": False,
            "capacity_simulator": {
                "strategy_summary": {
                    "historical_value_density": [{"capacity_bytes": 64, "active_capture_ratio": 0.5}]
                }
            },
            "scope_strategy_summary": [
                {
                    "name": "global",
                    "selected_prefix_value": 3,
                    "candidate_oracle_prefix_value": 4,
                    "zero_accept_positions": 1,
                }
            ],
        },
    )
    write_raw_result(raw, replay)
    write_compact_report(report, raw_result=raw, title="Offline replay")

    content = report.read_text(encoding="utf-8")
    assert "Logical active-view capture" in content
    assert "50.00%" in content
    assert "candidate oracle `4`" in content


def test_raw_receipts_are_not_overwritten(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    write_raw_result(raw, receipt())
    with pytest.raises(ContractError, match="already exists"):
        write_raw_result(raw, receipt())


def test_receipt_command_cannot_bind_a_different_checkout(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="supplied the running pcss package"):
        recorded_module_command("pcss.offline_runner", (), tmp_path)
