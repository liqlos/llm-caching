from __future__ import annotations

import json
from pathlib import Path

from pcss.offline_runner import execute_offline_fixture, main


ROOT = Path(__file__).resolve().parents[1]


def rfc_source(tmp_path: Path) -> Path:
    path = tmp_path / "PCSS_MASTER_RFC_EN.md"
    path.write_text("# Test technical source\n", encoding="utf-8")
    return path


def test_offline_fixture_emits_simulator_receipt_without_runtime_claim(tmp_path: Path) -> None:
    receipt = execute_offline_fixture(
        fixture_path=ROOT / "fixtures/replay/chronological-oracle-v1.json",
        rfc_path=rfc_source(tmp_path),
        command="offline-test-command",
        git_revision="test-revision",
    )

    assert receipt.evidence_class == "SIMULATOR"
    assert receipt.status == "VERIFIED_OFFLINE"
    assert receipt.payload["future_tokens_used_for_retrieval"] is False
    assert receipt.payload["runtime_claim"] == "none"
    assert receipt.payload["active_metrics"]["status"] == "NOT_RUN"
    assert receipt.payload["archive_validation"]["cursor_reads_match_input"] is True
    assert set(receipt.payload["capacity_simulator"]["simulations"]) == {
        "historical_value_density",
        "fixed_one_epoch_recency_window_control",
    }


def test_offline_cli_writes_raw_then_report(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    report = tmp_path / "report.md"
    rfc = rfc_source(tmp_path)

    assert (
        main(
            [
                "--fixture",
                str(ROOT / "fixtures/replay/chronological-oracle-v1.json"),
                "--rfc",
                str(rfc),
                "--output",
                str(raw),
                "--report",
                str(report),
                "--git-revision",
                "test-revision",
                "--exact-command",
                "offline-test-command",
            ]
        )
        == 0
    )

    payload = json.loads(raw.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "SIMULATOR"
    assert payload["payload"]["active_metrics"]["status"] == "NOT_RUN"
    assert "offline-test-command" in report.read_text(encoding="utf-8")
