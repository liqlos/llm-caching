from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import pcss.offline_runner as offline_runner
from pcss.canonical import ContractError
from pcss.offline_runner import (
    PCSS_MASTER_RFC_SHA256,
    execute_offline_fixture,
    main,
)
from pcss.results import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def rfc_source(tmp_path: Path) -> Path:
    path = tmp_path / "PCSS_MASTER_RFC_EN.md"
    path.write_text("# Test technical source\n", encoding="utf-8")
    return path


def test_offline_fixture_emits_simulator_receipt_without_runtime_claim(tmp_path: Path) -> None:
    rfc = rfc_source(tmp_path)
    receipt = execute_offline_fixture(
        fixture_path=ROOT / "fixtures/replay/chronological-oracle-v1.json",
        rfc_path=rfc,
        repository_root=ROOT,
        command_arguments=("--fixture", "fixture.json"),
        expected_rfc_sha256=file_sha256(rfc),
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
    assert receipt.git_revision == subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert "offline-test-command" not in receipt.command


def test_offline_cli_writes_raw_then_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw.json"
    report = tmp_path / "report.md"
    rfc = rfc_source(tmp_path)
    monkeypatch.setattr(
        offline_runner,
        "_validate_rfc_source",
        lambda path, *, expected_sha256: file_sha256(path),
    )

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
                "--repository-root",
                str(ROOT),
            ]
        )
        == 0
    )

    payload = json.loads(raw.read_text(encoding="utf-8"))
    assert payload["evidence_class"] == "SIMULATOR"
    assert payload["payload"]["active_metrics"]["status"] == "NOT_RUN"
    assert "pcss.offline_runner" in report.read_text(encoding="utf-8")


def test_offline_fixture_excludes_future_tokens_and_rejects_unpinned_rfc(tmp_path: Path) -> None:
    rfc = rfc_source(tmp_path)
    fixture = tmp_path / "future-only.json"
    record = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    record["content"] = {
        "prompt_token_ids": [101, 102, 103],
        "committed_token_ids": [104, 105, 106],
        "committed_token_status": "complete",
    }
    fixture.write_text(json.dumps([record]), encoding="utf-8")

    receipt = execute_offline_fixture(
        fixture_path=fixture,
        rfc_path=rfc,
        repository_root=ROOT,
        command_arguments=(),
        expected_rfc_sha256=file_sha256(rfc),
    )

    assert receipt.payload["future_tokens_used_for_retrieval"] is False
    assert receipt.payload["oracle_metrics"]["accepted_prefix_value"] == 0
    with pytest.raises(ContractError, match="fingerprint"):
        execute_offline_fixture(
            fixture_path=fixture,
            rfc_path=rfc,
            repository_root=ROOT,
            command_arguments=(),
            expected_rfc_sha256=PCSS_MASTER_RFC_SHA256,
        )


def test_offline_fixture_accepts_fail_closed_normalized_jsonl(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))
    fixture = tmp_path / "trace.jsonl"
    fixture.write_text("".join(json.dumps(record) + "\n" for record in source), encoding="utf-8")

    records = offline_runner.load_trace_fixture(fixture)

    assert [record["record_id"] for record in records] == ["history-repo-a", "target-repo-b"]
