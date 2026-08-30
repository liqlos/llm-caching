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
from pcss.replay import Candidate, ScopeIdentity, SourceTier


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
    assert {point["name"] for point in receipt.payload["scope_strategy_summary"]} == {
        "global",
        "repository_isolation",
        "block_isolation",
        "bounded_hierarchical_widening",
    }
    assert all(point["byte_budget"] == 4096 and point["probe_budget"] == 2 for point in receipt.payload["scope_strategy_summary"])
    evidence = receipt.payload["consensus_and_negative_evidence"]
    assert evidence["kind"] == "offline_consensus_and_negative_evidence_no_runtime_control"
    assert len(evidence["consensus"]["calibration"]) == 5
    assert evidence["negative_evidence"]["events"] >= 0
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
    assert "Logical active-view capture" in report.read_text(encoding="utf-8")
    assert "Negative-evidence simulation" in report.read_text(encoding="utf-8")


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


def test_capacity_ladder_admits_active_bundle_and_reports_policy_value(tmp_path: Path) -> None:
    rfc = rfc_source(tmp_path)
    receipt = execute_offline_fixture(
        fixture_path=ROOT / "fixtures/replay/chronological-oracle-v1.json",
        rfc_path=rfc,
        repository_root=ROOT,
        command_arguments=(),
        expected_rfc_sha256=file_sha256(rfc),
    )

    capacity = receipt.payload["capacity_simulator"]
    first = capacity["capacity_ladder_bytes"]["first"]
    assert first >= 24
    for simulation in capacity["simulations"].values():
        manifests = simulation["runs"][-1]["manifests"]
        assert manifests[-1]["main"]
    summary = capacity["strategy_summary"]
    assert set(summary) == set(capacity["simulations"])
    assert all("active_capture_ratio" in point for points in summary.values() for point in points)
    assert any(point["active_capture_ratio"] > 0 for points in summary.values() for point in points)
    assert all("marginal_value_rate" in point for points in summary.values() for point in points)
    assert all(
        point["clustered_request_mean_active_capture_ci"] is not None
        for points in summary.values()
        for point in points
    )


def test_capacity_observation_rank_does_not_use_posthoc_acceptance() -> None:
    scope = ScopeIdentity(None, None, "repo", None, None, None, None, None, None)
    shorter_better_acceptance = Candidate(
        tier=SourceTier.SAME_REPOSITORY,
        source_record_id="a",
        source_cursor=0,
        matched_suffix_length=4,
        continuation=(1,),
        accepted_prefix_length=99,
        source_token_bytes=4,
        source_scope=scope,
        provenance_family="test",
        source_class="visible_code",
    )
    longer_worse_acceptance = Candidate(
        tier=SourceTier.SAME_REPOSITORY,
        source_record_id="b",
        source_cursor=0,
        matched_suffix_length=5,
        continuation=(2,),
        accepted_prefix_length=0,
        source_token_bytes=4,
        source_scope=scope,
        provenance_family="test",
        source_class="visible_code",
    )

    assert offline_runner._persistent_candidate_rank(longer_worse_acceptance) < offline_runner._persistent_candidate_rank(
        shorter_better_acceptance
    )
