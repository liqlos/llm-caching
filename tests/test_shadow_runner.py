from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.shadow_runner import run_shadow


ROOT = Path(__file__).resolve().parents[1]


def test_shadow_runner_uses_pre_target_ranking_and_emits_provider_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / "raw.json"
    report = tmp_path / "report.md"
    monkeypatch.setattr("pcss.shadow_runner.resolve_git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        "pcss.shadow_runner.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.shadow_runner",
    )

    receipt = run_shadow(
        trace_path=trace,
        output_path=output,
        report_path=report,
        repository_root=ROOT,
        command_arguments=("--trace", str(trace)),
        min_suffix=1,
        k=2,
    )

    assert receipt.status == "VERIFIED_OFFLINE"
    assert receipt.payload["future_tokens_used_for_retrieval"] is False
    assert receipt.payload["analysis"]["proposed_positions"] > 0
    assert any(event["event"] == "feedback" for event in receipt.payload["telemetry"])
