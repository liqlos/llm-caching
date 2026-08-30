from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.shadow_runner import run_shadow
from pcss.exact_repeat_trace import derive_exact_repeat_records


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


def test_exact_prompt_admission_selects_only_full_prompt_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    exact, target = derive_exact_repeat_records([source])
    edited = json.loads(json.dumps(target))
    edited["record_id"] = "edited-target"
    edited["request_id"] = "edited-request"
    edited["session_id"] = "edited-session"
    edited["chronological_index"] = 3
    edited["timestamp_ns"] += 1_000_000_000
    edited["content"]["prompt_token_ids"] = [999_999, *edited["content"]["prompt_token_ids"][1:]]
    trace = tmp_path / "trace.jsonl"
    trace.write_text("".join(json.dumps(record) + "\n" for record in (exact, target, edited)), encoding="utf-8")
    monkeypatch.setattr("pcss.shadow_runner.resolve_git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        "pcss.shadow_runner.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.shadow_runner",
    )

    receipt = run_shadow(
        trace_path=trace,
        output_path=tmp_path / "raw.json",
        report_path=tmp_path / "report.md",
        repository_root=ROOT,
        command_arguments=(),
        min_suffix=1,
        k=2,
        exact_prompt_only=True,
    )

    assert receipt.payload["admission_policy"] == "exact_prompt_only/v1"
    assert receipt.payload["analysis"]["exact_prompt_only"] is True
    assert receipt.payload["analysis"]["proposed_positions"] == len(target["content"]["committed_token_ids"])
    assert receipt.payload["analysis"]["abstentions"] >= 2 * len(target["content"]["committed_token_ids"])
