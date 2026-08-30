import json
from copy import deepcopy
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.protocol_bakeoff import run_protocol_bakeoff


ROOT = Path(__file__).resolve().parents[1]
BASE_RECORD = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]


def trace_record(
    *,
    record_id: str,
    timestamp_ns: int,
    chronological_index: int,
    request_id: str,
    session_id: str = "session",
    repository_id: str = "repo",
    prompt_tokens: list[int] | None = None,
    committed_tokens: list[int] | None = None,
) -> dict:
    record = deepcopy(BASE_RECORD)
    record.update(
        record_id=record_id,
        timestamp_ns=timestamp_ns,
        chronological_index=chronological_index,
        request_id=request_id,
        session_id=session_id,
    )
    record["trusted_scope"]["repository_id"] = repository_id
    record["artifact_lineage"].update(
        repository_id=repository_id,
        revision=f"revision-{chronological_index}",
        parent_revision=None if chronological_index == 1 else f"revision-{chronological_index - 1}",
    )
    record["content"] = {
        "prompt_token_ids": prompt_tokens or [1, 2],
        "committed_token_ids": committed_tokens or [3, 4],
        "committed_token_status": "complete",
    }
    return record


def test_protocol_bakeoff_keeps_patch_and_persistent_credit_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = trace_record(
        record_id="source", timestamp_ns=1, chronological_index=1, request_id="request-source",
        session_id="session-one", repository_id="repo", prompt_tokens=[1, 2], committed_tokens=[10, 11, 12, 13],
    )
    target = trace_record(
        record_id="target", timestamp_ns=2, chronological_index=2, request_id="request-target",
        session_id="session-two", repository_id="repo", prompt_tokens=[1, 9], committed_tokens=[10, 20, 12, 13],
    )
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([source, target]), encoding="utf-8")
    monkeypatch.setattr("pcss.protocol_bakeoff.resolve_git_revision", lambda root: "a" * 40)
    receipt = run_protocol_bakeoff(
        trace_path=trace,
        pairs=("source:target",),
        output_path=tmp_path / "raw.json",
        report_path=tmp_path / "report.md",
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--pair", "source:target"),
    )
    case = receipt.payload["cases"][0]
    assert case["patch"]["copied_output_tokens"] == 3
    assert case["patch"]["edited_output_tokens"] == 1
    assert receipt.evidence_class == "SIMULATOR"
    assert receipt.payload["runtime_claim"].startswith("none")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Edit-protocol result" in report
    assert "offline scoring only" in report


def test_protocol_bakeoff_rejects_future_source(tmp_path: Path) -> None:
    source = trace_record(record_id="source", timestamp_ns=1, chronological_index=1, request_id="one")
    target = trace_record(record_id="target", timestamp_ns=2, chronological_index=2, request_id="two")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps([source, target]), encoding="utf-8")
    with pytest.raises(ContractError, match="precede target"):
        run_protocol_bakeoff(
            trace_path=trace,
            pairs=("target:source",),
            output_path=tmp_path / "raw.json",
            report_path=tmp_path / "report.md",
            repository_root=Path(__file__).resolve().parents[1],
            command_arguments=(),
        )
