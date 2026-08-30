from __future__ import annotations

import json
from pathlib import Path

from pcss.exact_prompt_archive import build_exact_prompt_archive, build_trace_archive, open_exact_prompt_archive
from pcss.exact_repeat_trace import derive_exact_repeat_records


ROOT = Path(__file__).parents[1]


def test_exact_prompt_archive_requires_full_prompt_and_returns_cursor(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    source, target = derive_exact_repeat_records([source])
    archive = build_exact_prompt_archive(tmp_path / "archive", [source])

    assert archive.lookup(target["content"]["prompt_token_ids"][:-1]) == ()
    hits = archive.lookup(target["content"]["prompt_token_ids"])
    assert len(hits) == 1
    assert hits[0].record_id == source["record_id"]
    assert hits[0].cursor.read(2) == tuple(target["content"]["committed_token_ids"][:2])

    reopened = open_exact_prompt_archive(tmp_path / "archive")
    assert reopened.lookup(target["content"]["prompt_token_ids"])[0].cursor.read(2) == tuple(
        target["content"]["committed_token_ids"][:2]
    )


def test_trace_archive_builds_a_sealed_receipt(tmp_path: Path, monkeypatch) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(source) + "\n", encoding="utf-8")
    monkeypatch.setattr("pcss.exact_prompt_archive.resolve_git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        "pcss.exact_prompt_archive.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.exact_prompt_archive",
    )

    receipt = build_trace_archive(
        trace_path=trace,
        record_ids=[source["record_id"]],
        archive_root=tmp_path / "archive",
        raw_result_path=tmp_path / "raw.json",
        report_path=tmp_path / "report.md",
        repository_root=ROOT,
        command_arguments=(),
    )

    assert receipt.status == "EXACT_PROMPT_ARCHIVE_BUILT"
    assert (tmp_path / "archive" / "archive-manifest.json").is_file()
