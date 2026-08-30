from __future__ import annotations

import json
from pathlib import Path

from pcss.exact_prompt_archive import build_exact_prompt_archive
from pcss.exact_prompt_profiler import _profile_request
from pcss.exact_repeat_trace import derive_exact_repeat_records


ROOT = Path(__file__).parents[1]


def test_exact_prompt_profile_reads_only_the_bound_cursor(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    source, target = derive_exact_repeat_records([source])
    archive = build_exact_prompt_archive(tmp_path / "archive", [source])

    row = _profile_request(archive, target, k=2, calls=3, warmup_calls=1)

    assert row["record_id"] == target["record_id"]
    assert row["hit_count"] == 1
    assert row["candidate_tokens"] == target["content"]["committed_token_ids"][:2]
    assert row["lookup_and_cursor_ns_per_call"] > 0
