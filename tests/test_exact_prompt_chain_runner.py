from __future__ import annotations

import json
from pathlib import Path

from pcss.exact_prompt_archive import build_exact_prompt_archive
from pcss.exact_prompt_chain_runner import _run_pair
from pcss.exact_repeat_trace import derive_exact_repeat_records


ROOT = Path(__file__).parents[1]


def test_chain_runner_advances_by_accepted_tokens_and_bonus(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    source, target = derive_exact_repeat_records([source])
    archive = build_exact_prompt_archive(tmp_path / "archive", [source])

    result = _run_pair(archive, source, target, k=2)

    assert result["complete"] is True
    assert result["final_committed_tokens"] == len(target["content"]["committed_token_ids"])
    assert all(round_["status"] == "all_accepted" for round_ in result["rounds"])
    assert result["rounds"][0]["committed_tokens"] == target["content"]["committed_token_ids"][:3]
