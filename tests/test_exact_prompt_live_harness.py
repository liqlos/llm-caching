from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.exact_prompt_archive import build_exact_prompt_archive
from pcss.exact_prompt_live_harness import _candidate_from_archive
from pcss.exact_repeat_trace import derive_exact_repeat_records


ROOT = Path(__file__).parents[1]


def test_archive_candidate_binds_exact_prompt_and_source_provenance(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    source, target = derive_exact_repeat_records([source])
    archive = build_exact_prompt_archive(tmp_path / "archive", [source])

    candidate, elapsed = _candidate_from_archive(
        archive, target, expected_source_record_id=source["record_id"], output_tokens=2
    )
    assert candidate == tuple(target["content"]["committed_token_ids"][:2])
    assert elapsed > 0

    with pytest.raises(ContractError, match="provenance"):
        _candidate_from_archive(archive, target, expected_source_record_id="other", output_tokens=2)
