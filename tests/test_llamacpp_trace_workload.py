from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.llamacpp_trace_workload import _load_prompts, _record
from pcss.trace import validate_trace_record


def _prompt() -> dict:
    return {
        "record_id": "record-1",
        "request_id": "request-1",
        "session_id": "session-1",
        "repository_id": "repo-1",
        "revision": "rev-1",
        "parent_revision": None,
        "workflow_stage": "edit",
        "semantic_block_id": "python-edit",
        "source_class": "visible_code",
        "provenance_family": "repository",
        "text": "def example():\n    pass\n",
    }


def test_generated_target_record_is_a_valid_hidden_reasoning_excluding_trace() -> None:
    record = _record(
        prompt=_prompt(),
        index=0,
        prompt_tokens=(1, 2),
        output_tokens=(3, 4),
        model_fingerprint="sha256:" + "a" * 64,
        tokenizer_fingerprint="sha256:" + "b" * 64,
        template_fingerprint="sha256:" + "c" * 64,
        protocol_fingerprint="sha256:" + "d" * 64,
    )

    validate_trace_record(record)
    assert record["reasoning_retention"] == "exclude"
    assert record["content"]["committed_token_status"] == "complete"


def test_prompt_plan_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    prompt = _prompt()
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pcss.target-trace-prompts/v1",
                "tokens_to_generate": 8,
                "prompts": [prompt, dict(prompt)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="record ids must be unique"):
        _load_prompts(path)


def test_diverse_edit_fixture_is_a_chronological_twelve_pair_workload() -> None:
    fixture = Path(__file__).parents[1] / "fixtures/replay/qwen3-8b-diverse-edit-prompts-v1.json"

    tokens_to_generate, prompts = _load_prompts(fixture)

    assert tokens_to_generate == 48
    assert len(prompts) == 24
    assert len({prompt["repository_id"] for prompt in prompts}) == 12
    assert all(prompt["parent_revision"] is None for prompt in prompts[:12])
    assert all(prompt["parent_revision"] is not None for prompt in prompts[12:])
