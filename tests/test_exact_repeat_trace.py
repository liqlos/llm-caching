from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.exact_repeat_trace import derive_exact_repeat_records


ROOT = Path(__file__).parents[1]


def _record(record_id: str, index: int, session_id: str) -> dict:
    return {
        "schema_version": "pcss.trace/v1",
        "record_id": record_id,
        "request_id": f"request-{record_id}",
        "session_id": session_id,
        "chronological_index": index,
        "timestamp_ns": index * 1_000_000_000,
        "trusted_scope": {
            "application_id": "app", "repository_id": "repo", "workflow_stage": "edit",
            "semantic_block_id": "block", "principal_namespace": "person", "scope_class": "personal_persistent",
        },
        "artifact_lineage": {
            "repository_id": "repo", "revision": "rev", "parent_revision": None,
            "revision_family": "git", "path_hash": "sha256:" + "a" * 64,
            "before_blob": None, "after_blob": None, "source_span": None,
        },
        "model_fingerprint": "sha256:" + "b" * 64,
        "tokenizer_fingerprint": "sha256:" + "c" * 64,
        "chat_template_fingerprint": "sha256:" + "d" * 64,
        "generation_protocol_fingerprint": "sha256:" + "e" * 64,
        "generation_protocol": {
            "decoding_mode": "greedy", "thinking_mode": "off", "reasoning_effort": None,
            "preserve_thinking": False, "schema_or_tool_version": None,
        },
        "source": {"source_class": "visible_code", "provenance_family": "repository", "eligible_for_persistent_store": True},
        "privacy": {"privacy_class": "public", "retention_class": "ephemeral"},
        "reasoning_retention": "exclude", "record_state": "active", "outcome_class": "unknown",
        "content": {"prompt_token_ids": [1, 2], "committed_token_ids": [3, 4], "committed_token_status": "complete"},
    }


def test_exact_repeat_derivation_is_cross_session_and_chronological() -> None:
    records = derive_exact_repeat_records([_record("later", 2, "two"), _record("earlier", 1, "one")])

    assert [record["record_id"] for record in records] == [
        "earlier", "later", "earlier--exact-repeat-target", "later--exact-repeat-target",
    ]
    assert [record["chronological_index"] for record in records] == [1, 2, 3, 4]
    assert records[0]["content"] == records[2]["content"]
    assert records[0]["session_id"] != records[2]["session_id"]
    assert records[0]["request_id"] != records[2]["request_id"]
    assert records[0]["reasoning_retention"] == records[2]["reasoning_retention"] == "exclude"


def test_exact_repeat_derivation_rejects_duplicate_source_ids() -> None:
    with pytest.raises(ContractError, match="must be unique"):
        derive_exact_repeat_records([_record("same", 1, "one"), _record("same", 2, "two")])
