from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.trace import (
    TRACE_SCHEMA_VERSION,
    chronological_key,
    compatibility_fingerprints,
    normalize_trace_record,
    trace_record_fingerprint,
    trace_stream_fingerprint,
    validate_fingerprint_compatibility,
    validate_trace_record,
    validate_trace_sequence,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "traces" / "chronological-v1.jsonl"


def load_records() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def test_visible_committed_trace_fixture_validates() -> None:
    records = load_records()
    validate_trace_sequence(records, require_uniform_fingerprints=True)
    assert records[0]["schema_version"] == TRACE_SCHEMA_VERSION
    assert chronological_key(records[0]) < chronological_key(records[1])
    assert records[0]["content"]["committed_token_ids"] == [4, 5, 6]


def test_record_normalization_and_fingerprints_are_stable() -> None:
    record = load_records()[0]
    assert normalize_trace_record(record) == normalize_trace_record(dict(reversed(record.items())))
    assert trace_record_fingerprint(record).startswith("sha256:")
    assert trace_stream_fingerprint(load_records()).startswith("sha256:")
    assert compatibility_fingerprints(record)["model_fingerprint"] == record["model_fingerprint"]


def test_hidden_reasoning_is_rejected_by_contract() -> None:
    record = load_records()[0]
    record["content"]["hidden_reasoning_token_ids"] = [99]
    with pytest.raises(ContractError, match="unknown"):
        validate_trace_record(record)

    record = load_records()[0]
    record["reasoning_retention"] = "retain"
    with pytest.raises(ContractError, match="must be exclude"):
        validate_trace_record(record)

    record = load_records()[0]
    record["generation_protocol"]["preserve_thinking"] = True
    with pytest.raises(ContractError, match="must remain false"):
        validate_trace_record(record)


def test_repositories_and_lineage_cannot_silently_diverge() -> None:
    record = load_records()[0]
    record["artifact_lineage"]["repository_id"] = "other-repository"
    with pytest.raises(ContractError, match="must match"):
        validate_trace_record(record)


def test_sequence_is_strictly_chronological_and_record_ids_unique() -> None:
    records = load_records()
    with pytest.raises(ContractError, match="strictly chronological"):
        validate_trace_sequence(list(reversed(records)))

    records = load_records()
    records[1]["record_id"] = records[0]["record_id"]
    with pytest.raises(ContractError, match="duplicate"):
        validate_trace_sequence(records)


def test_fingerprint_mismatch_is_explicit() -> None:
    record = load_records()[0]
    expected = compatibility_fingerprints(record)
    changed = copy.deepcopy(expected)
    changed["tokenizer_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ContractError, match="incompatible tokenizer_fingerprint"):
        validate_fingerprint_compatibility(record, changed)


def test_redacted_records_cannot_keep_token_ids() -> None:
    record = load_records()[0]
    record["record_state"] = "redacted"
    with pytest.raises(ContractError, match="must not retain token IDs"):
        validate_trace_record(record)
