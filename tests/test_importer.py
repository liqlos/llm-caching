from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pcss.importer import (
    TraceImportError,
    delete_record,
    edit_record,
    import_jsonl,
    import_jsonl_bytes,
    normalized_jsonl_bytes,
    redact_record,
    write_normalized_jsonl,
    write_receipt,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "traces"
VALID_FIXTURE = FIXTURES / "chronological-v1.jsonl"
INCOMPATIBLE_FIXTURE = FIXTURES / "incompatible-fingerprints-v1.jsonl"


def test_importer_normalizes_jsonl_and_records_receipt(tmp_path: Path) -> None:
    result = import_jsonl(VALID_FIXTURE)
    assert len(result.records) == 2
    assert result.receipt["record_count"] == 2
    assert result.receipt["raw_input_fingerprint"].startswith("sha256:")
    assert result.receipt["normalized_stream_fingerprint"] == result.normalized_stream_fingerprint

    output = tmp_path / "normalized.jsonl"
    write_normalized_jsonl(result, output)
    assert output.read_bytes() == normalized_jsonl_bytes(result.records)
    assert import_jsonl(output).normalized_stream_fingerprint == result.normalized_stream_fingerprint


def test_importer_rejects_out_of_order_records() -> None:
    lines = VALID_FIXTURE.read_bytes().splitlines()
    with pytest.raises(TraceImportError, match="strictly chronological"):
        import_jsonl_bytes(b"\n".join(reversed(lines)) + b"\n")


def test_importer_rejects_unknown_fields_and_duplicate_keys() -> None:
    record = json.loads(VALID_FIXTURE.read_text().splitlines()[0])
    record["untracked"] = True
    with pytest.raises(TraceImportError, match="unknown"):
        import_jsonl_bytes(json.dumps(record).encode() + b"\n")

    duplicate = VALID_FIXTURE.read_text().splitlines()[0][:-1] + ',"record_id":"duplicate"}'
    with pytest.raises(TraceImportError, match="duplicate JSON key"):
        import_jsonl_bytes(duplicate.encode() + b"\n")


def test_importer_rejects_incompatible_fingerprints_by_default() -> None:
    with pytest.raises(TraceImportError, match="incompatible fingerprints"):
        import_jsonl(INCOMPATIBLE_FIXTURE)


def test_importer_checks_an_expected_fingerprint_set() -> None:
    valid = import_jsonl(VALID_FIXTURE)
    expected = dict(valid.fingerprints)
    expected["generation_protocol_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(TraceImportError, match="incompatible generation_protocol_fingerprint"):
        import_jsonl(VALID_FIXTURE, expected_fingerprints=expected)


def test_delete_redact_and_edit_have_auditable_receipts(tmp_path: Path) -> None:
    result = import_jsonl(VALID_FIXTURE)
    deleted = delete_record(result.records, record_id="trace-0002", actor="test", reason="withdrawn")
    assert [record["record_id"] for record in deleted.records] == ["trace-0001"]
    assert deleted.receipt.as_dict["action"] == "delete"
    assert deleted.receipt.after_record_fingerprint is None

    redacted = redact_record(result.records, record_id="trace-0001", actor="test", reason="privacy request")
    replacement = redacted.records[0]
    assert replacement["record_state"] == "redacted"
    assert replacement["content"]["committed_token_ids"] == []
    assert replacement["source"]["eligible_for_persistent_store"] is False
    assert redacted.receipt.before_record_fingerprint != redacted.receipt.after_record_fingerprint
    assert redacted.receipt.as_dict["receipt_fingerprint"].startswith("sha256:")

    edited_record = copy.deepcopy(result.records[1])
    edited_record["privacy"]["retention_class"] = "delete_on_request"
    edited = edit_record(
        result.records,
        record_id="trace-0002",
        replacement=edited_record,
        actor="test",
        reason="correct retention class",
    )
    assert edited.records[1]["privacy"]["retention_class"] == "delete_on_request"
    receipt_path = tmp_path / "receipt.json"
    write_receipt(edited.receipt.as_dict, receipt_path)
    assert json.loads(receipt_path.read_text())["action"] == "edit"


def test_edit_cannot_rewrite_identity_order_or_compatibility_namespace() -> None:
    result = import_jsonl(VALID_FIXTURE)
    replacement = copy.deepcopy(result.records[0])
    replacement["model_fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(TraceImportError, match="preserve model_fingerprint"):
        edit_record(
            result.records,
            record_id="trace-0001",
            replacement=replacement,
            actor="test",
            reason="bad correction",
        )
