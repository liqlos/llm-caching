"""Strict JSONL import and auditable record mutation for PCSS traces."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import ContractError, canonical_json, require_string, sha256
from .trace import (
    FINGERPRINT_FIELDS,
    TRACE_SCHEMA_VERSION,
    compatibility_fingerprints,
    normalize_trace_record,
    trace_record_fingerprint,
    trace_stream_fingerprint,
    validate_trace_sequence,
)


IMPORT_RECEIPT_SCHEMA_VERSION = "pcss.trace.import-receipt/v1"
MUTATION_RECEIPT_SCHEMA_VERSION = "pcss.trace.mutation-receipt/v1"


class TraceImportError(ContractError):
    """A contract error annotated with its JSONL import boundary."""


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def normalized_jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Encode validated records in the one deterministic JSONL spelling."""

    validate_trace_sequence(records)
    return b"".join(canonical_json(normalize_trace_record(record)).encode("utf-8") + b"\n" for record in records)


@dataclass(frozen=True)
class ImportResult:
    """A normalized stream plus the evidence needed to reproduce its import."""

    records: tuple[dict[str, Any], ...]
    raw_input_fingerprint: str
    normalized_stream_fingerprint: str
    fingerprints: dict[str, str]

    @property
    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": IMPORT_RECEIPT_SCHEMA_VERSION,
            "kind": "jsonl_import",
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "record_count": len(self.records),
            "raw_input_fingerprint": self.raw_input_fingerprint,
            "normalized_stream_fingerprint": self.normalized_stream_fingerprint,
            "fingerprints": dict(self.fingerprints),
        }


def import_jsonl_bytes(
    payload: bytes,
    *,
    expected_fingerprints: Mapping[str, Any] | None = None,
    require_uniform_fingerprints: bool = True,
) -> ImportResult:
    """Parse a JSONL payload fail-closed and return canonical records.

    Uniform fingerprints are mandatory by default so an import cannot silently
    combine incompatible tokenization or generation-protocol namespaces.
    """

    if not payload:
        raise TraceImportError("JSONL input is empty")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TraceImportError("JSONL input must be UTF-8") from error
    if text.startswith("\ufeff"):
        raise TraceImportError("JSONL input must not contain a UTF-8 BOM")

    records: list[dict[str, Any]] = []
    lines = text.splitlines()
    if not lines:
        raise TraceImportError("JSONL input is empty")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise TraceImportError(f"line {line_number}: blank lines are not allowed")
        try:
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (_DuplicateJsonKeyError, json.JSONDecodeError) as error:
            raise TraceImportError(f"line {line_number}: invalid JSONL object: {error}") from error
        if not isinstance(value, dict):
            raise TraceImportError(f"line {line_number}: JSONL value must be an object")
        try:
            records.append(normalize_trace_record(value))
        except ContractError as error:
            raise TraceImportError(f"line {line_number}: {error}") from error

    try:
        validate_trace_sequence(
            records,
            expected_fingerprints=expected_fingerprints,
            require_uniform_fingerprints=require_uniform_fingerprints,
        )
    except ContractError as error:
        raise TraceImportError(str(error)) from error
    if not records:
        raise TraceImportError("JSONL input contains no records")
    return ImportResult(
        records=tuple(records),
        raw_input_fingerprint=_sha256_bytes(payload),
        normalized_stream_fingerprint=trace_stream_fingerprint(records),
        fingerprints=compatibility_fingerprints(records[0]),
    )


def import_jsonl(
    path: str | os.PathLike[str],
    *,
    expected_fingerprints: Mapping[str, Any] | None = None,
    require_uniform_fingerprints: bool = True,
) -> ImportResult:
    """Read one UTF-8 JSONL file with no implicit recovery or coercion."""

    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise TraceImportError(f"unable to read JSONL input {source}: {error}") from error
    return import_jsonl_bytes(
        payload,
        expected_fingerprints=expected_fingerprints,
        require_uniform_fingerprints=require_uniform_fingerprints,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
        os.replace(temporary_path, path)
    except OSError as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        raise TraceImportError(f"unable to write {path}: {error}") from error


def write_normalized_jsonl(result: ImportResult, path: str | os.PathLike[str]) -> None:
    """Atomically materialize canonical JSONL after a successful import."""

    _atomic_write(Path(path), normalized_jsonl_bytes(result.records))


def write_receipt(receipt: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    """Write an import or mutation receipt as canonical machine-readable JSON."""

    _atomic_write(Path(path), canonical_json(dict(receipt)).encode("utf-8") + b"\n")


@dataclass(frozen=True)
class MutationReceipt:
    action: str
    record_id: str
    actor: str
    reason: str
    before_record_fingerprint: str
    after_record_fingerprint: str | None
    input_stream_fingerprint: str
    output_stream_fingerprint: str
    receipt_fingerprint: str

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MUTATION_RECEIPT_SCHEMA_VERSION,
            "action": self.action,
            "record_id": self.record_id,
            "actor": self.actor,
            "reason": self.reason,
            "before_record_fingerprint": self.before_record_fingerprint,
            "after_record_fingerprint": self.after_record_fingerprint,
            "input_stream_fingerprint": self.input_stream_fingerprint,
            "output_stream_fingerprint": self.output_stream_fingerprint,
            "receipt_fingerprint": self.receipt_fingerprint,
        }


@dataclass(frozen=True)
class MutationResult:
    records: tuple[dict[str, Any], ...]
    receipt: MutationReceipt


def _normalized_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_trace_record(record) for record in records]
    validate_trace_sequence(normalized)
    return normalized


def _record_position(records: Sequence[Mapping[str, Any]], record_id: str) -> int:
    for position, record in enumerate(records):
        if record["record_id"] == record_id:
            return position
    raise TraceImportError(f"trace record {record_id!r} was not found")


def _make_receipt(
    *,
    action: str,
    record_id: str,
    actor: str,
    reason: str,
    before_record_fingerprint: str,
    after_record_fingerprint: str | None,
    input_stream_fingerprint: str,
    output_stream_fingerprint: str,
) -> MutationReceipt:
    payload = {
        "action": action,
        "record_id": record_id,
        "actor": actor,
        "reason": reason,
        "before_record_fingerprint": before_record_fingerprint,
        "after_record_fingerprint": after_record_fingerprint,
        "input_stream_fingerprint": input_stream_fingerprint,
        "output_stream_fingerprint": output_stream_fingerprint,
    }
    fingerprint_payload = {"schema_version": MUTATION_RECEIPT_SCHEMA_VERSION, **payload}
    return MutationReceipt(
        **payload,
        receipt_fingerprint=f"sha256:{sha256(fingerprint_payload)}",
    )


def _mutate(
    records: Sequence[Mapping[str, Any]],
    *,
    record_id: str,
    action: str,
    actor: str,
    reason: str,
    replacement: Mapping[str, Any] | None,
) -> MutationResult:
    require_string(record_id, "record_id")
    require_string(actor, "actor")
    require_string(reason, "reason")
    current = _normalized_records(records)
    position = _record_position(current, record_id)
    original = current[position]
    input_stream_fingerprint = trace_stream_fingerprint(current)
    before_record_fingerprint = trace_record_fingerprint(original)

    if replacement is None:
        updated = current[:position] + current[position + 1 :]
        after_record_fingerprint = None
    else:
        normalized_replacement = normalize_trace_record(replacement)
        if normalized_replacement["record_id"] != original["record_id"]:
            raise TraceImportError("an edit must preserve trace.record_id")
        if (
            normalized_replacement["timestamp_ns"],
            normalized_replacement["chronological_index"],
        ) != (original["timestamp_ns"], original["chronological_index"]):
            raise TraceImportError("an edit must preserve the chronological ordering key")
        for field in FINGERPRINT_FIELDS:
            if normalized_replacement[field] != original[field]:
                raise TraceImportError(f"an edit must preserve {field}")
        updated = current[:]
        updated[position] = normalized_replacement
        after_record_fingerprint = trace_record_fingerprint(normalized_replacement)

    try:
        validate_trace_sequence(updated)
    except ContractError as error:
        raise TraceImportError(f"mutation produced an invalid stream: {error}") from error
    output_stream_fingerprint = trace_stream_fingerprint(updated)
    return MutationResult(
        records=tuple(updated),
        receipt=_make_receipt(
            action=action,
            record_id=record_id,
            actor=actor,
            reason=reason,
            before_record_fingerprint=before_record_fingerprint,
            after_record_fingerprint=after_record_fingerprint,
            input_stream_fingerprint=input_stream_fingerprint,
            output_stream_fingerprint=output_stream_fingerprint,
        ),
    )


def delete_record(
    records: Sequence[Mapping[str, Any]], *, record_id: str, actor: str, reason: str
) -> MutationResult:
    """Remove one record and return a deterministic receipt proving the delta."""

    return _mutate(
        records,
        record_id=record_id,
        action="delete",
        actor=actor,
        reason=reason,
        replacement=None,
    )


def redact_record(
    records: Sequence[Mapping[str, Any]], *, record_id: str, actor: str, reason: str
) -> MutationResult:
    """Replace a record's usable token payload with an auditable redaction."""

    current = _normalized_records(records)
    position = _record_position(current, record_id)
    replacement = copy.deepcopy(current[position])
    replacement["record_state"] = "redacted"
    replacement["content"]["prompt_token_ids"] = []
    replacement["content"]["committed_token_ids"] = []
    replacement["source"]["eligible_for_persistent_store"] = False
    return _mutate(
        current,
        record_id=record_id,
        action="redact",
        actor=actor,
        reason=reason,
        replacement=replacement,
    )


def edit_record(
    records: Sequence[Mapping[str, Any]],
    *,
    record_id: str,
    replacement: Mapping[str, Any],
    actor: str,
    reason: str,
) -> MutationResult:
    """Apply one validated, identity-preserving correction with an audit receipt."""

    return _mutate(
        records,
        record_id=record_id,
        action="edit",
        actor=actor,
        reason=reason,
        replacement=replacement,
    )
