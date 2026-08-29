"""Versioned, fail-closed contracts for visible committed-token traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .canonical import (
    ContractError,
    canonical_json,
    require_exact_keys,
    require_int,
    require_mapping,
    require_string,
    sha256,
)


TRACE_SCHEMA_VERSION = "pcss.trace/v1"
FINGERPRINT_FIELDS = (
    "model_fingerprint",
    "tokenizer_fingerprint",
    "chat_template_fingerprint",
    "generation_protocol_fingerprint",
)

SCOPE_CLASSES = frozenset(
    {
        "current_context",
        "current_source",
        "same_session",
        "same_repository",
        "personal_persistent",
        "imported_corpus",
    }
)
SOURCE_CLASSES = frozenset(
    {
        "model_output",
        "visible_final",
        "visible_code",
        "tool_call",
        "tool_result",
        "source_file",
        "patch",
        "imported_corpus",
    }
)
PROVENANCE_FAMILIES = frozenset(
    {"session", "repository", "personal", "imported", "public"}
)
PRIVACY_CLASSES = frozenset({"private", "personal", "repository", "public", "restricted"})
RETENTION_CLASSES = frozenset(
    {"ephemeral", "session", "repository", "persistent", "licensed_import", "delete_on_request"}
)
OUTCOME_CLASSES = frozenset(
    {"unknown", "tests_passed", "patch_accepted", "tool_success", "failure"}
)
COMMITTED_TOKEN_STATUSES = frozenset({"none", "partial", "complete"})
RECORD_STATES = frozenset({"active", "redacted"})
DECODING_MODES = frozenset({"greedy", "sampling", "constrained"})
THINKING_MODES = frozenset({"on", "off"})
MAX_TOKEN_ID = (1 << 32) - 1

_TOP_LEVEL_KEYS = {
    "schema_version",
    "record_id",
    "timestamp_ns",
    "chronological_index",
    "request_id",
    "session_id",
    "trusted_scope",
    "artifact_lineage",
    "model_fingerprint",
    "tokenizer_fingerprint",
    "chat_template_fingerprint",
    "generation_protocol_fingerprint",
    "generation_protocol",
    "outcome_class",
    "source",
    "privacy",
    "reasoning_retention",
    "record_state",
    "content",
}
_TRUSTED_SCOPE_KEYS = {
    "scope_class",
    "principal_namespace",
    "application_id",
    "repository_id",
    "workflow_stage",
    "semantic_block_id",
}
_LINEAGE_KEYS = {
    "repository_id",
    "revision_family",
    "revision",
    "parent_revision",
    "path_hash",
    "before_blob",
    "after_blob",
    "source_span",
}
_SOURCE_SPAN_KEYS = {"start_byte", "end_byte"}
_GENERATION_PROTOCOL_KEYS = {
    "decoding_mode",
    "thinking_mode",
    "reasoning_effort",
    "preserve_thinking",
    "schema_or_tool_version",
}
_SOURCE_KEYS = {"source_class", "provenance_family", "eligible_for_persistent_store"}
_PRIVACY_KEYS = {"privacy_class", "retention_class"}
_CONTENT_KEYS = {"prompt_token_ids", "committed_token_ids", "committed_token_status"}


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    return value


def _require_nullable_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return require_string(value, field)


def _require_enum(value: Any, allowed: frozenset[str], field: str) -> str:
    item = require_string(value, field)
    if item not in allowed:
        raise ContractError(f"{field} is invalid")
    return item


def _require_fingerprint(value: Any, field: str) -> str:
    fingerprint = require_string(value, field)
    prefix, separator, digest = fingerprint.partition(":")
    if prefix != "sha256" or separator != ":" or len(digest) != 64:
        raise ContractError(f"{field} must be a sha256 fingerprint")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{field} must be a lowercase sha256 fingerprint")
    return fingerprint


def _validate_token_ids(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be an array")
    for index, token_id in enumerate(value):
        require_int(token_id, f"{field}[{index}]")
        if token_id > MAX_TOKEN_ID:
            raise ContractError(f"{field}[{index}] exceeds the u32 token range")
    return value


def _validate_trusted_scope(value: Any) -> Mapping[str, Any]:
    scope = require_mapping(value, "trace.trusted_scope")
    require_exact_keys(scope, _TRUSTED_SCOPE_KEYS, "trace.trusted_scope")
    scope_class = _require_enum(scope["scope_class"], SCOPE_CLASSES, "trace.trusted_scope.scope_class")
    require_string(scope["principal_namespace"], "trace.trusted_scope.principal_namespace")
    _require_nullable_string(scope["application_id"], "trace.trusted_scope.application_id")
    repository_id = _require_nullable_string(scope["repository_id"], "trace.trusted_scope.repository_id")
    _require_nullable_string(scope["workflow_stage"], "trace.trusted_scope.workflow_stage")
    _require_nullable_string(scope["semantic_block_id"], "trace.trusted_scope.semantic_block_id")
    if scope_class in {"current_source", "same_repository"} and repository_id is None:
        raise ContractError(f"trace.trusted_scope.repository_id is required for {scope_class}")
    return scope


def _validate_artifact_lineage(value: Any, trusted_scope: Mapping[str, Any]) -> None:
    lineage = require_mapping(value, "trace.artifact_lineage")
    require_exact_keys(lineage, _LINEAGE_KEYS, "trace.artifact_lineage")
    repository_id = _require_nullable_string(lineage["repository_id"], "trace.artifact_lineage.repository_id")
    scope_repository_id = trusted_scope["repository_id"]
    if repository_id != scope_repository_id:
        raise ContractError("trace.artifact_lineage.repository_id must match trace.trusted_scope.repository_id")

    lineage_details = (
        "revision_family",
        "revision",
        "parent_revision",
        "path_hash",
        "before_blob",
        "after_blob",
    )
    for key in lineage_details:
        item = _require_nullable_string(lineage[key], f"trace.artifact_lineage.{key}")
        if repository_id is None and item is not None:
            raise ContractError(f"trace.artifact_lineage.{key} requires a repository_id")

    source_span = lineage["source_span"]
    if source_span is None:
        return
    span = require_mapping(source_span, "trace.artifact_lineage.source_span")
    require_exact_keys(span, _SOURCE_SPAN_KEYS, "trace.artifact_lineage.source_span")
    start = require_int(span["start_byte"], "trace.artifact_lineage.source_span.start_byte")
    end = require_int(span["end_byte"], "trace.artifact_lineage.source_span.end_byte")
    if repository_id is None or lineage["path_hash"] is None:
        raise ContractError("trace.artifact_lineage.source_span requires repository_id and path_hash")
    if end < start:
        raise ContractError("trace.artifact_lineage.source_span.end_byte precedes start_byte")


def _validate_generation_protocol(value: Any) -> None:
    protocol = require_mapping(value, "trace.generation_protocol")
    require_exact_keys(protocol, _GENERATION_PROTOCOL_KEYS, "trace.generation_protocol")
    _require_enum(protocol["decoding_mode"], DECODING_MODES, "trace.generation_protocol.decoding_mode")
    _require_enum(protocol["thinking_mode"], THINKING_MODES, "trace.generation_protocol.thinking_mode")
    _require_nullable_string(protocol["reasoning_effort"], "trace.generation_protocol.reasoning_effort")
    if _require_bool(protocol["preserve_thinking"], "trace.generation_protocol.preserve_thinking"):
        raise ContractError("trace.generation_protocol.preserve_thinking must remain false")
    _require_nullable_string(protocol["schema_or_tool_version"], "trace.generation_protocol.schema_or_tool_version")


def _validate_source(value: Any) -> None:
    source = require_mapping(value, "trace.source")
    require_exact_keys(source, _SOURCE_KEYS, "trace.source")
    _require_enum(source["source_class"], SOURCE_CLASSES, "trace.source.source_class")
    _require_enum(source["provenance_family"], PROVENANCE_FAMILIES, "trace.source.provenance_family")
    _require_bool(source["eligible_for_persistent_store"], "trace.source.eligible_for_persistent_store")


def _validate_privacy(value: Any) -> None:
    privacy = require_mapping(value, "trace.privacy")
    require_exact_keys(privacy, _PRIVACY_KEYS, "trace.privacy")
    _require_enum(privacy["privacy_class"], PRIVACY_CLASSES, "trace.privacy.privacy_class")
    _require_enum(privacy["retention_class"], RETENTION_CLASSES, "trace.privacy.retention_class")


def _validate_content(value: Any, record_state: str) -> None:
    content = require_mapping(value, "trace.content")
    require_exact_keys(content, _CONTENT_KEYS, "trace.content")
    prompt_tokens = _validate_token_ids(content["prompt_token_ids"], "trace.content.prompt_token_ids")
    committed_tokens = _validate_token_ids(
        content["committed_token_ids"], "trace.content.committed_token_ids"
    )
    status = _require_enum(
        content["committed_token_status"], COMMITTED_TOKEN_STATUSES, "trace.content.committed_token_status"
    )
    if record_state == "active":
        if status == "none" and committed_tokens:
            raise ContractError("trace.content.committed_token_ids must be empty when status is none")
        if status != "none" and not committed_tokens:
            raise ContractError("trace.content.committed_token_ids are required for a committed status")
    elif prompt_tokens or committed_tokens:
        raise ContractError("redacted trace records must not retain token IDs")


def validate_trace_record(record: Mapping[str, Any]) -> None:
    """Validate one v1 trace record without silently accepting extra data.

    The ordinary v1 archive intentionally contains only visible, committed token
    streams. It does not provide an opt-in hidden-reasoning namespace.
    """

    trace = require_mapping(record, "trace")
    require_exact_keys(trace, _TOP_LEVEL_KEYS, "trace")
    if trace["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ContractError("unsupported trace.schema_version")
    require_string(trace["record_id"], "trace.record_id")
    require_int(trace["timestamp_ns"], "trace.timestamp_ns")
    require_int(trace["chronological_index"], "trace.chronological_index")
    require_string(trace["request_id"], "trace.request_id")
    require_string(trace["session_id"], "trace.session_id")

    trusted_scope = _validate_trusted_scope(trace["trusted_scope"])
    _validate_artifact_lineage(trace["artifact_lineage"], trusted_scope)
    for field in FINGERPRINT_FIELDS:
        _require_fingerprint(trace[field], f"trace.{field}")
    _validate_generation_protocol(trace["generation_protocol"])
    _require_enum(trace["outcome_class"], OUTCOME_CLASSES, "trace.outcome_class")
    _validate_source(trace["source"])
    _validate_privacy(trace["privacy"])
    if trace["reasoning_retention"] != "exclude":
        raise ContractError("trace.reasoning_retention must be exclude in the ordinary archive")
    record_state = _require_enum(trace["record_state"], RECORD_STATES, "trace.record_state")
    _validate_content(trace["content"], record_state)


def normalize_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical, key-sorted JSON representation as a fresh mapping."""

    validate_trace_record(record)
    return json.loads(canonical_json(record))


def compatibility_fingerprints(record: Mapping[str, Any]) -> dict[str, str]:
    """Return exactly the namespace fields that must not be silently mixed."""

    validate_trace_record(record)
    return {field: record[field] for field in FINGERPRINT_FIELDS}


def validate_fingerprint_compatibility(
    record: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Reject a record whose model/tokenizer/template/protocol namespace differs."""

    actual = compatibility_fingerprints(record)
    expected_mapping = require_mapping(expected, "expected_fingerprints")
    require_exact_keys(expected_mapping, set(FINGERPRINT_FIELDS), "expected_fingerprints")
    for field in FINGERPRINT_FIELDS:
        expected_value = _require_fingerprint(expected_mapping[field], f"expected_fingerprints.{field}")
        if actual[field] != expected_value:
            raise ContractError(f"incompatible {field}")


def chronological_key(record: Mapping[str, Any]) -> tuple[int, int]:
    """Return the deterministic ordering key used by chronological replay."""

    validate_trace_record(record)
    return (record["timestamp_ns"], record["chronological_index"])


def validate_trace_sequence(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_fingerprints: Mapping[str, Any] | None = None,
    require_uniform_fingerprints: bool = False,
) -> None:
    """Validate unique IDs and strict chronological order across trace records."""

    previous_key: tuple[int, int] | None = None
    seen_record_ids: set[str] = set()
    first_fingerprints: dict[str, str] | None = None
    for position, record in enumerate(records):
        validate_trace_record(record)
        record_id = record["record_id"]
        if record_id in seen_record_ids:
            raise ContractError(f"duplicate trace.record_id at position {position}")
        seen_record_ids.add(record_id)
        key = (record["timestamp_ns"], record["chronological_index"])
        if previous_key is not None and key <= previous_key:
            raise ContractError(f"trace sequence is not strictly chronological at position {position}")
        previous_key = key

        if expected_fingerprints is not None:
            validate_fingerprint_compatibility(record, expected_fingerprints)
        actual_fingerprints = compatibility_fingerprints(record)
        if first_fingerprints is None:
            first_fingerprints = actual_fingerprints
        elif require_uniform_fingerprints and actual_fingerprints != first_fingerprints:
            raise ContractError(f"incompatible fingerprints at position {position}")


def trace_record_fingerprint(record: Mapping[str, Any]) -> str:
    """Fingerprint a complete, normalized record for receipts and audit trails."""

    return f"sha256:{sha256(normalize_trace_record(record))}"


def trace_stream_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint a chronologically valid stream, including an empty post-delete stream."""

    validate_trace_sequence(records)
    normalized = [normalize_trace_record(record) for record in records]
    payload = {"schema_version": TRACE_SCHEMA_VERSION, "records": normalized}
    return f"sha256:{sha256(payload)}"
