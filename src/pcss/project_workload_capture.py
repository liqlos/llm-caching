"""Seal a prospectively captured project workload for PCSS evaluation.

The input is visible request metadata only.  It deliberately rejects responses,
expected outputs, hidden-token fields, and post-hoc labels so a later benchmark
cannot be tuned to its own answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence


EVENT_SCHEMA_V1 = "pcss.project-workload-capture-event/v1"
EVENT_SCHEMA_V2 = "pcss.project-workload-capture-event/v2"
EVENT_SCHEMA = EVENT_SCHEMA_V1
WORKLOAD_SCHEMA = "pcss.live-server-workload/v1"
RECEIPT_SCHEMA_V1 = "pcss.project-workload-capture-receipt/v1"
RECEIPT_SCHEMA_V2 = "pcss.project-workload-capture-receipt/v2"
RECEIPT_SCHEMA = RECEIPT_SCHEMA_V1
SOURCE_SIDECAR_EVENT_SCHEMA = "pcss.project-current-source-sidecar-event/v1"
SOURCE_COMMITMENT_SCHEMA = "pcss.current-source-capture-commitment/v1"
_EVENT_V1_KEYS = {
    "schema_version", "event_id", "episode_id", "capture_index", "captured_at",
    "origin", "capture_provenance", "capture_tool_version", "previous_event_sha256",
    "event_sha256", "endpoint", "body",
}
_EVENT_V2_KEYS = _EVENT_V1_KEYS | {"pcss_current_source_commitment"}
_SOURCE_SIDECAR_KEYS = {
    "schema_version", "event_id", "capture_index", "ledger_event_sha256",
    "previous_sidecar_sha256", "pcss_current_source_commitment",
    "pcss_current_source", "sidecar_event_sha256",
}
_SOURCE_COMMITMENT_KEYS = {
    "schema_version", "source_envelope_sha256", "token_count", "tokens_sha256",
}
_RECEIPT_V1_KEYS = {
    "schema_version", "source_event_count", "selected_event_count", "episodes", "per_episode",
    "source_sha256", "workload_sha256", "capture_tool_version", "origin",
    "request_episode_map", "final_event_sha256", "selection_rule", "contains_responses",
}
_RECEIPT_V2_KEYS = _RECEIPT_V1_KEYS | {
    "source_sidecar_sha256", "source_sidecar_event_count", "source_attachment_count",
    "source_attachment_commitments", "final_sidecar_event_sha256",
}
_CURRENT_SOURCE_KEYS = {
    "schema_version", "compatibility_fingerprint", "source_id", "visible_token_ids",
}
_FORBIDDEN_KEYS = {
    "answer", "candidate", "candidate_tokens", "committed_token_ids", "completion",
    "expected", "expected_output", "expected_text", "future", "future_tokens",
    "output", "output_text", "response", "target", "target_tokens",
}
_SENSITIVE_KEY_FRAGMENT = re.compile(
    r"(?:reason|hidden|logit|(?:^|_)kv(?:_|$)|(?:^|_)state(?:_|$)|token_ids?|completion|response|output|candidate|target|expected|future)",
    re.IGNORECASE,
)
CAPTURE_TOOL_VERSION = "pcss-project-capture-proxy/1"
SOURCE_CAPTURE_TOOL_VERSION = "pcss-project-capture-proxy/2"
ORDINARY_LOCAL_LLAMA_SERVER_ORIGIN = "ordinary_local_llama_server"
AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN = "agent_assisted_local_llama_server"
CAPTURE_ORIGINS = frozenset({
    ORDINARY_LOCAL_LLAMA_SERVER_ORIGIN,
    AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN,
})
CURRENT_SOURCE_REQUEST_KEY = "pcss_current_source"
CURRENT_SOURCE_SCHEMA = "pcss.current-source/v1"
CURRENT_SOURCE_MAX_TOKENS = 16_384


class CaptureContractError(ValueError):
    """The source is not a prospective visible-request capture."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_capture_origin(value: object, label: str = "origin") -> str:
    if not isinstance(value, str) or value not in CAPTURE_ORIGINS:
        supported = ", ".join(sorted(CAPTURE_ORIGINS))
        raise CaptureContractError(f"{label} must be one of: {supported}")
    return value


def reject_capture_current_source(value: object, label: str = "capture") -> None:
    """Reject request-scoped source buffers before any capture persistence.

    ``pcss_current_source`` is intentionally a local, ephemeral benchmark
    extension.  A pre-forward capture log is durable source material, so it
    must never contain the buffer, opaque source identifier, or compatibility
    metadata.  Search recursively because clients can nest arbitrary metadata
    in OpenAI-compatible request bodies.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() == CURRENT_SOURCE_REQUEST_KEY:
                raise CaptureContractError(
                    f"{label} contains forbidden request-scoped current-source material"
                )
            reject_capture_current_source(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_capture_current_source(child, f"{label}[{index}]")


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _sha256_value(value: object) -> str:
    return _sha256(_canonical_json(value))


def _compact_canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _source_commitment(source: Mapping[str, Any]) -> dict[str, Any]:
    tokens = source["visible_token_ids"]
    return {
        "schema_version": SOURCE_COMMITMENT_SCHEMA,
        "source_envelope_sha256": _sha256_value(source),
        "token_count": len(tokens),
        # This is deliberately the same compact canonical token digest retained
        # by the benchmark's redacted source receipt, so offline analysis can
        # bind the two public artifacts without reopening the sidecar.
        "tokens_sha256": _sha256(_compact_canonical_json(tokens)),
    }


def _current_source(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CURRENT_SOURCE_KEYS:
        raise CaptureContractError(f"{label} must contain exactly the current-source fields")
    if value["schema_version"] != CURRENT_SOURCE_SCHEMA:
        raise CaptureContractError(f"{label}.schema_version is unsupported")
    fingerprint = value["compatibility_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint:
        raise CaptureContractError(f"{label}.compatibility_fingerprint must be non-empty")
    source_id = value["source_id"]
    if (
        not isinstance(source_id, str)
        or not source_id.startswith("sha256:")
        or len(source_id) != len("sha256:") + 64
        or any(character not in "0123456789abcdef" for character in source_id[len("sha256:") :])
    ):
        raise CaptureContractError(f"{label}.source_id must be a lowercase SHA-256 identifier")
    raw_tokens = value["visible_token_ids"]
    if not isinstance(raw_tokens, list) or not raw_tokens or len(raw_tokens) > CURRENT_SOURCE_MAX_TOKENS:
        raise CaptureContractError(f"{label}.visible_token_ids is outside the bounded source contract")
    tokens: list[int] = []
    for index, token in enumerate(raw_tokens):
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token > 2_147_483_647
        ):
            raise CaptureContractError(f"{label}.visible_token_ids[{index}] is invalid")
        tokens.append(token)
    if source_id != "sha256:" + _sha256_value(tokens):
        raise CaptureContractError(f"{label}.source_id does not bind visible_token_ids")
    return {
        "schema_version": CURRENT_SOURCE_SCHEMA,
        "compatibility_fingerprint": fingerprint,
        "source_id": source_id,
        "visible_token_ids": tokens,
    }


def extract_capture_current_source(
    body: Mapping[str, Any], label: str = "request body"
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Split one top-level source extension from capture-safe visible body data.

    The durable ledger receives only the public body and a non-reversible
    commitment. The raw source envelope is returned solely to the caller for a
    private 0600 sidecar and must never be copied into the ledger or receipt.
    """

    if not isinstance(body, Mapping):
        raise CaptureContractError(f"{label} must be an object")
    public_body = dict(body)
    raw_source = public_body.pop(CURRENT_SOURCE_REQUEST_KEY, None)
    reject_capture_current_source(public_body, label)
    _reject_future(public_body, label)
    if raw_source is None:
        return public_body, None, None
    source = _current_source(raw_source, f"{label}.{CURRENT_SOURCE_REQUEST_KEY}")
    return public_body, source, _source_commitment(source)


def _reject_future(value: object, label: str = "capture") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.lower() in _FORBIDDEN_KEYS or _SENSITIVE_KEY_FRAGMENT.search(key):
                raise CaptureContractError(f"{label} contains forbidden response/future field {key!r}")
            _reject_future(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_future(child, f"{label}[{index}]")


def event_hash_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "event_sha256"}


def _validate_source_commitment(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_COMMITMENT_KEYS:
        raise CaptureContractError(f"{label} must contain exactly the source-commitment fields")
    if value["schema_version"] != SOURCE_COMMITMENT_SCHEMA:
        raise CaptureContractError(f"{label}.schema_version is unsupported")
    result: dict[str, Any] = {"schema_version": SOURCE_COMMITMENT_SCHEMA}
    for key in ("source_envelope_sha256", "tokens_sha256"):
        digest = value[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CaptureContractError(f"{label}.{key} is invalid")
        result[key] = digest
    count = value["token_count"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= CURRENT_SOURCE_MAX_TOKENS:
        raise CaptureContractError(f"{label}.token_count is outside the bounded source contract")
    result["token_count"] = count
    return result


def _event(value: object, line: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CaptureContractError(f"line {line} must contain a capture event object")
    schema = value.get("schema_version")
    expected_keys = {
        EVENT_SCHEMA_V1: _EVENT_V1_KEYS,
        EVENT_SCHEMA_V2: _EVENT_V2_KEYS,
    }.get(schema)
    if expected_keys is None or set(value) != expected_keys:
        raise CaptureContractError(f"line {line} must contain exactly the capture event fields")
    if schema not in {EVENT_SCHEMA_V1, EVENT_SCHEMA_V2}:
        raise CaptureContractError(f"line {line} has unsupported event schema")
    for key in ("event_id", "episode_id", "captured_at", "origin", "endpoint"):
        if not isinstance(value[key], str) or not value[key]:
            raise CaptureContractError(f"line {line}.{key} must be a non-empty string")
    validate_capture_origin(value["origin"], f"line {line}.origin")
    if value["capture_provenance"] != "local_forward_proxy_pre_forward/v1":
        raise CaptureContractError(f"line {line} was not captured by the trusted local forward proxy")
    expected_tool_version = (
        CAPTURE_TOOL_VERSION if schema == EVENT_SCHEMA_V1 else SOURCE_CAPTURE_TOOL_VERSION
    )
    if value["capture_tool_version"] != expected_tool_version:
        raise CaptureContractError(f"line {line} has unsupported capture tool version")
    for key in ("event_id", "episode_id"):
        try:
            uuid.UUID(value[key])
        except ValueError as error:
            raise CaptureContractError(f"line {line}.{key} must be an opaque UUID") from error
    previous = value["previous_event_sha256"]
    if previous is not None and (not isinstance(previous, str) or len(previous) != 64 or any(c not in "0123456789abcdef" for c in previous)):
        raise CaptureContractError(f"line {line}.previous_event_sha256 is invalid")
    event_sha = value["event_sha256"]
    if not isinstance(event_sha, str) or event_sha != _sha256(_canonical_json(event_hash_payload(value))):
        raise CaptureContractError(f"line {line}.event_sha256 does not bind the captured request")
    if not isinstance(value["capture_index"], int) or value["capture_index"] < 1:
        raise CaptureContractError(f"line {line}.capture_index must be a positive integer")
    if value["endpoint"] not in {"/v1/completions", "/v1/chat/completions"}:
        raise CaptureContractError(f"line {line}.endpoint is unsupported")
    if not isinstance(value["body"], dict):
        raise CaptureContractError(f"line {line}.body must be an object")
    reject_capture_current_source(value["body"], f"line {line}.body")
    _reject_future(value["body"], f"line {line}.body")
    if schema == EVENT_SCHEMA_V2 and value["pcss_current_source_commitment"] is not None:
        _validate_source_commitment(
            value["pcss_current_source_commitment"],
            f"line {line}.pcss_current_source_commitment",
        )
    if value["endpoint"] == "/v1/completions":
        if not isinstance(value["body"].get("prompt"), (str, list)):
            raise CaptureContractError(f"line {line} completion request needs visible prompt")
    else:
        if not isinstance(value["body"].get("messages"), list) or not value["body"]["messages"]:
            raise CaptureContractError(f"line {line} chat request needs visible messages")
    return dict(value)


def sidecar_hash_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "sidecar_event_sha256"}


def _sha256_field(value: object, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptureContractError(f"{label} is invalid")
    return value


def _source_sidecar_event(value: object, line: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_SIDECAR_KEYS:
        raise CaptureContractError(f"source sidecar line {line} must contain exactly the sidecar fields")
    if value["schema_version"] != SOURCE_SIDECAR_EVENT_SCHEMA:
        raise CaptureContractError(f"source sidecar line {line} has unsupported schema")
    for key in ("event_id",):
        if not isinstance(value[key], str):
            raise CaptureContractError(f"source sidecar line {line}.{key} must be a UUID")
        try:
            uuid.UUID(value[key])
        except ValueError as error:
            raise CaptureContractError(f"source sidecar line {line}.{key} must be a UUID") from error
    index = value["capture_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise CaptureContractError(f"source sidecar line {line}.capture_index must be positive")
    _sha256_field(value["ledger_event_sha256"], f"source sidecar line {line}.ledger_event_sha256")
    _sha256_field(
        value["previous_sidecar_sha256"],
        f"source sidecar line {line}.previous_sidecar_sha256",
        allow_none=True,
    )
    commitment = _validate_source_commitment(
        value["pcss_current_source_commitment"],
        f"source sidecar line {line}.pcss_current_source_commitment",
    )
    source = _current_source(
        value["pcss_current_source"],
        f"source sidecar line {line}.pcss_current_source",
    )
    if commitment != _source_commitment(source):
        raise CaptureContractError(f"source sidecar line {line} commitment does not bind source")
    sidecar_sha = value["sidecar_event_sha256"]
    if sidecar_sha != _sha256(_canonical_json(sidecar_hash_payload(value))):
        raise CaptureContractError(f"source sidecar line {line}.sidecar_event_sha256 does not bind source")
    return {
        "schema_version": SOURCE_SIDECAR_EVENT_SCHEMA,
        "event_id": value["event_id"],
        "capture_index": index,
        "ledger_event_sha256": value["ledger_event_sha256"],
        "previous_sidecar_sha256": value["previous_sidecar_sha256"],
        "pcss_current_source_commitment": commitment,
        "pcss_current_source": source,
        "sidecar_event_sha256": sidecar_sha,
    }


def _canonical_jsonl_sha256(events: Sequence[Mapping[str, Any]]) -> str:
    return _sha256(b"".join(_canonical_json(event) for event in events))


def _bound_source_sidecars(
    events: Sequence[Mapping[str, Any]],
    source_sidecar_events: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, dict[str, Any]], str, str | None]:
    """Return raw source only after both append-only chains bind exactly."""

    schemas = {event["schema_version"] for event in events}
    if len(schemas) != 1:
        raise CaptureContractError("capture cannot mix event schema versions")
    schema = next(iter(schemas))
    has_committed_source = any(
        event.get("pcss_current_source_commitment") is not None for event in events
    )
    if schema == EVENT_SCHEMA_V1:
        if source_sidecar_events is not None:
            raise CaptureContractError("v1 capture cannot have a current-source sidecar")
        return {}, _canonical_jsonl_sha256(()), None
    if schema != EVENT_SCHEMA_V2:  # _event already checks; retain a local guard.
        raise CaptureContractError("capture has unsupported event schema")
    if source_sidecar_events is None:
        required = "current-source capture" if has_committed_source else "v2 capture"
        raise CaptureContractError(f"{required} requires a private source sidecar")

    parsed_sidecars = [
        _source_sidecar_event(value, index + 1)
        for index, value in enumerate(source_sidecar_events or ())
    ]
    by_event_id = {event["event_id"]: event for event in events}
    committed_event_ids = {
        event["event_id"]
        for event in events
        if event["pcss_current_source_commitment"] is not None
    }
    attached: dict[str, dict[str, Any]] = {}
    previous_sidecar_sha: str | None = None
    previous_capture_index = 0
    for sidecar in parsed_sidecars:
        event_id = sidecar["event_id"]
        if event_id in attached:
            raise CaptureContractError("source sidecar event_id must be unique")
        if sidecar["previous_sidecar_sha256"] != previous_sidecar_sha:
            raise CaptureContractError("source sidecar hash chain is broken")
        if sidecar["capture_index"] <= previous_capture_index:
            raise CaptureContractError("source sidecar capture_index must be strictly chronological")
        ledger_event = by_event_id.get(event_id)
        if ledger_event is None:
            raise CaptureContractError("source sidecar contains an orphan event")
        if (
            ledger_event["capture_index"] != sidecar["capture_index"]
            or ledger_event["event_sha256"] != sidecar["ledger_event_sha256"]
            or ledger_event["pcss_current_source_commitment"]
            != sidecar["pcss_current_source_commitment"]
        ):
            raise CaptureContractError("source sidecar does not bind its public ledger event")
        attached[event_id] = sidecar["pcss_current_source"]
        previous_sidecar_sha = sidecar["sidecar_event_sha256"]
        previous_capture_index = sidecar["capture_index"]
    if set(attached) != committed_event_ids:
        raise CaptureContractError(
            "current-source ledger commitments and source sidecar entries must match exactly"
        )
    return attached, _canonical_jsonl_sha256(parsed_sidecars), previous_sidecar_sha


def seal_capture(
    events: Sequence[Mapping[str, Any]],
    *,
    source_sidecar_events: Sequence[Mapping[str, Any]] | None = None,
    episodes: int = 8,
    per_episode: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if episodes < 1 or per_episode < 1:
        raise CaptureContractError("episodes and per_episode must be positive")
    parsed = [_event(value, index + 1) for index, value in enumerate(events)]
    if not parsed:
        raise CaptureContractError("capture is empty")
    indices = [event["capture_index"] for event in parsed]
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        raise CaptureContractError("capture_index must be globally unique and strictly chronological")
    event_ids = [event["event_id"] for event in parsed]
    if len(set(event_ids)) != len(event_ids):
        raise CaptureContractError("event_id must be unique")
    previous_sha: str | None = None
    for event in parsed:
        if event["previous_event_sha256"] != previous_sha:
            raise CaptureContractError("capture hash chain is broken")
        previous_sha = event["event_sha256"]
    origins = {event["origin"] for event in parsed}
    if len(origins) != 1:
        raise CaptureContractError("capture must contain exactly one declared origin")
    origin = next(iter(origins))
    source_by_event_id, sidecar_digest, final_sidecar_sha = _bound_source_sidecars(
        parsed, source_sidecar_events
    )

    selected_episodes: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for event in parsed:
        episode_id = event["episode_id"]
        if episode_id not in selected_episodes and len(selected_episodes) < episodes:
            selected_episodes[episode_id] = []
        if episode_id in selected_episodes and len(selected_episodes[episode_id]) < per_episode:
            selected_episodes[episode_id].append(event)
    if len(selected_episodes) != episodes or any(len(rows) != per_episode for rows in selected_episodes.values()):
        raise CaptureContractError("capture does not contain the first required requests for every selected episode")

    selected = sorted(
        (event for rows in selected_episodes.values() for event in rows),
        key=lambda event: event["capture_index"],
    )
    requests: list[dict[str, Any]] = []
    attachment_commitments: dict[str, dict[str, Any]] = {}
    for event in selected:
        body = dict(event["body"])
        source = source_by_event_id.get(event["event_id"])
        if source is not None:
            body[CURRENT_SOURCE_REQUEST_KEY] = source
            attachment_commitments[event["event_id"]] = event[
                "pcss_current_source_commitment"
            ]
        requests.append(
            {
                "request_id": event["event_id"],
                "chronological_index": event["capture_index"],
                "endpoint": event["endpoint"],
                "body": body,
            }
        )
    if parsed[0]["schema_version"] == EVENT_SCHEMA_V2 and not attachment_commitments:
        raise CaptureContractError(
            "v2 capture selection must contain at least one current-source attachment"
        )
    workload = {
        "schema_version": WORKLOAD_SCHEMA,
        "requests": requests,
    }
    request_episode_map = dict(sorted(
        (event["event_id"], event["episode_id"])
        for event in selected
    ))
    if len(request_episode_map) != len(workload["requests"]):
        raise CaptureContractError("selected request-to-episode mapping must include every workload request exactly once")
    if list(request_episode_map) != sorted(request_episode_map):
        raise CaptureContractError("selected request-to-episode mapping must be canonical")
    for request, event in zip(workload["requests"], selected, strict=True):
        if request_episode_map.get(request["request_id"]) != event["episode_id"]:
            raise CaptureContractError("selected request-to-episode mapping does not bind the workload")
    source_digest = _sha256(_canonical_json(parsed))
    workload_digest = _sha256(_canonical_json(workload))
    origin_description = {
        ORDINARY_LOCAL_LLAMA_SERVER_ORIGIN: "ordinary local llama-server",
        AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN: "agent-assisted local llama-server",
    }[origin]
    receipt: dict[str, Any] = {
        "schema_version": (
            RECEIPT_SCHEMA_V1
            if parsed[0]["schema_version"] == EVENT_SCHEMA_V1
            else RECEIPT_SCHEMA_V2
        ),
        "source_event_count": len(parsed),
        "selected_event_count": len(selected),
        "episodes": list(selected_episodes),
        "per_episode": per_episode,
        "source_sha256": source_digest,
        "workload_sha256": workload_digest,
        "capture_tool_version": (
            CAPTURE_TOOL_VERSION
            if parsed[0]["schema_version"] == EVENT_SCHEMA_V1
            else SOURCE_CAPTURE_TOOL_VERSION
        ),
        "origin": origin,
        "request_episode_map": request_episode_map,
        "final_event_sha256": previous_sha,
        "selection_rule": f"first visible pre-forward {origin_description} requests from first captured episodes, preserved chronologically",
        "contains_responses": False,
    }
    if parsed[0]["schema_version"] == EVENT_SCHEMA_V2:
        receipt.update(
            {
                "source_sidecar_sha256": sidecar_digest,
                "source_sidecar_event_count": len(source_by_event_id),
                "source_attachment_count": len(attachment_commitments),
                "source_attachment_commitments": dict(sorted(attachment_commitments.items())),
                "final_sidecar_event_sha256": final_sidecar_sha,
            }
        )
    return workload, receipt


def _read_jsonl(path: Path, label: str = "capture input") -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CaptureContractError(f"cannot read {label}: {path}") from error
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise CaptureContractError(f"line {line_number} must not be blank")
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise CaptureContractError(f"line {line_number} is not JSON") from error
    return events


def _write_new_file(path: Path, payload: bytes, mode: int, label: str) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    except FileExistsError as error:
        raise CaptureContractError(f"refusing to overwrite {label}") from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise CaptureContractError(f"cannot write sealed {label}")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    path.chmod(mode)


def seal_paths(
    input_path: Path,
    workload_path: Path,
    receipt_path: Path,
    *,
    source_sidecar_path: Path | None = None,
    episodes: int = 8,
    per_episode: int = 4,
) -> dict[str, Any]:
    if workload_path.exists() or receipt_path.exists():
        raise CaptureContractError("refusing to overwrite a sealed workload or receipt")
    source_sidecars: list[dict[str, Any]] | None = None
    if source_sidecar_path is not None:
        try:
            sidecar_mode = source_sidecar_path.stat().st_mode & 0o777
        except OSError as error:
            raise CaptureContractError(f"cannot stat source sidecar: {source_sidecar_path}") from error
        if sidecar_mode & 0o077:
            raise CaptureContractError("source sidecar must not be group/world accessible")
        source_sidecars = _read_jsonl(source_sidecar_path, "source sidecar")
    workload, receipt = seal_capture(
        _read_jsonl(input_path),
        source_sidecar_events=source_sidecars,
        episodes=episodes,
        per_episode=per_episode,
    )
    workload_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    workload_mode = 0o600 if receipt.get("source_attachment_count", 0) > 0 else 0o444
    _write_new_file(workload_path, _canonical_json(workload), workload_mode, "workload")
    _write_new_file(receipt_path, _canonical_json(receipt), 0o444, "receipt")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--source-sidecar",
        type=Path,
        help="private 0600 current-source sidecar emitted by the capture proxy",
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--per-episode", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        receipt = seal_paths(
            args.input,
            args.workload,
            args.receipt,
            source_sidecar_path=args.source_sidecar,
            episodes=args.episodes,
            per_episode=args.per_episode,
        )
    except CaptureContractError as error:
        parser.error(str(error))
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
