"""Trusted local pre-forward request capture proxy for PCSS workload collection.

The proxy records only visible request bodies and fsyncs one event before it
forwards that request to the local upstream server.  It never records upstream
responses.  Each proxy process represents one UUID-backed work episode with an
explicit declared origin.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .project_workload_capture import (
    CAPTURE_ORIGINS,
    CAPTURE_TOOL_VERSION,
    CURRENT_SOURCE_REQUEST_KEY,
    ORDINARY_LOCAL_LLAMA_SERVER_ORIGIN,
    CaptureContractError,
    EVENT_SCHEMA,
    EVENT_SCHEMA_V2,
    SOURCE_CAPTURE_TOOL_VERSION,
    SOURCE_SIDECAR_EVENT_SCHEMA,
    _canonical_json,
    _event,
    _sha256,
    _source_sidecar_event,
    extract_capture_current_source,
    reject_capture_current_source,
    sidecar_hash_payload,
    validate_capture_origin,
)


MAX_REQUEST_BYTES = 1_048_576


def _durably_append(fd: int, payload: bytes, label: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise CaptureContractError(f"cannot durably append {label}")
        offset += written
    os.fsync(fd)


def _decode_request_body(raw: bytes) -> dict[str, Any]:
    """Reject duplicate JSON keys before preserving or forwarding a request."""

    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CaptureContractError(f"request JSON contains duplicate object key {key!r}")
            result[key] = value
        return result

    body = json.loads(raw, object_pairs_hook=object_without_duplicates)
    if not isinstance(body, dict):
        raise CaptureContractError("request body must be JSON object")
    return body


class PreForwardCaptureLog:
    def __init__(
        self,
        path: Path,
        episode_id: str | None = None,
        origin: str = ORDINARY_LOCAL_LLAMA_SERVER_ORIGIN,
        current_source_sidecar: Path | None = None,
    ) -> None:
        if episode_id is None:
            episode_id = str(uuid.uuid4())
        try:
            uuid.UUID(episode_id)
        except ValueError as error:
            raise CaptureContractError("episode_id must be an opaque UUID") from error
        origin = validate_capture_origin(origin)
        if path.exists():
            raise CaptureContractError("refusing to append to an existing capture log")
        if current_source_sidecar is not None:
            if current_source_sidecar.exists():
                raise CaptureContractError("refusing to append to an existing current-source sidecar")
            if current_source_sidecar.resolve() == path.resolve():
                raise CaptureContractError("current-source sidecar must differ from capture log")
        path.parent.mkdir(parents=True, exist_ok=True)
        if current_source_sidecar is not None:
            current_source_sidecar.parent.mkdir(parents=True, exist_ok=True)
        self._fd = -1
        self._sidecar_fd = -1
        try:
            if current_source_sidecar is not None:
                self._sidecar_fd = os.open(
                    current_source_sidecar,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except OSError as error:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            if self._sidecar_fd >= 0:
                os.close(self._sidecar_fd)
                self._sidecar_fd = -1
            raise CaptureContractError("cannot create private capture log") from error
        self.episode_id = episode_id
        self.origin = origin
        self.index = 0
        self.previous_event_sha256: str | None = None
        self.previous_sidecar_sha256: str | None = None
        self.sidecar_index = 0
        self._lock = threading.Lock()

    def begin_next_episode(self) -> str:
        with self._lock:
            self.episode_id = str(uuid.uuid4())
            return self.episode_id

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._sidecar_fd >= 0:
            os.close(self._sidecar_fd)
            self._sidecar_fd = -1

    def append(self, endpoint: str, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._sidecar_fd < 0:
                # Preserve the strict v1 rejection boundary before parsing a
                # source envelope that the caller did not authorize us to retain.
                reject_capture_current_source(body, "request body")
            public_body, source, commitment = extract_capture_current_source(body, "request body")
            if source is not None and self._sidecar_fd < 0:
                raise CaptureContractError(
                    "request-scoped current-source material requires a private source sidecar"
                )
            next_index = self.index + 1
            source_enabled = self._sidecar_fd >= 0
            event = {
                "schema_version": EVENT_SCHEMA_V2 if source_enabled else EVENT_SCHEMA,
                "event_id": str(uuid.uuid4()),
                "episode_id": self.episode_id,
                "capture_index": next_index,
                "captured_at": datetime.now(UTC).isoformat(),
                "origin": self.origin,
                "capture_provenance": "local_forward_proxy_pre_forward/v1",
                "capture_tool_version": (
                    SOURCE_CAPTURE_TOOL_VERSION if source_enabled else CAPTURE_TOOL_VERSION
                ),
                "previous_event_sha256": self.previous_event_sha256,
                "endpoint": endpoint,
                "body": public_body,
            }
            if source_enabled:
                event["pcss_current_source_commitment"] = commitment
            event["event_sha256"] = _sha256(_canonical_json(event))
            _event(event, next_index)
            next_sidecar_sha: str | None = None
            if source is not None:
                sidecar = {
                    "schema_version": SOURCE_SIDECAR_EVENT_SCHEMA,
                    "event_id": event["event_id"],
                    "capture_index": next_index,
                    "ledger_event_sha256": event["event_sha256"],
                    "previous_sidecar_sha256": self.previous_sidecar_sha256,
                    "pcss_current_source_commitment": commitment,
                    CURRENT_SOURCE_REQUEST_KEY: source,
                }
                sidecar["sidecar_event_sha256"] = _sha256(
                    _canonical_json(sidecar_hash_payload(sidecar))
                )
                _source_sidecar_event(sidecar, self.sidecar_index + 1)
                _durably_append(
                    self._sidecar_fd,
                    _canonical_json(sidecar),
                    "current-source sidecar event",
                )
                next_sidecar_sha = sidecar["sidecar_event_sha256"]
            _durably_append(self._fd, _canonical_json(event), "capture event")
            self.previous_event_sha256 = event["event_sha256"]
            self.index = next_index
            if next_sidecar_sha is not None:
                self.previous_sidecar_sha256 = next_sidecar_sha
                self.sidecar_index += 1
            return event


def _require_loopback(host: str, upstream: str) -> None:
    try:
        if not ipaddress.ip_address(host).is_loopback:
            raise ValueError
    except ValueError as error:
        raise CaptureContractError("capture proxy host must be a loopback IP address") from error
    parsed = urlparse(upstream)
    try:
        if parsed.scheme != "http" or parsed.username or parsed.password or not parsed.hostname or not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError
    except ValueError as error:
        raise CaptureContractError("capture proxy upstream must be an http loopback IP address") from error


def _next_response_chunk(response: Any) -> bytes:
    content_type = response.headers.get("Content-Type", "")
    if content_type.lower().startswith("text/event-stream"):
        return response.readline()
    read1 = getattr(response, "read1", None)
    if not callable(read1):
        read1 = getattr(getattr(response, "fp", None), "read1", None)
    if callable(read1):
        return read1(8192)
    # A one-byte fallback preserves first-byte/TTFT transparency when the
    # urllib wrapper does not expose a non-buffering read primitive.
    return response.read(1)


def _relay_response(response: Any, writer: Any) -> bool:
    while chunk := _next_response_chunk(response):
        try:
            writer.write(chunk)
            writer.flush()
        except (BrokenPipeError, ConnectionResetError):
            response.close()
            return False
    return True


def _handler(log: PreForwardCaptureLog, upstream: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover - avoid prompt/body logs
            del format, args

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/pcss/capture/begin-episode":
                    if int(self.headers.get("Content-Length", "0")) != 0:
                        raise CaptureContractError("episode rotation must not contain a request body")
                    payload = json.dumps({"episode_id": log.begin_next_episode()}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path not in {"/v1/completions", "/v1/chat/completions"}:
                    raise CaptureContractError("only completion endpoints are captured")
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > MAX_REQUEST_BYTES:
                    raise CaptureContractError("request body size is outside the capture bound")
                raw = self.rfile.read(size)
                body = _decode_request_body(raw)
                log.append(self.path, body)
                request = Request(
                    upstream.rstrip("/") + self.path,
                    data=raw,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    response = urlopen(request, timeout=120)
                    status = response.status
                except HTTPError as error:
                    response = error
                    status = error.code
                except URLError as error:
                    raise CaptureContractError(f"upstream unavailable after capture: {error.reason}") from error
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        self.send_header("Content-Length", content_length)
                    self.end_headers()
                    _relay_response(response, self.wfile)
                finally:
                    response.close()
            except (CaptureContractError, ValueError, json.JSONDecodeError) as error:
                payload = json.dumps({"error": {"message": str(error), "type": "capture_proxy_error"}}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-log", type=Path, required=True)
    parser.add_argument(
        "--origin",
        choices=sorted(CAPTURE_ORIGINS),
        required=True,
        help="declared provenance of all requests captured by this proxy process",
    )
    parser.add_argument("--episode-id", help="optional initial UUID; omit to have the proxy generate one")
    parser.add_argument(
        "--current-source-sidecar",
        type=Path,
        help="new private 0600 sidecar for request-scoped current-source envelopes",
    )
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)
    try:
        _require_loopback(args.host, args.upstream)
        log = PreForwardCaptureLog(
            args.capture_log,
            args.episode_id,
            args.origin,
            args.current_source_sidecar,
        )
    except CaptureContractError as error:
        parser.error(str(error))
    server = ThreadingHTTPServer((args.host, args.port), _handler(log, args.upstream))
    print(json.dumps({
        "episode_id": log.episode_id,
        "origin": log.origin,
        "capture_tool_version": (
            SOURCE_CAPTURE_TOOL_VERSION
            if args.current_source_sidecar is not None
            else CAPTURE_TOOL_VERSION
        ),
    }), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - operator action
        pass
    finally:
        server.server_close()
        log.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
