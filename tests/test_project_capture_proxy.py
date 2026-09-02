import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from pcss import project_capture_proxy as capture_proxy
from pcss.project_capture_proxy import PreForwardCaptureLog, _handler, _relay_response, _require_loopback, main
from pcss.project_workload_capture import (
    AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN,
    CaptureContractError,
    _canonical_json,
    seal_capture,
    sidecar_hash_payload,
)


def current_source_payload() -> dict:
    tokens = [7, 11, 13]
    return {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "private-source-compatibility",
        "source_id": "sha256:" + hashlib.sha256(_canonical_json(tokens)).hexdigest(),
        "visible_token_ids": tokens,
    }


def test_proxy_log_generates_opaque_metadata_and_seals_before_forward(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    log = PreForwardCaptureLog(path, "00000000-0000-4000-8000-000000000001")
    event = log.append("/v1/completions", {"prompt": "ordinary visible request"})
    log.close()
    stored = json.loads(path.read_text())
    assert stored == event
    assert event["capture_provenance"] == "local_forward_proxy_pre_forward/v1"
    assert event["episode_id"] == "00000000-0000-4000-8000-000000000001"
    assert path.stat().st_mode & 0o777 == 0o600


def test_proxy_log_carries_declared_agent_assisted_origin_into_sealed_receipt(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    episode_id = "00000000-0000-4000-8000-000000000001"
    log = PreForwardCaptureLog(path, episode_id, AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN)
    event = log.append("/v1/completions", {"prompt": "agent-assisted visible request"})
    log.close()

    workload, receipt = seal_capture([event], episodes=1, per_episode=1)

    assert event["origin"] == AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN
    assert receipt["origin"] == AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN
    assert receipt["request_episode_map"] == {workload["requests"][0]["request_id"]: episode_id}


def test_proxy_rejects_unknown_origin_before_creating_capture_log(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    with pytest.raises(CaptureContractError, match="must be one of"):
        PreForwardCaptureLog(path, origin="unknown")
    assert not path.exists()


def test_proxy_cli_requires_explicit_origin(tmp_path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--capture-log", str(tmp_path / "capture.jsonl")])
    assert error.value.code == 2


def test_proxy_cli_propagates_declared_origin(tmp_path, monkeypatch, capsys) -> None:
    class StoppedServer:
        def __init__(self, *args) -> None:
            self.args = args

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    monkeypatch.setattr("pcss.project_capture_proxy.ThreadingHTTPServer", StoppedServer)

    assert main([
        "--capture-log", str(tmp_path / "capture.jsonl"),
        "--origin", AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN,
    ]) == 0

    announced = json.loads(capsys.readouterr().out)
    assert announced["origin"] == AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN


def test_proxy_log_refuses_future_fields_before_writing(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    log = PreForwardCaptureLog(path, "00000000-0000-4000-8000-000000000001")
    with pytest.raises(CaptureContractError, match="forbidden"):
        log.append("/v1/completions", {"prompt": "ordinary", "output_text": "forbidden"})
    log.close()
    assert path.read_text() == ""


def test_proxy_log_rejects_current_source_before_writing(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    log = PreForwardCaptureLog(path, "00000000-0000-4000-8000-000000000001")
    source = current_source_payload()

    with pytest.raises(CaptureContractError, match="request-scoped current-source"):
        log.append("/v1/completions", {"prompt": "ordinary", "pcss_current_source": source})

    log.close()
    assert path.read_text() == ""
    assert log.index == 0
    assert source["source_id"] not in path.read_text()
    assert "visible_token_ids" not in path.read_text()


def test_proxy_opt_in_sidecar_seals_source_without_leaking_it_to_ledger_or_receipt(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(
        ledger_path,
        "00000000-0000-4000-8000-000000000001",
        current_source_sidecar=sidecar_path,
    )
    source = current_source_payload()
    source_event = log.append(
        "/v1/completions",
        {"prompt": "ordinary request with private source", "pcss_current_source": source},
    )
    ordinary_event = log.append("/v1/completions", {"prompt": "ordinary request"})
    log.close()

    ledger_text = ledger_path.read_text()
    sidecar_rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    ledger_rows = [json.loads(line) for line in ledger_text.splitlines()]
    workload, receipt = seal_capture(
        ledger_rows,
        source_sidecar_events=sidecar_rows,
        episodes=1,
        per_episode=2,
    )

    assert source_event["schema_version"].endswith("/v2")
    assert ordinary_event["pcss_current_source_commitment"] is None
    assert sidecar_path.stat().st_mode & 0o777 == 0o600
    assert sidecar_rows[0]["pcss_current_source"] == source
    assert source["source_id"] not in ledger_text
    assert source["compatibility_fingerprint"] not in ledger_text
    assert "visible_token_ids" not in ledger_text
    assert workload["requests"][0]["body"]["pcss_current_source"] == source
    receipt_text = json.dumps(receipt, sort_keys=True)
    assert receipt["schema_version"].endswith("/v2")
    assert receipt["source_attachment_count"] == 1
    assert source["source_id"] not in receipt_text
    assert source["compatibility_fingerprint"] not in receipt_text
    assert "visible_token_ids" not in receipt_text


def test_proxy_sidecar_write_failure_never_appends_public_ledger_event(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    calls: list[str] = []

    def fail_sidecar(_fd: int, _payload: bytes, label: str) -> None:
        calls.append(label)
        if label == "current-source sidecar event":
            raise CaptureContractError("injected sidecar failure")

    monkeypatch.setattr("pcss.project_capture_proxy._durably_append", fail_sidecar)
    with pytest.raises(CaptureContractError, match="injected sidecar failure"):
        log.append(
            "/v1/completions",
            {"prompt": "ordinary request", "pcss_current_source": current_source_payload()},
        )
    log.close()

    assert calls == ["current-source sidecar event"]
    assert ledger_path.read_text() == ""
    assert sidecar_path.read_text() == ""


def test_proxy_ledger_write_failure_stops_after_private_sidecar_fsync(tmp_path, monkeypatch) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    original_append = capture_proxy._durably_append
    calls: list[str] = []

    def fail_ledger(fd: int, payload: bytes, label: str) -> None:
        calls.append(label)
        if label == "capture event":
            raise CaptureContractError("injected ledger failure")
        original_append(fd, payload, label)

    monkeypatch.setattr("pcss.project_capture_proxy._durably_append", fail_ledger)
    with pytest.raises(CaptureContractError, match="injected ledger failure"):
        log.append(
            "/v1/completions",
            {"prompt": "ordinary request", "pcss_current_source": current_source_payload()},
        )
    log.close()

    assert calls == ["current-source sidecar event", "capture event"]
    assert ledger_path.read_text() == ""
    assert sidecar_path.read_text()
    assert log.index == 0


def test_proxy_sidecar_chain_tampering_is_rejected_by_the_sealer(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    log.append("/v1/completions", {"prompt": "first", "pcss_current_source": current_source_payload()})
    log.append("/v1/completions", {"prompt": "second", "pcss_current_source": current_source_payload()})
    log.close()
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    sidecar_rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    sidecar_rows[1]["previous_sidecar_sha256"] = None
    sidecar_rows[1]["sidecar_event_sha256"] = hashlib.sha256(
        _canonical_json(sidecar_hash_payload(sidecar_rows[1]))
    ).hexdigest()

    with pytest.raises(CaptureContractError, match="sidecar hash chain"):
        seal_capture(
            ledger_rows,
            source_sidecar_events=sidecar_rows,
            episodes=1,
            per_episode=2,
        )


def test_proxy_rejects_duplicate_json_keys_before_capture_or_forwarding(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    log = PreForwardCaptureLog(ledger_path)
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0), _handler(log, "http://127.0.0.1:1")
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/completions",
            data=b'{"prompt":"first","prompt":"second"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
        assert "duplicate object key" in error.value.read().decode()
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(3)
        log.close()

    assert ledger_path.read_text() == ""


def test_proxy_log_rejects_case_insensitive_hidden_internal_fields(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    log = PreForwardCaptureLog(path, "00000000-0000-4000-8000-000000000001")
    with pytest.raises(CaptureContractError, match="forbidden"):
        log.append("/v1/completions", {"prompt": "ordinary", "Reasoning_Content": "private"})
    log.close()


def test_proxy_generates_and_rotates_opaque_episode_ids(tmp_path) -> None:
    log = PreForwardCaptureLog(tmp_path / "episode.jsonl")
    first = log.episode_id
    second = log.begin_next_episode()
    log.close()
    assert first != second
    assert len(first) == len(second) == 36


def test_proxy_rotation_is_an_ordered_boundary_between_appends(tmp_path) -> None:
    log = PreForwardCaptureLog(tmp_path / "episode.jsonl")
    before = log.append("/v1/completions", {"prompt": "before"})
    next_episode = log.begin_next_episode()
    after = log.append("/v1/completions", {"prompt": "after"})
    log.close()
    assert before["episode_id"] != next_episode
    assert after["episode_id"] == next_episode
    assert before["event_sha256"] == after["previous_event_sha256"]


def test_proxy_serializes_concurrent_pre_forward_events_into_one_valid_chain(tmp_path) -> None:
    path = tmp_path / "episode.jsonl"
    log = PreForwardCaptureLog(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda index: log.append("/v1/completions", {"prompt": f"ordinary {index}"}), range(16)))
    log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    workload, receipt = seal_capture(rows, episodes=1, per_episode=16)
    assert len(workload["requests"]) == 16
    assert receipt["final_event_sha256"] == rows[-1]["event_sha256"]


def test_proxy_requires_loopback_bind_and_upstream() -> None:
    _require_loopback("127.0.0.1", "http://127.0.0.1:8080")
    with pytest.raises(CaptureContractError, match="loopback"):
        _require_loopback("0.0.0.0", "http://127.0.0.1:8080")
    with pytest.raises(CaptureContractError, match="loopback"):
        _require_loopback("127.0.0.1", "http://192.168.1.9:8080")


class _DelayedSseResponse:
    def __init__(self) -> None:
        self.headers = {"Content-Type": "text/event-stream"}
        self.lines = [b"data: first\n\n", b"data: second\n\n", b""]
        self.readline_calls = 0
        self.closed = False

    def readline(self) -> bytes:
        value = self.lines[self.readline_calls]
        self.readline_calls += 1
        return value

    def close(self) -> None:
        self.closed = True


class _Writer:
    def __init__(self, *, disconnect: bool = False) -> None:
        self.parts: list[bytes] = []
        self.disconnect = disconnect
        self.flushes = 0

    def write(self, value: bytes) -> None:
        if self.disconnect:
            raise BrokenPipeError
        self.parts.append(value)

    def flush(self) -> None:
        self.flushes += 1


def test_sse_relay_forwards_first_event_incrementally_before_next_read() -> None:
    response = _DelayedSseResponse()
    writer = _Writer()
    assert _relay_response(response, writer)
    assert writer.parts[0] == b"data: first\n\n"
    assert writer.parts[1] == b"data: second\n\n"
    assert response.readline_calls == 3


def test_relay_closes_upstream_promptly_on_client_disconnect() -> None:
    response = _DelayedSseResponse()
    writer = _Writer(disconnect=True)
    assert not _relay_response(response, writer)
    assert response.closed
    assert response.readline_calls == 1


def test_loopback_proxy_streams_first_sse_event_and_seals_request_before_completion(tmp_path) -> None:
    release_second = threading.Event()
    upstream_first_sent = threading.Event()

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: first\n\n")
            self.wfile.flush()
            upstream_first_sent.set()
            assert release_second.wait(3)
            self.wfile.write(b"data: second\n\n")
            self.wfile.flush()

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    log = PreForwardCaptureLog(tmp_path / "capture.jsonl")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), _handler(log, f"http://127.0.0.1:{upstream.server_port}"))
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    first_received = threading.Event()
    received: list[bytes] = []

    def client() -> None:
        request = Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/completions",
            data=b'{"prompt":"ordinary streamed request","stream":true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            received.append(response.readline())
            first_received.set()
            received.extend(response.readlines())

    client_thread = threading.Thread(target=client, daemon=True)
    client_thread.start()
    try:
        assert upstream_first_sent.wait(3)
        assert first_received.wait(3)
        assert received[0] == b"data: first\n"
        release_second.set()
        client_thread.join(3)
        assert not client_thread.is_alive()
        assert b"data: second\n" in received
        rows = [json.loads(line) for line in (tmp_path / "capture.jsonl").read_text().splitlines()]
        assert rows[0]["body"] == {"prompt": "ordinary streamed request", "stream": True}
    finally:
        release_second.set()
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(3)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(3)
        log.close()


def test_loopback_proxy_fsyncs_private_source_sidecar_before_forwarding(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    source = current_source_payload()
    observed: dict[str, object] = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            del format, args

        def do_POST(self) -> None:  # noqa: N802
            observed["request"] = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            observed["ledger_before_forward"] = ledger_path.read_text()
            observed["sidecar_before_forward"] = sidecar_path.read_text()
            payload = b'{"choices":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    log = PreForwardCaptureLog(
        ledger_path,
        current_source_sidecar=sidecar_path,
    )
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(log, f"http://127.0.0.1:{upstream.server_port}"),
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/completions",
            data=json.dumps(
                {"prompt": "ordinary source-bearing request", "pcss_current_source": source}
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assert response.read() == b'{"choices":[]}'
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(3)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(3)
        log.close()

    assert observed["request"] == {
        "prompt": "ordinary source-bearing request",
        "pcss_current_source": source,
    }
    assert source["source_id"] not in observed["ledger_before_forward"]
    assert source["source_id"] in observed["sidecar_before_forward"]
    assert "visible_token_ids" not in observed["ledger_before_forward"]
