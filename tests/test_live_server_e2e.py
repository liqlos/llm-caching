from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcss import live_server_e2e as e2e


class _Headers:
    def get(self, name: str) -> str:
        return "application/json" if name == "Content-Type" else ""


class _Response:
    def __init__(
        self, body: bytes, *, status: int = 200, first_line: bytes | None = None
    ) -> None:
        self._body = body
        self._first_line = first_line if first_line is not None else body
        self.status = status
        self.headers = _Headers()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def readline(self) -> bytes:
        return self._first_line


def _config(tmp_path: Path) -> e2e.QualificationConfig:
    server = tmp_path / "fake-server"
    server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    return e2e.QualificationConfig(
        server=server,
        model=model,
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        patch_sha256="a" * 64,
        output=tmp_path / "result.json",
        startup_timeout_seconds=1,
        request_timeout_seconds=1,
        context_size=512,
    )


def test_validate_config_checks_explicit_model_hash_and_executable(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    hashes = e2e._validate_config(config)
    assert hashes["model_sha256"] == config.model_sha256
    assert hashes["patch_sha256"] == "a" * 64
    assert (
        hashes["server_sha256"]
        == hashlib.sha256(config.server.read_bytes()).hexdigest()
    )

    wrong = e2e.QualificationConfig(**{**config.__dict__, "model_sha256": "b" * 64})
    with pytest.raises(e2e.QualificationError, match="model SHA-256 mismatch"):
        e2e._validate_config(wrong)


def test_http_completion_metrics_and_explicit_cancel_are_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def fake_urlopen(request: object, timeout: float) -> _Response:
        assert timeout == 2
        method = request.get_method()  # type: ignore[attr-defined]
        url = request.full_url  # type: ignore[attr-defined]
        conversation = request.headers.get("X-conversation-id")  # type: ignore[attr-defined]
        calls.append((method, url, conversation))
        if url.endswith("/metrics"):
            return _Response(b"llamacpp:pcss_records_learned_total 3\n", status=200)
        if method == "DELETE":
            return _Response(b"", status=204)
        if conversation:
            return _Response(b"", status=200, first_line=b'data: {"choices":[]}\n')
        return _Response(json.dumps({"choices": [{"text": " ok"}]}).encode())

    monkeypatch.setattr(e2e, "urlopen", fake_urlopen)
    server = e2e._Server("http://127.0.0.1:1234", 2)
    assert server.completion("hello")["output_text"] == " ok"
    assert e2e._metric(server.metrics(), "pcss_records_learned_total") == 3
    cancellation = server.cancel_completion("long")
    assert cancellation["stop"]["status"] == 204
    assert calls[-1][0] == "DELETE"
    assert "/v1/stream?conv_id=pcss-e2e-" in calls[-1][1]
    assert calls[-2][2].startswith("pcss-e2e-")


def test_harness_command_is_loopback_and_pcss_controls_are_explicit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    harness = e2e._Harness(config, tmp_path)
    command = harness._command(
        4567,
        store=tmp_path / "store",
        parallel=2,
        read_only=True,
        disable_learning=True,
    )
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--parallel") + 1] == "2"
    assert "--pcss-read-only" in command
    assert "--pcss-disable-learning" in command
    assert "--metrics" in command


def test_harness_uses_fake_process_and_captures_cleanup_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    popen_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pid = 321
        returncode = None

        def poll(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        popen_calls.append((command, kwargs))
        stdout = kwargs["stdout"]
        stdout.write(b"fake server log\n")  # type: ignore[union-attr]
        return FakeProcess()

    monkeypatch.setattr(e2e.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        e2e,
        "_http_json",
        lambda *args, **kwargs: {"status": 200, "json": {"status": "ok"}},
    )
    monkeypatch.setattr(
        e2e,
        "_terminate_group",
        lambda process: {
            "action": "terminate_group",
            "returncode": 0,
            "process_group": process.pid,
            "descendants_absent": True,
        },
    )
    harness = e2e._Harness(config, tmp_path)
    with harness.server("fake", store=tmp_path / "store") as server:
        assert server.base_url.startswith("http://127.0.0.1:")
        assert harness.active == 1
    assert harness.active == 0
    assert harness.max_active == 1
    assert harness.launches[0]["cleanup"]["descendants_absent"] is True
    assert harness.launches[0]["log"]["captured_utf8"] == "fake server log\n"
    assert popen_calls[0][1]["start_new_session"] is True


def test_harness_refuses_a_third_live_server(tmp_path: Path) -> None:
    harness = e2e._Harness(_config(tmp_path), tmp_path)
    harness.active = 2
    with pytest.raises(e2e.QualificationError, match="more than two live servers"):
        with harness.server("third"):
            pass


def test_harness_never_reuses_a_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ports = iter((4000, 4000, 4001))
    monkeypatch.setattr(e2e, "_reserve_loopback_port", lambda: next(ports))
    harness = e2e._Harness(_config(tmp_path), tmp_path)
    assert harness._fresh_port() == 4000
    assert harness._fresh_port() == 4001


def test_run_writes_claim_bounded_immutable_result_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    launch = {
        "base_url": "http://127.0.0.1:4444",
        "cleanup": {"descendants_absent": True},
    }
    fake_harness = SimpleNamespace(launches=[launch], max_active=2)
    monkeypatch.setattr(
        e2e,
        "_execute_gates",
        lambda supplied, work: ([{"name": "all", "status": "PASS"}], fake_harness),
    )
    result = e2e.run_qualification(config)
    assert result["status"] == "PASS"
    assert result["claim_boundary"] == e2e.CLAIM_BOUNDARY
    assert "performance" in result["not_claimed"]
    assert config.output.stat().st_mode & 0o777 == 0o444
    assert json.loads(config.output.read_text(encoding="utf-8"))["status"] == "PASS"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e2e.run_qualification(config)


def test_fatal_validation_failure_is_preserved_without_qualification_claim(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config = e2e.QualificationConfig(
        **{**config.__dict__, "patch_sha256": "not-a-hash"}
    )
    result = e2e.run_qualification(config)
    assert result["status"] == "FAIL"
    assert result["qualification_claim"] == "no qualification claim"
    assert "patch_sha256 must be" in result["fatal_error"]
    assert config.output.exists()


def test_store_snapshot_detects_content_change(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    record = store / "records.pcss"
    record.write_bytes(b"one")
    before = e2e._store_snapshot(store)
    record.write_bytes(b"two")
    after = e2e._store_snapshot(store)
    assert before["tree_sha256"] != after["tree_sha256"]
