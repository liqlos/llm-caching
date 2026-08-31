"""Optional, claim-bounded real-model lifecycle qualification for PCSS.

This driver is intentionally separate from performance benchmarking.  It runs a
fixed set of correctness and lifecycle gates against a local ``llama-server``,
preserves raw responses/metrics/logs, and emits one immutable JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


RESULT_SCHEMA = "pcss.live-server-lifecycle-qualification/v1"
CLAIM_BOUNDARY = (
    "local-real-model-lifecycle-correctness-only/not-performance/not-general-speedup/v1"
)
_SHA256_LENGTH = 64
_LOG_CAPTURE_LIMIT = 2 * 1024 * 1024


class QualificationError(RuntimeError):
    """Raised for a violated driver or qualification contract."""


@dataclass(frozen=True)
class QualificationConfig:
    server: Path
    model: Path
    model_sha256: str
    patch_sha256: str
    output: Path
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 120.0
    context_size: int = 1024


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str, label: str) -> str:
    if len(value) != _SHA256_LENGTH or any(c not in "0123456789abcdef" for c in value):
        raise QualificationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_config(config: QualificationConfig) -> dict[str, str]:
    if config.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {config.output}")
    if not config.server.is_file() or not os.access(config.server, os.X_OK):
        raise QualificationError(f"server is not an executable file: {config.server}")
    if not config.model.is_file():
        raise QualificationError(f"model is not a file: {config.model}")
    if config.startup_timeout_seconds <= 0 or config.request_timeout_seconds <= 0:
        raise QualificationError("timeouts must be positive")
    if config.context_size < 256:
        raise QualificationError("context size must be at least 256")
    expected_model = _validate_sha256(config.model_sha256, "model_sha256")
    patch_sha = _validate_sha256(config.patch_sha256, "patch_sha256")
    actual_model = _sha256_file(config.model)
    if actual_model != expected_model:
        raise QualificationError(
            f"model SHA-256 mismatch: expected {expected_model}, observed {actual_model}"
        )
    return {
        "model_sha256": actual_model,
        "server_sha256": _sha256_file(config.server),
        "patch_sha256": patch_sha,
    }


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _body(prompt: str, *, max_tokens: int = 32, stream: bool = False) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": 7,
        "stream": stream,
        "cache_prompt": False,
    }


def _response_text(document: Mapping[str, Any]) -> str:
    choices = document.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise QualificationError("completion response has no first choice")
    text = choices[0].get("text")
    if not isinstance(text, str):
        raise QualificationError("completion response first choice has no text")
    return text


def _http_json(
    base_url: str,
    method: str,
    path: str,
    timeout: float,
    *,
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not base_url.startswith("http://127.0.0.1:"):
        raise QualificationError("HTTP target must be numeric IPv4 loopback")
    payload = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request_headers = {"Accept": "application/json"}
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    request = Request(
        f"{base_url}{path}", data=payload, headers=request_headers, method=method
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type")
    except HTTPError as error:
        raw = error.read()
        status = error.code
        content_type = error.headers.get("Content-Type")
    except (URLError, TimeoutError, OSError) as error:
        raise QualificationError(
            f"{method} {path} failed: {type(error).__name__}: {error}"
        ) from error
    decoded = raw.decode("utf-8", errors="replace")
    parsed: Any = None
    if decoded:
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            parsed = None
    return {
        "method": method,
        "path": path,
        "status": status,
        "content_type": content_type,
        "body_utf8": decoded,
        "json": parsed,
    }


def _fetch_metrics(base_url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        f"{base_url}/metrics", headers={"Accept": "text/plain"}, method="GET"
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type")
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise QualificationError(
            f"GET /metrics failed: {type(error).__name__}: {error}"
        ) from error
    body = raw.decode("utf-8", errors="replace")
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line.startswith("llamacpp:") or " " not in line:
            continue
        name, raw_value = line.split(None, 1)
        try:
            values[name.removeprefix("llamacpp:")] = float(raw_value)
        except ValueError:
            continue
    return {
        "status": status,
        "content_type": content_type,
        "body_utf8": body,
        "values": values,
    }


def _metric(receipt: Mapping[str, Any], name: str) -> float:
    values = receipt.get("values")
    value = values.get(name) if isinstance(values, dict) else None
    if not isinstance(value, (int, float)):
        raise QualificationError(f"required metric is missing: {name}")
    return float(value)


def _store_snapshot(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    if path.exists():
        for candidate in sorted(path.rglob("*")):
            relative = str(candidate.relative_to(path))
            if candidate.is_file():
                entries.append(
                    {
                        "path": relative,
                        "size": candidate.stat().st_size,
                        "sha256": _sha256_file(candidate),
                    }
                )
            elif candidate.is_dir():
                entries.append({"path": relative, "directory": True})
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return {
        "path": str(path),
        "entries": entries,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "bytes": sum(int(row.get("size", 0)) for row in entries),
    }


class _Server:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def completion(self, prompt: str, *, max_tokens: int = 32) -> dict[str, Any]:
        receipt = _http_json(
            self.base_url,
            "POST",
            "/v1/completions",
            self.timeout,
            body=_body(prompt, max_tokens=max_tokens),
        )
        if receipt["status"] != 200 or not isinstance(receipt["json"], dict):
            raise QualificationError(f"completion failed: HTTP {receipt['status']}")
        receipt["output_text"] = _response_text(receipt["json"])
        return receipt

    def metrics(self) -> dict[str, Any]:
        return _fetch_metrics(self.base_url, self.timeout)

    def cancel_completion(self, prompt: str) -> dict[str, Any]:
        conversation_id = f"pcss-e2e-{uuid.uuid4()}"
        body = _body(prompt, max_tokens=512, stream=True)
        payload = json.dumps(body, separators=(",", ":")).encode()
        request = Request(
            f"{self.base_url}/v1/completions",
            data=payload,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "X-Conversation-Id": conversation_id,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                first_line = response.readline()
                response_status = response.status
                stop = _http_json(
                    self.base_url,
                    "DELETE",
                    f"/v1/stream?{urlencode({'conv_id': conversation_id})}",
                    self.timeout,
                )
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise QualificationError(
                f"cancellation request failed: {type(error).__name__}: {error}"
            ) from error
        if response_status != 200 or stop["status"] != 204:
            raise QualificationError(
                f"explicit stream cancellation failed: stream={response_status}, stop={stop['status']}"
            )
        return {
            "conversation_id": conversation_id,
            "stream_status": response_status,
            "first_line_utf8": first_line.decode("utf-8", errors="replace"),
            "stop": stop,
        }


class _Harness:
    def __init__(self, config: QualificationConfig, work: Path) -> None:
        self.config = config
        self.work = work
        self.active = 0
        self.max_active = 0
        self.launches: list[dict[str, Any]] = []
        self.used_ports: set[int] = set()

    def _fresh_port(self) -> int:
        for _ in range(100):
            port = _reserve_loopback_port()
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise QualificationError("could not reserve a fresh loopback port")

    def _command(
        self,
        port: int,
        *,
        store: Path | None,
        parallel: int,
        read_only: bool,
        disable_learning: bool,
    ) -> list[str]:
        command = [
            str(self.config.server),
            "--model",
            str(self.config.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--metrics",
            "--no-warmup",
            "--parallel",
            str(parallel),
            "--ctx-size",
            str(self.config.context_size),
        ]
        if store is not None:
            command.extend(
                [
                    "--spec-type",
                    "pcss-cache",
                    "--pcss-store",
                    str(store),
                    "--pcss-fingerprint",
                    f"model-sha256:{self.config.model_sha256}|api=completions|greedy=v1",
                    "--pcss-namespace",
                    "local-e2e",
                    "--pcss-max-disk-gib",
                    "1",
                    "--pcss-ram-mib",
                    "64",
                ]
            )
            if read_only:
                command.append("--pcss-read-only")
            if disable_learning:
                command.append("--pcss-disable-learning")
        return command

    @contextmanager
    def server(
        self,
        label: str,
        *,
        store: Path | None = None,
        parallel: int = 1,
        read_only: bool = False,
        disable_learning: bool = False,
    ) -> Iterator[_Server]:
        if self.active >= 2:
            raise QualificationError(
                "driver invariant violated: more than two live servers"
            )
        if parallel not in {1, 2}:
            raise QualificationError("server parallelism must be one or two")
        port = self._fresh_port()
        base_url = f"http://127.0.0.1:{port}"
        command = self._command(
            port,
            store=store,
            parallel=parallel,
            read_only=read_only,
            disable_learning=disable_learning,
        )
        log_path = self.work / f"{len(self.launches):02d}-{label}.log"
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        launch: dict[str, Any] = {
            "label": label,
            "command": command,
            "base_url": base_url,
            "pid": process.pid,
            "started_at": _utc_now(),
        }
        try:
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            attempts: list[dict[str, Any]] = []
            while time.monotonic() < deadline:
                returncode = process.poll()
                if returncode is not None:
                    raise QualificationError(
                        f"{label} exited before readiness: {returncode}"
                    )
                try:
                    receipt = _http_json(base_url, "GET", "/health", 1.0)
                    attempts.append({"status": receipt["status"]})
                    if receipt["status"] == 200:
                        break
                except QualificationError as error:
                    attempts.append({"error": str(error)})
                time.sleep(0.1)
            else:
                raise QualificationError(f"{label} readiness timeout")
            launch["readiness"] = attempts
            yield _Server(base_url, self.config.request_timeout_seconds)
        finally:
            cleanup = _terminate_group(process)
            self.active -= 1
            log_handle.flush()
            log_handle.close()
            raw_log = log_path.read_bytes() if log_path.exists() else b""
            captured = raw_log[-_LOG_CAPTURE_LIMIT:]
            launch.update(
                {
                    "ended_at": _utc_now(),
                    "cleanup": cleanup,
                    "log": {
                        "bytes": len(raw_log),
                        "sha256": hashlib.sha256(raw_log).hexdigest(),
                        "truncated_to_tail": len(raw_log) > len(captured),
                        "captured_utf8": captured.decode("utf-8", errors="replace"),
                    },
                }
            )
            self.launches.append(launch)


def _terminate_group(
    process: subprocess.Popen[Any], timeout: float = 10.0
) -> dict[str, Any]:
    pid = process.pid

    def exists() -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    action = "already_exited" if process.poll() is not None else "terminate_group"
    try:
        if exists():
            os.killpg(pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        action = "kill_group"
        if exists():
            os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)
    for _ in range(20):
        if not exists():
            break
        time.sleep(0.05)
    if exists():
        action = "kill_lingering_group"
        os.killpg(pid, signal.SIGKILL)
        for _ in range(20):
            if not exists():
                break
            time.sleep(0.05)
    return {
        "action": action,
        "returncode": process.returncode,
        "process_group": pid,
        "descendants_absent": not exists(),
    }


def _require_equal(
    left: Mapping[str, Any], right: Mapping[str, Any], label: str
) -> None:
    if left.get("output_text") != right.get("output_text"):
        raise QualificationError(f"{label}: greedy output mismatch")


def _gate_output_equality(harness: _Harness, root: Path) -> dict[str, Any]:
    prompt = "PCSS equality gate. Continue deterministically: one two three"
    with (
        harness.server("equality-baseline") as baseline,
        harness.server("equality-pcss", store=root / "equality-store") as pcss,
    ):
        baseline_response = baseline.completion(prompt)
        pcss_response = pcss.completion(prompt)
    _require_equal(baseline_response, pcss_response, "PCSS on/off")
    return {"passed": True, "baseline": baseline_response, "pcss": pcss_response}


def _gate_persistence(harness: _Harness, root: Path) -> dict[str, Any]:
    store = root / "persistent-store"
    prompt = "PCSS persistence gate. Continue this stable sequence: alpha beta gamma"
    with harness.server("learn", store=store) as server:
        before = server.metrics()
        response = server.completion(prompt, max_tokens=48)
        after = server.metrics()
    learned_delta = _metric(after, "pcss_records_learned_total") - _metric(
        before, "pcss_records_learned_total"
    )
    if learned_delta <= 0:
        raise QualificationError("successful completion did not learn a PCSS record")
    with harness.server("persistent-restart", store=store) as restarted:
        restart_before = restarted.metrics()
        replay = restarted.completion(prompt, max_tokens=48)
        restart_after = restarted.metrics()
    _require_equal(response, replay, "persistent restart")
    hit_delta = _metric(restart_after, "pcss_persistent_hits_total") - _metric(
        restart_before, "pcss_persistent_hits_total"
    )
    if hit_delta <= 0:
        raise QualificationError("restart replay did not record a persistent hit")
    return {
        "passed": True,
        "store": _store_snapshot(store),
        "learned_delta": learned_delta,
        "persistent_hit_delta": hit_delta,
        "learn_response": response,
        "replay_response": replay,
        "metrics": {
            "learn_before": before,
            "learn_after": after,
            "restart_before": restart_before,
            "restart_after": restart_after,
        },
    }


def _gate_cancellation(harness: _Harness, root: Path) -> dict[str, Any]:
    store = root / "cancel-store"
    with harness.server("cancel", store=store) as server:
        before = server.metrics()
        cancellation = server.cancel_completion(
            "PCSS cancellation gate. Produce a very long numbered list without stopping."
        )
        deadline = time.monotonic() + min(10.0, harness.config.request_timeout_seconds)
        after = server.metrics()
        while (
            _metric(after, "pcss_abort_learning_suppressions_total")
            <= _metric(before, "pcss_abort_learning_suppressions_total")
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
            after = server.metrics()
        next_response = server.completion(
            "After cancellation, answer with a short continuation: ready"
        )
        final_metrics = server.metrics()
    abort_delta = _metric(after, "pcss_abort_learning_suppressions_total") - _metric(
        before, "pcss_abort_learning_suppressions_total"
    )
    learned_delta_at_abort = _metric(after, "pcss_records_learned_total") - _metric(
        before, "pcss_records_learned_total"
    )
    if abort_delta <= 0 or learned_delta_at_abort != 0:
        raise QualificationError("cancelled request was not excluded from learning")
    return {
        "passed": True,
        "cancellation": cancellation,
        "next_response": next_response,
        "abort_suppression_delta": abort_delta,
        "learned_delta_at_abort": learned_delta_at_abort,
        "metrics": {
            "before": before,
            "after_abort": after,
            "after_next": final_metrics,
        },
    }


def _gate_same_slot(harness: _Harness, root: Path) -> dict[str, Any]:
    prompt_a = "Long A context: " + "red green blue yellow " * 80
    prompt_b = "Short B context. Continue: north south"
    with harness.server(
        "same-slot", store=root / "same-slot-store", parallel=1
    ) as reused:
        long_a = reused.completion(prompt_a, max_tokens=24)
        reused_b = reused.completion(prompt_b, max_tokens=32)
    with harness.server("fresh-b", parallel=1) as fresh:
        baseline_b = fresh.completion(prompt_b, max_tokens=32)
    _require_equal(reused_b, baseline_b, "long A then short B same-slot")
    return {
        "passed": True,
        "long_a": long_a,
        "reused_b": reused_b,
        "fresh_baseline_b": baseline_b,
    }


def _gate_concurrent(harness: _Harness, root: Path) -> dict[str, Any]:
    prompts = [
        "Concurrent slot zero. Continue: mercury venus earth",
        "Concurrent slot one. Continue: copper silver gold",
    ]
    with (
        harness.server("concurrent-baseline", parallel=2) as baseline,
        harness.server(
            "concurrent-pcss", store=root / "concurrent-store", parallel=2
        ) as pcss,
    ):
        baseline_responses = [baseline.completion(prompt) for prompt in prompts]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pcss.completion, prompt) for prompt in prompts]
            pcss_responses = [future.result() for future in futures]
    for index, (expected, observed) in enumerate(
        zip(baseline_responses, pcss_responses, strict=True)
    ):
        _require_equal(expected, observed, f"concurrent slot {index}")
    return {"passed": True, "baseline": baseline_responses, "pcss": pcss_responses}


def _gate_bounded_context(harness: _Harness, root: Path) -> dict[str, Any]:
    prompt = "Bounded context gate. " + "zero one two three four five six seven " * 45
    with (
        harness.server("bounded-baseline") as baseline,
        harness.server("bounded-pcss", store=root / "bounded-store") as pcss,
    ):
        expected = baseline.completion(prompt, max_tokens=24)
        observed = pcss.completion(prompt, max_tokens=24)
    _require_equal(expected, observed, "bounded context")
    return {
        "passed": True,
        "configured_context_size": harness.config.context_size,
        "baseline": expected,
        "pcss": observed,
    }


def _gate_invalid_store(harness: _Harness, root: Path) -> dict[str, Any]:
    invalid = root / "invalid-store-is-a-file"
    invalid.write_text("not a directory\n", encoding="utf-8")
    with harness.server("invalid-store", store=invalid) as server:
        before = server.metrics()
        response = server.completion(
            "Invalid store fallback gate. Continue: still operational"
        )
        after = server.metrics()
    fallback_before = _metric(before, "pcss_fallback_disable_events_total")
    fallback_after = _metric(after, "pcss_fallback_disable_events_total")
    if fallback_before < 1 or fallback_after < fallback_before:
        raise QualificationError(
            "invalid store did not produce a fallback-disable event"
        )
    return {
        "passed": True,
        "response": response,
        "fallback_disable_at_readiness": fallback_before,
        "fallback_disable_after_request": fallback_after,
        "metrics": {"before": before, "after": after},
    }


def _configured_suppression(
    harness: _Harness,
    label: str,
    store: Path,
    *,
    read_only: bool,
    disable_learning: bool,
) -> dict[str, Any]:
    with harness.server(
        label,
        store=store,
        read_only=read_only,
        disable_learning=disable_learning,
    ) as server:
        before_metrics = server.metrics()
        before_store = _store_snapshot(store)
        response = server.completion(
            f"{label} gate. Continue: writes must remain suppressed"
        )
        after_store = _store_snapshot(store)
        after_metrics = server.metrics()
    suppression = _metric(
        after_metrics, "pcss_config_learning_suppressions_total"
    ) - _metric(before_metrics, "pcss_config_learning_suppressions_total")
    writes = _metric(after_metrics, "pcss_store_bytes_written_total") - _metric(
        before_metrics, "pcss_store_bytes_written_total"
    )
    if (
        suppression <= 0
        or writes != 0
        or before_store["tree_sha256"] != after_store["tree_sha256"]
    ):
        raise QualificationError(
            f"{label} did not suppress all request-time store writes"
        )
    return {
        "passed": True,
        "response": response,
        "config_suppression_delta": suppression,
        "store_bytes_written_delta": writes,
        "store_before": before_store,
        "store_after": after_store,
        "metrics": {"before": before_metrics, "after": after_metrics},
    }


def _gate_learning_controls(harness: _Harness, root: Path) -> dict[str, Any]:
    seed_store = root / "read-only-store"
    with harness.server("read-only-seed", store=seed_store) as server:
        server.completion(
            "Seed the read-only store. Continue: immutable replay material"
        )
    read_only = _configured_suppression(
        harness, "read-only", seed_store, read_only=True, disable_learning=False
    )
    disable_store = root / "disable-learning-store"
    disable_learning = _configured_suppression(
        harness,
        "disable-learning",
        disable_store,
        read_only=False,
        disable_learning=True,
    )
    return {
        "passed": True,
        "read_only": read_only,
        "disable_learning": disable_learning,
    }


_GATES: tuple[tuple[str, Callable[[_Harness, Path], dict[str, Any]]], ...] = (
    ("pcss_on_off_greedy_output_equality", _gate_output_equality),
    ("successful_end_learning_and_persistent_restart_hit", _gate_persistence),
    ("cancellation_suppresses_learning_and_server_recovers", _gate_cancellation),
    ("long_a_short_b_same_slot_matches_fresh_baseline", _gate_same_slot),
    ("two_concurrent_slots_match_baseline", _gate_concurrent),
    ("bounded_context_matches_baseline", _gate_bounded_context),
    ("invalid_store_falls_back_without_server_failure", _gate_invalid_store),
    ("read_only_and_disable_learning_suppress_writes", _gate_learning_controls),
)


def _execute_gates(
    config: QualificationConfig, work: Path
) -> tuple[list[dict[str, Any]], _Harness]:
    harness = _Harness(config, work)
    gates: list[dict[str, Any]] = []
    for name, gate in _GATES:
        started = _utc_now()
        gate_root = work / name
        gate_root.mkdir(parents=True, exist_ok=False)
        try:
            evidence = gate(harness, gate_root)
            gates.append(
                {
                    "name": name,
                    "status": "PASS",
                    "started_at": started,
                    "ended_at": _utc_now(),
                    "evidence": evidence,
                }
            )
        except (
            Exception
        ) as error:  # preserve every independent gate failure and continue
            gates.append(
                {
                    "name": name,
                    "status": "FAIL",
                    "started_at": started,
                    "ended_at": _utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return gates, harness


def _write_immutable(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while creating immutable result")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def run_qualification(config: QualificationConfig) -> dict[str, Any]:
    """Run all gates and atomically create the immutable result path."""
    if config.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {config.output}")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "started_at": started,
        "inputs": {
            "server": str(config.server),
            "model": str(config.model),
            "expected_model_sha256": config.model_sha256,
            "declared_patch_sha256": config.patch_sha256,
            "output": str(config.output),
            "startup_timeout_seconds": config.startup_timeout_seconds,
            "request_timeout_seconds": config.request_timeout_seconds,
            "context_size": config.context_size,
        },
        "not_claimed": [
            "performance",
            "speedup",
            "throughput",
            "latency improvement",
            "representative workload quality",
            "production qualification",
        ],
    }
    try:
        result["hashes"] = _validate_config(config)
        with tempfile.TemporaryDirectory(
            prefix="pcss-live-e2e-", dir=config.output.parent
        ) as raw_work:
            work = Path(raw_work)
            gates, harness = _execute_gates(config, work)
            result["gates"] = gates
            result["server_launches"] = harness.launches
            result["process_invariants"] = {
                "maximum_live_servers": harness.max_active,
                "limit": 2,
                "loopback_only": all(
                    launch["base_url"].startswith("http://127.0.0.1:")
                    for launch in harness.launches
                ),
                "all_process_groups_cleaned": all(
                    launch["cleanup"]["descendants_absent"]
                    for launch in harness.launches
                ),
            }
        passed = all(gate["status"] == "PASS" for gate in result["gates"])
        invariants = result["process_invariants"]
        passed = passed and invariants["maximum_live_servers"] <= 2
        passed = (
            passed
            and invariants["loopback_only"]
            and invariants["all_process_groups_cleaned"]
        )
        result["status"] = "PASS" if passed else "FAIL"
    except Exception as error:
        result["status"] = "FAIL"
        result["fatal_error"] = f"{type(error).__name__}: {error}"
    result["ended_at"] = _utc_now()
    result["qualification_claim"] = (
        "all fixed local lifecycle gates passed"
        if result["status"] == "PASS"
        else "no qualification claim"
    )
    _write_immutable(config.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument(
        "--patch-sha256", "--patch-sha", dest="patch_sha256", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--context-size", type=int, default=1024)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = QualificationConfig(
        server=args.server.resolve(),
        model=args.model.resolve(),
        model_sha256=args.model_sha256,
        patch_sha256=args.patch_sha256,
        output=args.output.resolve(),
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        context_size=args.context_size,
    )
    try:
        result = run_qualification(config)
    except FileExistsError as error:
        print(str(error), file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {"status": result["status"], "output": str(config.output)}, sort_keys=True
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
