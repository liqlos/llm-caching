"""Real-model lifecycle qualification for request-scoped PCSS current source.

This is deliberately a small correctness probe, not a performance benchmark.
It derives the static fixture's token IDs through the isolated local server at
runtime, keeps the source envelope in memory only, and writes a redacted,
immutable receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RESULT_SCHEMA = "pcss.live-current-source-lifecycle-qualification/v1"
CLAIM_BOUNDARY = (
    "local-real-model-current-source-correctness-only/"
    "not-performance/not-general-speedup/v1"
)
CURRENT_SOURCE_SCHEMA = "pcss.current-source/v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_POLICY_ID_RE = re.compile(
    r"^pcss\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)*/v[1-9][0-9]*$"
)
_POLICY_INFO_RE = re.compile(
    r'^llamacpp:pcss_policy_info\{policy_id="(?P<policy_id>[^"\\]+)",'
    r'policy_spec_sha256="(?P<policy_sha256>[0-9a-f]{64})"\} '
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)$"
)
_METRICS = (
    "pcss_current_source_lookups_total",
    "pcss_current_source_hits_total",
    "pcss_current_source_draft_tokens_total",
    "pcss_current_source_accepted_tokens_total",
    "pcss_current_source_mismatch_abstentions_total",
    "pcss_current_source_index_bytes",
    "pcss_current_source_peak_index_bytes",
)
_LOG_CAPTURE_LIMIT = 2 * 1024 * 1024

# This is a deterministic, non-user fixture.  The source text and its runtime
# token IDs exist only in process memory; the receipt retains neither.
_FIXTURE_SOURCE_START = 0
_FIXTURE_SOURCE_STOP = 1024
_FIXTURE_PROMPT_START = 384
_FIXTURE_PROMPT_STOP = 512


class QualificationError(RuntimeError):
    """Raised when a fixed current-source qualification gate is not met."""


@dataclass(frozen=True)
class QualificationConfig:
    server: Path
    model: Path
    model_sha256: str
    patch_sha256: str
    policy: Path
    policy_sha256: str
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _require_sha256(value: str, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise QualificationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_policy(config: QualificationConfig) -> dict[str, str]:
    if not config.policy.is_file():
        raise QualificationError(f"policy is not a file: {config.policy}")
    expected_sha256 = _require_sha256(config.policy_sha256, "policy_sha256")
    actual_sha256 = _sha256_file(config.policy)
    if actual_sha256 != expected_sha256:
        raise QualificationError(
            f"policy SHA-256 mismatch: expected {expected_sha256}, observed {actual_sha256}"
        )
    try:
        document = json.loads(config.policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot read policy JSON: {config.policy}") from error
    if not isinstance(document, dict):
        raise QualificationError("policy must be a JSON object")
    policy_id = document.get("policy_id")
    if not isinstance(policy_id, str) or not _POLICY_ID_RE.fullmatch(policy_id):
        raise QualificationError("policy.policy_id is not a canonical PCSS policy ID")
    if document.get("max_draft_tokens") != 2:
        raise QualificationError("policy.max_draft_tokens must be exactly 2")
    if document.get("current_source_request_schema") != CURRENT_SOURCE_SCHEMA:
        raise QualificationError("policy does not bind the current-source request schema")
    if document.get("source_is_request_scoped") is not True:
        raise QualificationError("policy must require request-scoped source material")
    if document.get("target_verification_required") is not True:
        raise QualificationError("policy must require target verification")
    return {"policy_id": policy_id, "policy_sha256": actual_sha256}


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
        raise QualificationError("context_size must be at least 256")
    expected_model_sha256 = _require_sha256(config.model_sha256, "model_sha256")
    patch_sha256 = _require_sha256(config.patch_sha256, "patch_sha256")
    actual_model_sha256 = _sha256_file(config.model)
    if actual_model_sha256 != expected_model_sha256:
        raise QualificationError(
            "model SHA-256 mismatch: "
            f"expected {expected_model_sha256}, observed {actual_model_sha256}"
        )
    policy = _read_policy(config)
    return {
        "model_sha256": actual_model_sha256,
        "server_sha256": _sha256_file(config.server),
        "patch_sha256": patch_sha256,
        **policy,
    }


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _number_fixture() -> tuple[str, str]:
    """Return a static long source and an instruction ending in its suffix."""

    source = " ".join(str(value) for value in range(_FIXTURE_SOURCE_START, _FIXTURE_SOURCE_STOP))
    prompt_tail = " ".join(
        str(value) for value in range(_FIXTURE_PROMPT_START, _FIXTURE_PROMPT_STOP)
    )
    prompt = (
        "Continue the exact space-separated increasing integer sequence. "
        "Return only the next four integers, separated by spaces:\n"
        f"{prompt_tail}"
    )
    return prompt, source


def _source_identifier(tokens: Sequence[int]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(list(tokens))).hexdigest()


def _assert_loopback(base_url: str) -> None:
    if not base_url.startswith("http://127.0.0.1:"):
        raise QualificationError("HTTP target must be numeric IPv4 loopback")


def _http_json(
    base_url: str,
    method: str,
    path: str,
    timeout: float,
    *,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _assert_loopback(base_url)
    payload = None if body is None else _canonical_json(body)
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except HTTPError as error:
        raw = error.read()
        status = error.code
    except (URLError, TimeoutError, OSError) as error:
        raise QualificationError(
            f"{method} {path} failed: {type(error).__name__}"
        ) from error
    try:
        document: object = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        document = None
    return {"status": status, "json": document}


def _metric_snapshot(
    base_url: str,
    timeout: float,
    *,
    policy: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _assert_loopback(base_url)
    request = Request(f"{base_url}/metrics", headers={"Accept": "text/plain"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise QualificationError(f"GET /metrics failed: {type(error).__name__}") from error
    if status != 200:
        raise QualificationError(f"GET /metrics returned HTTP {status}")
    text = raw.decode("utf-8", errors="replace")
    values: dict[str, float] = {}
    for name in _METRICS:
        prefix = f"llamacpp:{name} "
        for line in text.splitlines():
            if line.startswith(prefix):
                try:
                    values[name] = float(line[len(prefix) :])
                except ValueError as error:
                    raise QualificationError(f"malformed metric: {name}") from error
                break
        else:
            raise QualificationError(f"required metric is missing: {name}")
    attestation = {"required": policy is not None, "passed": policy is None}
    if policy is not None:
        observed: list[dict[str, str]] = []
        for line in text.splitlines():
            match = _POLICY_INFO_RE.fullmatch(line)
            if match is not None and float(match["value"]) == 1.0:
                observed.append(
                    {
                        "policy_id": match["policy_id"],
                        "policy_sha256": match["policy_sha256"],
                    }
                )
        expected = {
            "policy_id": policy["policy_id"],
            "policy_sha256": policy["policy_sha256"],
        }
        attestation.update(
            {
                "passed": observed == [expected],
                "observed_count": len(observed),
            }
        )
        if observed != [expected]:
            raise QualificationError("server policy metric does not attest the supplied policy")
    return {"values": values, "policy_attestation": attestation}


def _metric_delta(after: Mapping[str, Any], before: Mapping[str, Any], name: str) -> float:
    after_values = after.get("values")
    before_values = before.get("values")
    if not isinstance(after_values, dict) or not isinstance(before_values, dict):
        raise QualificationError("invalid metric snapshot")
    after_value = after_values.get(name)
    before_value = before_values.get(name)
    if not isinstance(after_value, (int, float)) or not isinstance(before_value, (int, float)):
        raise QualificationError(f"metric missing from snapshot: {name}")
    delta = float(after_value) - float(before_value)
    if delta < 0:
        raise QualificationError(f"metric decreased within one server lifecycle: {name}")
    return delta


class _Server:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def tokenize(self, content: str) -> tuple[int, ...]:
        response = _http_json(
            self.base_url,
            "POST",
            "/tokenize",
            self.timeout,
            body={"content": content, "add_special": False, "parse_special": True},
        )
        document = response["json"]
        tokens = document.get("tokens") if isinstance(document, dict) else None
        if response["status"] != 200 or not isinstance(tokens, list) or not tokens:
            raise QualificationError("local /tokenize did not return a non-empty token list")
        if len(tokens) > 16_384:
            raise QualificationError("fixture exceeds current-source token limit")
        normalized: list[int] = []
        for token in tokens:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise QualificationError("local /tokenize returned invalid token IDs")
            normalized.append(token)
        return tuple(normalized)

    def completion(
        self, prompt: str, *, current_source: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": 4,
            "temperature": 0,
            "top_k": 1,
            "seed": 7,
            "stream": False,
            "cache_prompt": False,
        }
        if current_source is not None:
            body["pcss_current_source"] = dict(current_source)
        response = _http_json(
            self.base_url, "POST", "/v1/completions", self.timeout, body=body
        )
        document = response["json"]
        choices = document.get("choices") if isinstance(document, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices else None
        if (
            response["status"] != 200
            or not isinstance(first_choice, dict)
            or not isinstance(first_choice.get("text"), str)
        ):
            raise QualificationError(
                f"completion did not return a normal target response (HTTP {response['status']})"
            )
        usage = document.get("usage") if isinstance(document, dict) else None
        output_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
            output_tokens = None
        finish_reason = first_choice.get("finish_reason")
        return {
            "status": response["status"],
            "output_text": first_choice["text"],
            "output_tokens": output_tokens,
            "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        }

    def metrics(self, *, policy: Mapping[str, str] | None = None) -> dict[str, Any]:
        return _metric_snapshot(self.base_url, self.timeout, policy=policy)


class _Harness:
    def __init__(
        self, config: QualificationConfig, work: Path, policy: Mapping[str, str]
    ) -> None:
        self.config = config
        self.work = work
        self.policy = dict(policy)
        self.active = 0
        self.maximum_active = 0
        self.used_ports: set[int] = set()
        self.launches: list[dict[str, Any]] = []

    def _fresh_port(self) -> int:
        for _ in range(100):
            port = _reserve_loopback_port()
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise QualificationError("could not reserve a fresh loopback port")

    def _command(self, port: int, *, pcss: bool, store: Path | None) -> list[str]:
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
            "1",
            "--ctx-size",
            str(self.config.context_size),
        ]
        if not pcss:
            return command
        if store is None:
            raise QualificationError("PCSS server requires an isolated store")
        fingerprint = f"model-sha256:{self.config.model_sha256}|api=completions|greedy=v1"
        command.extend(
            (
                "--spec-type",
                "pcss-cache",
                "--pcss-store",
                str(store),
                "--pcss-fingerprint",
                fingerprint,
                "--pcss-namespace",
                "current-source-e2e",
                "--pcss-max-disk-gib",
                "1",
                "--pcss-ram-mib",
                "64",
                "--pcss-read-only",
                "--pcss-disable-learning",
                "--pcss-enable-current-source",
                "--pcss-max-draft-tokens",
                "2",
                "--pcss-policy-id",
                self.policy["policy_id"],
                "--pcss-policy-spec-sha256",
                self.policy["policy_sha256"],
            )
        )
        return command

    @contextmanager
    def server(self, label: str, *, pcss: bool = False) -> Iterator[_Server]:
        if self.active:
            raise QualificationError("baseline and PCSS servers must remain isolated")
        port = self._fresh_port()
        base_url = f"http://127.0.0.1:{port}"
        store = self.work / "pcss-store" if pcss else None
        if store is not None:
            # A private empty directory lets the source-only probe run with a
            # read-only PCSS store: no durable records are created or reused.
            store.mkdir(mode=0o700)
            store.chmod(0o700)
        command = self._command(port, pcss=pcss, store=store)
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
        self.maximum_active = max(self.maximum_active, self.active)
        launch: dict[str, Any] = {
            "label": label,
            "base_url": base_url,
            "pid": process.pid,
            "started_at": _utc_now(),
            "pcss": pcss,
        }
        try:
            deadline = time.monotonic() + self.config.startup_timeout_seconds
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise QualificationError(f"{label} exited before readiness")
                try:
                    health = _http_json(base_url, "GET", "/health", 1.0)
                    if health["status"] == 200:
                        break
                except QualificationError:
                    pass
                time.sleep(0.1)
            else:
                raise QualificationError(f"{label} readiness timeout")
            yield _Server(base_url, self.config.request_timeout_seconds)
        finally:
            cleanup = _terminate_group(process)
            self.active -= 1
            log_handle.flush()
            log_handle.close()
            raw_log = log_path.read_bytes() if log_path.exists() else b""
            launch.update(
                {
                    "ended_at": _utc_now(),
                    "cleanup": cleanup,
                    "log_path": log_path,
                    "log_scan_bytes": raw_log,
                    "log_bytes": raw_log[-_LOG_CAPTURE_LIMIT:],
                    "log_total_bytes": len(raw_log),
                }
            )
            self.launches.append(launch)


def _terminate_group(
    process: subprocess.Popen[bytes], timeout: float = 10.0
) -> dict[str, Any]:
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
            return True
        except ProcessLookupError:
            return False

    action = "already_exited" if process.poll() is not None else "terminate_group"
    try:
        if group_exists():
            os.killpg(process_group, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        action = "kill_group"
        if group_exists():
            os.killpg(process_group, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)
    for _ in range(20):
        if not group_exists():
            break
        time.sleep(0.05)
    if group_exists():
        action = "kill_lingering_group"
        os.killpg(process_group, signal.SIGKILL)
        time.sleep(0.05)
    return {
        "action": action,
        "returncode": process.returncode,
        "process_group": process_group,
        "descendants_absent": not group_exists(),
    }


def _source_request(tokens: Sequence[int], fingerprint: str) -> dict[str, Any]:
    return {
        "schema_version": CURRENT_SOURCE_SCHEMA,
        "compatibility_fingerprint": fingerprint,
        "source_id": _source_identifier(tokens),
        "visible_token_ids": list(tokens),
    }


def _mismatch_source_request(fingerprint: str) -> dict[str, Any]:
    tokens = (-1, 2_147_483_647)
    return {
        "schema_version": CURRENT_SOURCE_SCHEMA,
        "compatibility_fingerprint": f"{fingerprint}|mismatch-control=v1",
        "source_id": _source_identifier(tokens),
        "visible_token_ids": list(tokens),
    }


def _require_equal(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], label: str
) -> None:
    for field in ("status", "output_text", "output_tokens", "finish_reason"):
        if baseline.get(field) != candidate.get(field):
            raise QualificationError(f"{label}: target output mismatch at {field}")


def _redacted_launches(
    launches: Sequence[Mapping[str, Any]], sensitive: Sequence[bytes]
) -> tuple[list[dict[str, Any]], bool]:
    redacted: list[dict[str, Any]] = []
    absent = True
    for launch in launches:
        log_bytes = launch.get("log_bytes")
        log_scan_bytes = launch.get("log_scan_bytes")
        if not isinstance(log_bytes, bytes) or not isinstance(log_scan_bytes, bytes):
            raise QualificationError("captured server log is unavailable")
        if any(marker and marker in log_scan_bytes for marker in sensitive):
            absent = False
        cleanup = launch.get("cleanup")
        if not isinstance(cleanup, dict):
            raise QualificationError("server cleanup receipt is unavailable")
        redacted.append(
            {
                "label": launch["label"],
                "pcss": launch["pcss"],
                "started_at": launch["started_at"],
                "ended_at": launch["ended_at"],
                "cleanup": cleanup,
                "log": {
                    "bytes": launch["log_total_bytes"],
                    "sha256": hashlib.sha256(log_bytes).hexdigest(),
                    "raw_log_retained": False,
                },
            }
        )
    return redacted, absent


def _write_immutable(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write while creating immutable receipt")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o444)


def _execute(config: QualificationConfig, hashes: Mapping[str, str]) -> dict[str, Any]:
    policy = {
        "policy_id": hashes["policy_id"],
        "policy_sha256": hashes["policy_sha256"],
    }
    prompt, source_text = _number_fixture()
    with tempfile.TemporaryDirectory(
        prefix="pcss-current-source-e2e-", dir=config.output.parent
    ) as raw_work:
        work = Path(raw_work)
        harness = _Harness(config, work, policy)
        with harness.server("baseline") as baseline:
            baseline_response = baseline.completion(prompt)
        with harness.server("pcss-current-source", pcss=True) as pcss:
            startup_metrics = pcss.metrics(policy=policy)
            source_tokens = pcss.tokenize(source_text)
            fingerprint = f"model-sha256:{config.model_sha256}|api=completions|greedy=v1"
            matching_source = _source_request(source_tokens, fingerprint)
            before_matching = pcss.metrics()
            matching_response = pcss.completion(prompt, current_source=matching_source)
            after_matching = pcss.metrics()
            mismatch_response = pcss.completion(
                prompt, current_source=_mismatch_source_request(fingerprint)
            )
            after_mismatch = pcss.metrics()

        _require_equal(baseline_response, matching_response, "matching current source")
        _require_equal(baseline_response, mismatch_response, "fingerprint mismatch control")
        lookup_delta = _metric_delta(
            after_matching, before_matching, "pcss_current_source_lookups_total"
        )
        hit_delta = _metric_delta(
            after_matching, before_matching, "pcss_current_source_hits_total"
        )
        accepted_delta = _metric_delta(
            after_matching, before_matching, "pcss_current_source_accepted_tokens_total"
        )
        mismatch_delta = _metric_delta(
            after_mismatch,
            after_matching,
            "pcss_current_source_mismatch_abstentions_total",
        )
        peak_index_bytes = after_matching["values"]["pcss_current_source_peak_index_bytes"]
        active_index_bytes = after_mismatch["values"]["pcss_current_source_index_bytes"]
        if lookup_delta <= 0 or hit_delta <= 0 or accepted_delta <= 0:
            raise QualificationError("matching source did not produce a verified accepted draft")
        if peak_index_bytes <= 0:
            raise QualificationError("matching source did not allocate a request-scoped index")
        if active_index_bytes != 0:
            raise QualificationError("request-scoped source index remained active after end")
        if mismatch_delta <= 0:
            raise QualificationError("mismatch control did not record an abstention")

        source_id = matching_source["source_id"]
        source_tokens_compact = _canonical_json(source_tokens)
        mismatch_id = _mismatch_source_request(fingerprint)["source_id"]
        launches, logs_absent = _redacted_launches(
            harness.launches,
            (
                source_id.encode(),
                mismatch_id.encode(),
                b"visible_token_ids",
                source_tokens_compact,
            ),
        )
        if not logs_absent:
            raise QualificationError("captured server log contained request-scoped source material")
        return {
            "gates": {
                "baseline_matching_source_output_equality": True,
                "matching_source_lookup_delta": lookup_delta,
                "matching_source_hit_delta": hit_delta,
                "matching_source_accepted_tokens_delta": accepted_delta,
                "matching_source_peak_index_bytes": peak_index_bytes,
                "active_source_index_bytes_after_end": active_index_bytes,
                "mismatch_control_output_equality": True,
                "mismatch_control_abstention_delta": mismatch_delta,
            },
            "source_scope": {
                "fixture": "static-number-sequence/v1",
                "tokenization": "local-loopback-runtime",
                "request_scoped": True,
                "token_count": len(source_tokens),
                "raw_source_retained": False,
            },
            "startup_metrics": startup_metrics,
            "responses": {
                "baseline": baseline_response,
                "matching_source": matching_response,
                "mismatch_control": mismatch_response,
            },
            "metrics": {
                "before_matching": before_matching,
                "after_matching": after_matching,
                "after_mismatch": after_mismatch,
            },
            "server_launches": launches,
            "process_invariants": {
                "maximum_live_servers": harness.maximum_active,
                "isolated_arms": harness.maximum_active == 1,
                "loopback_only": True,
                "all_process_groups_cleaned": all(
                    bool(launch["cleanup"].get("descendants_absent"))
                    for launch in launches
                ),
                "server_logs_contain_no_request_source_material": logs_absent,
            },
        }


def run_qualification(config: QualificationConfig) -> dict[str, Any]:
    """Run one fixed source lifecycle probe and atomically create its receipt."""

    if config.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {config.output}")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "started_at": _utc_now(),
        "inputs": {
            "server": str(config.server),
            "model": str(config.model),
            "expected_model_sha256": config.model_sha256,
            "declared_patch_sha256": config.patch_sha256,
            "policy": str(config.policy),
            "expected_policy_sha256": config.policy_sha256,
            "context_size": config.context_size,
            "max_draft_tokens": 2,
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
        hashes = _validate_config(config)
        result["hashes"] = hashes
        result.update(_execute(config, hashes))
        invariants = result["process_invariants"]
        result["status"] = "PASS" if (
            invariants["isolated_arms"]
            and invariants["loopback_only"]
            and invariants["all_process_groups_cleaned"]
            and invariants["server_logs_contain_no_request_source_material"]
        ) else "FAIL"
    except Exception as error:
        result["status"] = "FAIL"
        result["fatal_error"] = f"{type(error).__name__}: {error}"
    result["ended_at"] = _utc_now()
    result["qualification_claim"] = (
        "fixed local current-source lifecycle gates passed"
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
    parser.add_argument("--patch-sha256", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
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
        policy=args.policy.resolve(),
        policy_sha256=args.policy_sha256,
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
    print(json.dumps({"status": result["status"], "output": str(config.output)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
