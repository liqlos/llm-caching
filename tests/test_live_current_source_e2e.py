from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pcss import live_current_source_e2e as e2e


class _Headers:
    def get(self, name: str) -> str:
        return "text/plain" if name == "Content-Type" else ""


class _Response:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self._body = body
        self.status = status
        self.headers = _Headers()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _policy_document() -> dict[str, object]:
    return {
        "policy_id": "pcss.current-source-exact-anchor/v1",
        "max_draft_tokens": 2,
        "current_source_request_schema": e2e.CURRENT_SOURCE_SCHEMA,
        "source_is_request_scoped": True,
        "target_verification_required": True,
    }


def _config(tmp_path: Path) -> e2e.QualificationConfig:
    server = tmp_path / "fake-server"
    server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(_policy_document()), encoding="utf-8")
    return e2e.QualificationConfig(
        server=server,
        model=model,
        model_sha256=hashlib.sha256(b"model").hexdigest(),
        patch_sha256="a" * 64,
        policy=policy,
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
        output=tmp_path / "receipt.json",
        startup_timeout_seconds=1,
        request_timeout_seconds=1,
        context_size=512,
    )


def _metrics(policy_id: str, policy_sha256: str) -> bytes:
    lines = [f"llamacpp:{name} 0" for name in e2e._METRICS]
    lines.append(
        "llamacpp:pcss_policy_info{"
        f'policy_id="{policy_id}",policy_spec_sha256="{policy_sha256}"'
        "} 1"
    )
    return ("\n".join(lines) + "\n").encode()


def test_validate_config_binds_model_patch_and_policy_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    hashes = e2e._validate_config(config)
    assert hashes["model_sha256"] == config.model_sha256
    assert hashes["patch_sha256"] == "a" * 64
    assert hashes["policy_sha256"] == config.policy_sha256
    assert hashes["policy_id"] == "pcss.current-source-exact-anchor/v1"

    incorrect = e2e.QualificationConfig(
        **{**config.__dict__, "policy_sha256": "b" * 64}
    )
    with pytest.raises(e2e.QualificationError, match="policy SHA-256 mismatch"):
        e2e._validate_config(incorrect)


def test_harness_uses_isolated_loopback_pcss_source_server_with_k2(tmp_path: Path) -> None:
    config = _config(tmp_path)
    policy = {"policy_id": "pcss.current-source-exact-anchor/v1", "policy_sha256": "a" * 64}
    harness = e2e._Harness(config, tmp_path, policy)

    baseline = harness._command(4567, pcss=False, store=None)
    pcss = harness._command(4568, pcss=True, store=tmp_path / "store")

    assert baseline[baseline.index("--host") + 1] == "127.0.0.1"
    assert "--spec-type" not in baseline
    assert pcss[pcss.index("--host") + 1] == "127.0.0.1"
    assert pcss[pcss.index("--parallel") + 1] == "1"
    assert pcss[pcss.index("--pcss-max-draft-tokens") + 1] == "2"
    assert "--pcss-enable-current-source" in pcss
    assert "--pcss-read-only" in pcss
    assert "--pcss-disable-learning" in pcss
    assert pcss[pcss.index("--pcss-policy-id") + 1] == policy["policy_id"]
    assert pcss[pcss.index("--pcss-policy-spec-sha256") + 1] == policy["policy_sha256"]


def test_runtime_tokenize_and_completion_keep_source_only_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_http(
        base_url: str,
        method: str,
        path: str,
        timeout: float,
        *,
        body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        assert base_url == "http://127.0.0.1:1234"
        assert method == "POST"
        assert timeout == 2
        calls.append({"path": path, "body": body or {}})
        if path == "/tokenize":
            return {"status": 200, "json": {"tokens": [11, 12, 13, 14]}}
        return {
            "status": 200,
            "json": {"choices": [{"text": " 512 513", "finish_reason": "length"}], "usage": {"completion_tokens": 2}},
        }

    monkeypatch.setattr(e2e, "_http_json", fake_http)
    server = e2e._Server("http://127.0.0.1:1234", 2)
    tokens = server.tokenize("fixture source")
    source = e2e._source_request(tokens, "matching-fingerprint")
    response = server.completion("fixture prompt", current_source=source)

    assert tokens == (11, 12, 13, 14)
    assert response["output_text"] == " 512 513"
    assert calls[0]["path"] == "/tokenize"
    request = calls[1]["body"]
    assert isinstance(request, dict)
    assert request["pcss_current_source"] == source
    assert "pcss_current_source" not in response


def test_metric_snapshot_requires_exact_policy_attestation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = {"policy_id": "pcss.current-source-exact-anchor/v1", "policy_sha256": "a" * 64}

    def fake_urlopen(request: object, timeout: float) -> _Response:
        assert timeout == 2
        assert request.full_url == "http://127.0.0.1:1234/metrics"  # type: ignore[attr-defined]
        return _Response(_metrics(policy["policy_id"], policy["policy_sha256"]))

    monkeypatch.setattr(e2e, "urlopen", fake_urlopen)
    snapshot = e2e._metric_snapshot("http://127.0.0.1:1234", 2, policy=policy)
    assert snapshot["policy_attestation"]["passed"] is True
    assert snapshot["values"]["pcss_current_source_index_bytes"] == 0

    wrong = {**policy, "policy_sha256": "b" * 64}
    with pytest.raises(e2e.QualificationError, match="policy metric"):
        e2e._metric_snapshot("http://127.0.0.1:1234", 2, policy=wrong)


def test_redacted_launches_check_full_log_and_do_not_retain_it(tmp_path: Path) -> None:
    sensitive = b"sha256:" + b"a" * 64
    clean_launches = [
        {
            "label": "pcss-current-source",
            "pcss": True,
            "started_at": "now",
            "ended_at": "later",
            "cleanup": {"descendants_absent": True},
            "log_path": tmp_path / "ignored.log",
            "log_scan_bytes": b"server redacted request\n",
            "log_bytes": b"server redacted request\n",
            "log_total_bytes": 24,
        }
    ]
    redacted, absent = e2e._redacted_launches(clean_launches, (sensitive,))
    assert absent is True
    serialized = json.dumps(redacted)
    assert "captured_utf8" not in serialized
    assert sensitive.decode() not in serialized

    leaked = [
        {
            **clean_launches[0],
            "log_scan_bytes": b"prefix " + sensitive,
            "log_bytes": b"clean retained tail",
        }
    ]
    _, absent = e2e._redacted_launches(leaked, (sensitive,))
    assert absent is False


def test_run_writes_immutable_redacted_claim_bounded_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    hashes = {
        "model_sha256": config.model_sha256,
        "server_sha256": "b" * 64,
        "patch_sha256": config.patch_sha256,
        "policy_id": "pcss.current-source-exact-anchor/v1",
        "policy_sha256": config.policy_sha256,
    }
    monkeypatch.setattr(e2e, "_validate_config", lambda supplied: hashes)
    monkeypatch.setattr(
        e2e,
        "_execute",
        lambda supplied, supplied_hashes: {
            "source_scope": {"raw_source_retained": False},
            "process_invariants": {
                "isolated_arms": True,
                "loopback_only": True,
                "all_process_groups_cleaned": True,
                "server_logs_contain_no_request_source_material": True,
            },
        },
    )

    result = e2e.run_qualification(config)
    text = config.output.read_text(encoding="utf-8")
    assert result["status"] == "PASS"
    assert result["claim_boundary"] == e2e.CLAIM_BOUNDARY
    assert config.output.stat().st_mode & 0o777 == 0o444
    assert "visible_token_ids" not in text
    assert "source_id" not in text
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        e2e.run_qualification(config)
