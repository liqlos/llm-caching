from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.canonical import sha256
from pcss.llamacpp_adapter import LlamaCppBridgeAdapter
from pcss.verifier import VerifierCase


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime() -> dict[str, object]:
    return {
        "n_ctx": 512,
        "n_batch": 128,
        "n_gpu_layers": 99,
        "flash_attn": True,
        "seed": 7,
        "n_threads": 2,
        "n_threads_batch": 2,
        "chat_template": "chatml",
        "thinking_mode": "disabled",
    }


def _runtime_fingerprint(library: Path, model: Path) -> str:
    return sha256(
        {
            "schema_version": "pcss.llamacpp-runtime-config/v1",
            "bridge_protocol": "pcss.llama-bridge/v1",
            "model_artifact": str(model.resolve()),
            "llama_cpp_library": str(library.resolve()),
            "llama_cpp_revision": "b5401",
            "runtime": _runtime(),
        }
    )


def _capabilities(
    library: Path,
    model: Path,
    *,
    runtime_fingerprint: str,
    comparison: str = "byte_exact",
    state_round_trip: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": "pcss.llama-bridge-capabilities/v1",
        "bridge_protocol": "pcss.llama-bridge/v1",
        "operation": "verify_case",
        "backend": {
            "runtime": "llama.cpp",
            "llama_cpp_revision": "b5401",
            "library_sha256": _sha256(library),
            "model_artifact_sha256": _sha256(model),
        },
        "full_state": {
            "snapshot_api": "llama_state_seq_get_data",
            "restore_api": "llama_state_seq_set_data",
            "comparison": comparison,
            "covered_components": ["llama_state", "logits", "token_history", "positions"],
        },
        "probe": {
            "model_loaded": True,
            "state_round_trip": state_round_trip,
            "comparison": "byte_exact",
            "covered_components": ["llama_state", "logits", "token_history", "positions"],
            "before_digest": "b" * 64,
            "after_digest": "b" * 64,
        },
        "runtime_configuration_fingerprint": runtime_fingerprint,
        "generation_mode": "greedy_argmax",
    }


def _outcome_response(*, runtime_fingerprint: str, malformed: bool = False) -> dict[str, object]:
    if malformed:
        return {"schema_version": "pcss.llama-bridge-response/v1"}
    digest = "a" * 64
    return {
        "schema_version": "pcss.llama-bridge-response/v1",
        "bridge_protocol": "pcss.llama-bridge/v1",
        "operation": "verify_case",
        "outcome": {
            "accepted_draft_tokens": 1,
            "correction_token": 8,
            "bonus_token": None,
            "committed_tokens": [7, 8],
            "mutable_state_digest": digest,
        },
        "state_comparison": {
            "comparison": "byte_exact",
            "reference_digest": digest,
            "candidate_digest": digest,
            "equal": True,
        },
        "runtime_configuration_fingerprint": runtime_fingerprint,
        "generation_mode": "greedy_argmax",
    }


def _write_bridge(
    path: Path,
    *,
    capabilities: dict[str, object],
    verify_response: dict[str, object],
    request_path: Path | None = None,
    capability_count_path: Path | None = None,
) -> None:
    request_write = ""
    if request_path is not None:
        request_write = f"Path({str(request_path)!r}).write_text(request, encoding='utf-8')"
    capability_count = ""
    if capability_count_path is not None:
        capability_count = (
            f"counter = Path({str(capability_count_path)!r}); "
            "counter.write_text(str((int(counter.read_text()) if counter.exists() else 0) + 1), encoding='utf-8')"
        )
    script = f"""#!{sys.executable}
import json
import sys
from pathlib import Path

CAPABILITIES = {capabilities!r}
VERIFY_RESPONSE = {verify_response!r}

if sys.argv[1:] == ['--pcss-capabilities']:
    {capability_count or 'pass'}
    print(json.dumps(CAPABILITIES, sort_keys=True))
elif sys.argv[1:] == ['--pcss-verify-case']:
    request = sys.stdin.read()
    {request_write or 'assert request'}
    print(json.dumps(VERIFY_RESPONSE, sort_keys=True))
else:
    raise SystemExit(2)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_config(tmp_path: Path, bridge: Path, model: Path, library: Path, *, extra: dict[str, object] | None = None) -> Path:
    payload: dict[str, object] = {
        "schema_version": "pcss.llamacpp-adapter-config/v1",
        "bridge_protocol": "pcss.llama-bridge/v1",
        "bridge_executable": str(bridge),
        "model_artifact": str(model),
        "llama_cpp_library": str(library),
        "llama_cpp_revision": "b5401",
        "runtime": _runtime(),
        "request_timeout_seconds": 5,
    }
    if extra:
        payload.update(extra)
    config = tmp_path / "llamacpp-config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _ready_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    comparison: str = "byte_exact",
    state_round_trip: bool = True,
    malformed_response: bool = False,
) -> tuple[LlamaCppBridgeAdapter, Path, Path, Path]:
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    bridge = tmp_path / "pcss_llama_bridge"
    model.write_bytes(b"deterministic model fixture")
    library.write_bytes(b"deterministic library fixture")
    runtime_fingerprint = _runtime_fingerprint(library, model)
    _write_bridge(
        bridge,
        capabilities=_capabilities(
            library,
            model,
            runtime_fingerprint=runtime_fingerprint,
            comparison=comparison,
            state_round_trip=state_round_trip,
        ),
        verify_response=_outcome_response(runtime_fingerprint=runtime_fingerprint, malformed=malformed_response),
    )
    config = _write_config(tmp_path, bridge, model, library)
    monkeypatch.setenv("PCSS_LLAMACPP_CONFIG", str(config))
    return LlamaCppBridgeAdapter(), bridge, model, library


def _case() -> VerifierCase:
    values: dict[str, object] = {
        "case_id": "partial-rejection",
        "draft_tokens": (7, 9),
        "target_tokens": (7, 8),
        "expected_committed_tokens": (7, 8),
        "expected_state_digest": "a" * 64,
        "k": 2,
    }
    # The verifier contract adds this field after the first adapter increment.
    if "initial_context_tokens" in getattr(VerifierCase, "__dataclass_fields__", {}):
        values["initial_context_tokens"] = (101, 102)
    return VerifierCase(**values)


def test_missing_explicit_config_is_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCSS_LLAMACPP_CONFIG", raising=False)
    availability = LlamaCppBridgeAdapter().availability()
    assert availability.execution_class == "NOT_RUN"
    assert "PCSS_LLAMACPP_CONFIG" in availability.reason


def test_unknown_config_field_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    bridge = tmp_path / "pcss_llama_bridge"
    model.write_bytes(b"model")
    library.write_bytes(b"library")
    runtime_fingerprint = _runtime_fingerprint(library, model)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, runtime_fingerprint=runtime_fingerprint),
        verify_response=_outcome_response(runtime_fingerprint=runtime_fingerprint),
    )
    config = _write_config(tmp_path, bridge, model, library, extra={"unrecognized": True})
    monkeypatch.setenv("PCSS_LLAMACPP_CONFIG", str(config))

    availability = LlamaCppBridgeAdapter().availability()

    assert availability.execution_class == "NOT_RUN"
    assert "unknown" in availability.reason


def test_ambiguous_capability_response_never_claims_target_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch, comparison="semantic")

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "byte exact" in availability.reason


def test_static_support_without_model_state_round_trip_is_not_target_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch, state_round_trip=False)

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "round trip" in availability.reason


def test_target_binding_uses_real_configured_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)

    availability = adapter.availability()
    binding = adapter.target_binding()

    assert availability.execution_class == "TARGET_BACKEND"
    assert binding.adapter_source_path.endswith("pcss/llamacpp_adapter.py")
    assert binding.backend_binary_path == str(bridge.resolve())
    assert binding.model_artifact_path == str(model.resolve())
    assert len(binding.llama_cpp_build_fingerprint) == 64
    if hasattr(binding, "runtime_configuration_fingerprint"):
        assert len(binding.runtime_configuration_fingerprint) == 64

    previous_fingerprint = binding.llama_cpp_build_fingerprint
    library.write_bytes(b"changed library fixture")
    runtime_fingerprint = _runtime_fingerprint(library, model)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, runtime_fingerprint=runtime_fingerprint),
        verify_response=_outcome_response(runtime_fingerprint=runtime_fingerprint),
    )
    assert adapter.target_binding().llama_cpp_build_fingerprint == previous_fingerprint
    refreshed = LlamaCppBridgeAdapter()
    assert refreshed.availability().execution_class == "TARGET_BACKEND"
    assert refreshed.target_binding().llama_cpp_build_fingerprint != previous_fingerprint


def test_availability_runs_one_frozen_model_loading_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    counter = tmp_path / "capability-count.txt"
    runtime_fingerprint = _runtime_fingerprint(library, model)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, runtime_fingerprint=runtime_fingerprint),
        verify_response=_outcome_response(runtime_fingerprint=runtime_fingerprint),
        capability_count_path=counter,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    assert adapter.availability().execution_class == "TARGET_BACKEND"
    adapter.target_binding()

    assert counter.read_text(encoding="utf-8") == "1"


def test_verify_passes_strict_case_request_and_maps_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    request_path = tmp_path / "bridge-request.json"
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, runtime_fingerprint=_runtime_fingerprint(library, model)),
        verify_response=_outcome_response(runtime_fingerprint=_runtime_fingerprint(library, model)),
        request_path=request_path,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    outcome = adapter.verify(_case())
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert outcome.accepted_draft_tokens == 1
    assert outcome.correction_token == 8
    assert outcome.bonus_token is None
    assert outcome.committed_tokens == (7, 8)
    assert request["operation"] == "verify_case"
    assert request["case"]["draft_tokens"] == [7, 9]
    assert "expected_state_digest" not in request["case"]


def test_malformed_bridge_verify_response_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch, malformed_response=True)
    assert adapter.availability().execution_class == "TARGET_BACKEND"

    with pytest.raises(ContractError, match="runtime_configuration_fingerprint"):
        adapter.verify(_case())


def test_verify_requires_one_frozen_capability_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch)

    with pytest.raises(ContractError, match="availability\\(\\)"):
        adapter.verify(_case())
