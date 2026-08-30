from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from pcss.canonical import ContractError, sha256
from pcss.llamacpp_adapter import LlamaCppBridgeAdapter
from pcss.verifier import TargetFixtureProvenance, VerifierCase


def _runtime() -> dict[str, object]:
    return {
        "n_ctx": 512,
        "n_batch": 128,
        "n_gpu_layers": 99,
        "flash_attn": "auto",
        "seed": 7,
        "n_threads": 2,
        "n_threads_batch": 2,
        "chat_template": "caller-pretokenized/v1",
        "thinking_mode": "disabled",
    }


def _configuration(library: Path, model: Path) -> dict[str, object]:
    return {
        "model_artifact": str(model.resolve()),
        "llama_cpp_library": str(library.resolve()),
        "llama_cpp_revision": "b5401",
        "runtime": _runtime(),
    }


def _runtime_fingerprint(library: Path, model: Path) -> str:
    return sha256(
        {
            "schema_version": "pcss.llamacpp-runtime-config/v1",
            "bridge_protocol": "pcss.llama-bridge/v1",
            **_configuration(library, model),
        }
    )


def _capabilities(library: Path, model: Path, **overrides: object) -> dict[str, object]:
    response: dict[str, object] = {
        "schema_version": "pcss.llama-bridge-capabilities/v2",
        "operation": "capabilities",
        "status": "ok",
        "backend": "llama.cpp",
        "state_export": "model-architecture+memory-state/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "raw_token_ids": True,
        "llama_c_api_version": "not-exposed-by-stable-c-api/v1",
        "supports_gpu_offload": True,
        "loaded_llama_cpp_library_path": str(library.resolve()),
        "supports_mmap": True,
        "supports_mlock": False,
        "vocabulary_size": 128,
        "configuration": _configuration(library, model),
        "observed_context": {
            "actual_n_ctx": 512,
            "actual_n_batch": 128,
            "actual_n_threads": 2,
            "actual_n_threads_batch": 2,
        },
        "state_components": ["model_architecture_identity", "llama_memory"],
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "sampler_rng_state": "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1",
        "probe_token": 0,
        "probe_state_sha256": "b" * 64,
        "probe_state_bytes": 64,
        "restored_state_sha256": "b" * 64,
        "restored_state_bytes": 64,
        "state_round_trip_byte_exact": True,
        "operations": ["capabilities", "reference", "verify", "profile"],
        "declared_identity_fields": "caller-declared-not-verified-by-bridge/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
    }
    response.update(overrides)
    return response


def _verify_response(
    library: Path,
    model: Path,
    *,
    target_tokens: tuple[int, ...] = (7, 8, 12),
    accepted_draft_tokens: int = 1,
    correction_token: int | None = 8,
    bonus_token: int | None = None,
    committed_tokens: tuple[int, ...] = (7, 8),
    digest: str = "a" * 64,
    **overrides: object,
) -> dict[str, object]:
    response: dict[str, object] = {
        "schema_version": "pcss.llama-bridge-verify/v2",
        "operation": "verify",
        "status": "ok",
        "backend": "llama.cpp",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "state_export": "model-architecture+memory-state/v2",
        "reconstruction": "batch-score+serial-canonical-commit/v2",
        "seed_id": "fixture-seed-001",
        "case_id": "partial-rejection",
        "target_tokens": list(target_tokens),
        "verification_target_tokens": list(target_tokens),
        "verification_targets_match": True,
        "vocabulary_size": 128,
        "accepted_draft_tokens": accepted_draft_tokens,
        "reference_accepted_draft_tokens": accepted_draft_tokens,
        "correction_token": correction_token,
        "bonus_token": bonus_token,
        "committed_tokens": list(committed_tokens),
        "reference_committed_tokens": list(committed_tokens),
        "prompt_state_sha256": "c" * 64,
        "prompt_state_bytes": 64,
        "reference_state_sha256": digest,
        "reference_state_bytes": 72,
        "committed_state_sha256": digest,
        "committed_state_bytes": 72,
        "verifier_draft_state_sha256": "e" * 64,
        "verifier_draft_state_bytes": 80,
        "reference_mutable_state_digest": digest,
        "committed_mutable_state_digest": digest,
        "mutable_state_digest": digest,
        "state_matches": True,
        "reference_next_token": 13,
        "committed_next_token": 13,
        "continuation_matches": True,
        "configuration": _configuration(library, model),
        "observed_context": {
            "actual_n_ctx": 512,
            "actual_n_batch": 128,
            "actual_n_threads": 2,
            "actual_n_threads_batch": 2,
        },
        "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "sampler_rng_state": "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1",
        "declared_identity_fields": "caller-declared-not-verified-by-bridge/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
    }
    response.update(overrides)
    return response


def _pending_verify_response(library: Path, model: Path, **overrides: object) -> dict[str, object]:
    response = _verify_response(
        library,
        model,
        schema_version="pcss.llama-bridge-pending-verify/v1",
        operation="verify_pending",
        reconstruction="pending-anchor+batch-score+native-tail-trim/v1",
    )
    response.update(
        {
            "pending_state_sha256": "d" * 64,
            "pending_state_bytes": 68,
            "reference_pending_state_sha256": "d" * 64,
            "reference_pending_state_bytes": 68,
            "pending_state_matches": True,
            "tail_trimmed": True,
            "pending_token": 8,
        }
    )
    response.update(overrides)
    return response


def _write_bridge(
    path: Path,
    *,
    capabilities: dict[str, object],
    verify_response: dict[str, object],
    request_log_path: Path | None = None,
    capability_count_path: Path | None = None,
    verify_exit_status: int = 0,
) -> None:
    script = f"""#!{sys.executable}
import json
import sys
from pathlib import Path

CAPABILITIES = {capabilities!r}
VERIFY_RESPONSE = {verify_response!r}
REQUEST_LOG_PATH = {str(request_log_path) if request_log_path is not None else None!r}
CAPABILITY_COUNT_PATH = {str(capability_count_path) if capability_count_path is not None else None!r}
VERIFY_EXIT_STATUS = {verify_exit_status!r}

def read_request():
    payload = json.loads(sys.stdin.read())
    if REQUEST_LOG_PATH is not None:
        with Path(REQUEST_LOG_PATH).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({{"argv": sys.argv[1:], "request": payload}}, sort_keys=True) + "\\n")
    return payload

read_request()
if sys.argv[1:] == ["--pcss-capabilities"]:
    if CAPABILITY_COUNT_PATH is not None:
        counter = Path(CAPABILITY_COUNT_PATH)
        count = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
        counter.write_text(str(count + 1), encoding="utf-8")
    print(json.dumps(CAPABILITIES, sort_keys=True))
elif sys.argv[1:] in (["--pcss-verify-case"], ["--pcss-verify-pending-case"]):
    print(json.dumps(VERIFY_RESPONSE, sort_keys=True))
    raise SystemExit(VERIFY_EXIT_STATUS)
else:
    raise SystemExit(2)
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_config(
    tmp_path: Path,
    bridge: Path,
    model: Path,
    library: Path,
    *,
    extra: dict[str, object] | None = None,
) -> Path:
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
    if extra is not None:
        payload.update(extra)
    config = tmp_path / "llamacpp-config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _ready_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LlamaCppBridgeAdapter, Path, Path, Path]:
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    bridge = tmp_path / "pcss_llama_bridge"
    model.write_bytes(b"deterministic model fixture")
    library.write_bytes(b"deterministic library fixture")
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model),
    )
    config = _write_config(tmp_path, bridge, model, library)
    monkeypatch.setenv("PCSS_LLAMACPP_CONFIG", str(config))
    return LlamaCppBridgeAdapter(), bridge, model, library


def _case(**overrides: object) -> VerifierCase:
    values: dict[str, object] = {
        "case_id": "partial-rejection",
        "draft_tokens": (7, 9),
        "target_tokens": (7, 8),
        "expected_committed_tokens": (7, 8),
        "expected_state_digest": "a" * 64,
        "k": 2,
        "initial_context_tokens": (101, 102),
        "seed_id": "fixture-seed-001",
    }
    values.update(overrides)
    return VerifierCase(**values)


def _requests(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_missing_explicit_config_is_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PCSS_LLAMACPP_CONFIG", raising=False)

    availability = LlamaCppBridgeAdapter().availability()

    assert availability.execution_class == "NOT_RUN"
    assert "PCSS_LLAMACPP_CONFIG" in availability.reason


def test_unknown_config_field_is_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch)
    config = tmp_path / "llamacpp-config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["unrecognized"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "unknown" in availability.reason


def test_boolean_flash_attn_is_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch)
    config = tmp_path / "llamacpp-config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["runtime"]["flash_attn"] = True
    config.write_text(json.dumps(payload), encoding="utf-8")

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "flash_attn" in availability.reason


def test_incomplete_full_state_probe_never_claims_target_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, state_round_trip_byte_exact=False),
        verify_response=_verify_response(library, model),
    )

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "round trip" in availability.reason


def test_malformed_capability_state_fields_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, state_components=1),
        verify_response=_verify_response(library, model),
    )

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "model-memory state coverage" in availability.reason


def test_library_path_mismatch_is_not_target_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, loaded_llama_cpp_library_path=str(model.resolve())),
        verify_response=_verify_response(library, model),
    )

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "loaded library path differs" in availability.reason


def test_configuration_echo_mismatch_is_not_target_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    mismatched_configuration = _configuration(library, model)
    mismatched_configuration["llama_cpp_revision"] = "different-revision"
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model, configuration=mismatched_configuration),
        verify_response=_verify_response(library, model),
    )

    availability = adapter.availability()

    assert availability.execution_class == "NOT_RUN"
    assert "differs from explicit adapter configuration" in availability.reason


def test_target_binding_uses_frozen_configured_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    binding = adapter.target_binding()

    assert binding.schema_version == "pcss.verifier-adapter-binding/v2"
    assert binding.adapter_source_path.endswith("pcss/llamacpp_adapter.py")
    assert binding.backend_binary_path == str(bridge.resolve())
    assert binding.model_artifact_path == str(model.resolve())
    assert binding.runtime_configuration_fingerprint == _runtime_fingerprint(library, model)
    previous_fingerprint = binding.llama_cpp_build_fingerprint

    library.write_bytes(b"changed library fixture")
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model),
    )

    assert adapter.target_binding().llama_cpp_build_fingerprint == previous_fingerprint
    refreshed = LlamaCppBridgeAdapter()
    assert refreshed.availability().execution_class == "TARGET_BACKEND"
    assert refreshed.target_binding().llama_cpp_build_fingerprint != previous_fingerprint


def test_availability_runs_one_frozen_model_loading_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    counter = tmp_path / "capability-count.txt"
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model),
        capability_count_path=counter,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    assert adapter.availability().execution_class == "TARGET_BACKEND"
    adapter.target_binding()

    assert counter.read_text(encoding="utf-8") == "1"


def test_verify_uses_exact_native_request_and_maps_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    request_log = tmp_path / "requests.jsonl"
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model),
        request_log_path=request_log,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    outcome = adapter.verify(_case())
    requests = _requests(request_log)

    assert outcome.accepted_draft_tokens == 1
    assert outcome.correction_token == 8
    assert outcome.bonus_token is None
    assert outcome.committed_tokens == (7, 8)
    assert outcome.mutable_state_digest == "a" * 64
    assert len(requests) == 2
    capability_request = requests[0]["request"]
    verify_request = requests[1]["request"]
    assert requests[0]["argv"] == ["--pcss-capabilities"]
    assert requests[1]["argv"] == ["--pcss-verify-case"]
    assert set(capability_request) == {
        "schema_version",
        "operation",
        "model_artifact",
        "llama_cpp_library",
        "llama_cpp_revision",
        "runtime",
    }
    assert capability_request["operation"] == "capabilities"
    assert set(verify_request) == {
        "schema_version",
        "operation",
        "model_artifact",
        "llama_cpp_library",
        "llama_cpp_revision",
        "runtime",
        "seed",
        "case",
    }
    assert verify_request["operation"] == "verify"
    assert verify_request["seed"] == {"seed_id": "fixture-seed-001", "initial_context_tokens": [101, 102]}
    assert verify_request["case"] == {
        "case_id": "partial-rejection",
        "draft_tokens": [7, 9],
        "include_bonus_on_full_accept": False,
    }
    assert "target_tokens" not in verify_request
    assert "expected_committed_tokens" not in verify_request
    assert "expected_state_digest" not in verify_request


def test_verify_maps_structured_native_mismatch_when_bridge_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(
            library,
            model,
            status="mismatch",
            state_matches=False,
            committed_state_sha256="b" * 64,
            committed_state_bytes=73,
            committed_mutable_state_digest="b" * 64,
            mutable_state_digest="b" * 64,
        ),
        verify_exit_status=1,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    outcome = adapter.verify(_case())

    assert not outcome.reconstruction_matches
    assert outcome.continuation_matches


def test_verify_maps_batch_target_divergence_as_measured_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(
            library,
            model,
            status="mismatch",
            verification_target_tokens=[7, 10, 12],
            verification_targets_match=False,
            correction_token=10,
            committed_tokens=[7, 10],
            reference_committed_tokens=[7, 8],
            committed_state_sha256="b" * 64,
            committed_mutable_state_digest="b" * 64,
            mutable_state_digest="b" * 64,
            state_matches=False,
            committed_next_token=14,
            continuation_matches=False,
        ),
        verify_exit_status=1,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    outcome = adapter.verify(_case())

    assert not outcome.verification_matches
    assert outcome.committed_tokens == (7, 10)
    assert not outcome.reconstruction_matches
    assert not outcome.continuation_matches


def test_verify_rejects_structured_mismatch_with_an_unexpected_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model, status="mismatch", state_matches=False),
        verify_exit_status=2,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="non-zero exit status"):
        adapter.verify(_case())


def test_verify_rejects_native_target_prefix_that_disagrees_with_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(
            library,
            model,
            target_tokens=(7, 10, 12),
            correction_token=10,
            committed_tokens=(7, 10),
        ),
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="target prefix"):
        adapter.verify(_case())


def test_verify_rejects_fixture_commit_and_state_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="fixture expectation"):
        adapter.verify(_case(expected_committed_tokens=(7, 99)))

    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response=_verify_response(library, model, digest="d" * 64),
    )
    with pytest.raises(ContractError, match="fixture expectation"):
        adapter.verify(_case())


def test_verify_requires_fixture_declared_seed_and_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _ready_adapter(tmp_path, monkeypatch)

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="seed_id"):
        adapter.verify(_case(seed_id=None))
    with pytest.raises(ContractError, match="initial_context_tokens"):
        adapter.verify(_case(initial_context_tokens=()))


def test_malformed_verify_response_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    _write_bridge(
        bridge,
        capabilities=_capabilities(library, model),
        verify_response={"schema_version": "pcss.llama-bridge-verify/v2"},
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="invalid keys"):
        adapter.verify(_case())


def test_pending_native_adapter_uses_explicit_operation_and_validates_pending_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    adapter = LlamaCppBridgeAdapter(
        environment={"PCSS_LLAMACPP_CONFIG": str(tmp_path / "llamacpp-config.json")},
        commit_protocol="pending_native",
    )
    request_log = tmp_path / "pending-requests.jsonl"
    _write_bridge(
        bridge,
        capabilities=_capabilities(
            library,
            model,
            operations=["capabilities", "reference", "verify", "verify_pending", "profile"],
        ),
        verify_response=_pending_verify_response(library, model),
        request_log_path=request_log,
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    outcome = adapter.verify(_case())

    assert outcome.reconstruction_matches
    calls = _requests(request_log)
    assert calls[1]["argv"] == ["--pcss-verify-pending-case"]
    assert calls[1]["request"]["operation"] == "verify_pending"


def test_pending_native_adapter_rejects_false_pending_state_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bridge, model, library = _ready_adapter(tmp_path, monkeypatch)
    adapter = LlamaCppBridgeAdapter(
        environment={"PCSS_LLAMACPP_CONFIG": str(tmp_path / "llamacpp-config.json")},
        commit_protocol="pending_native",
    )
    _write_bridge(
        bridge,
        capabilities=_capabilities(
            library,
            model,
            operations=["capabilities", "reference", "verify", "verify_pending", "profile"],
        ),
        verify_response=_pending_verify_response(library, model, pending_state_matches=False),
    )

    assert adapter.availability().execution_class == "TARGET_BACKEND"
    with pytest.raises(ContractError, match="pending-state match claim"):
        adapter.verify(_case())


def test_pending_native_adapter_rejects_serial_reference_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, _ = _ready_adapter(tmp_path, monkeypatch)
    adapter = LlamaCppBridgeAdapter(
        environment={"PCSS_LLAMACPP_CONFIG": str(tmp_path / "llamacpp-config.json")},
        commit_protocol="pending_native",
    )
    provenance = TargetFixtureProvenance(
        schema_version="pcss.target-verifier-fixture-provenance/v2",
        reference_method="serial-greedy-memory-state/v2",
        model_artifact_sha256="a" * 64,
        runtime_configuration_fingerprint="b" * 64,
    )

    with pytest.raises(ContractError, match="reference method is incompatible"):
        adapter.verify(_case(reference_provenance=provenance))
