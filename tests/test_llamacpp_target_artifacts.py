from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError, canonical_json, sha256
from pcss.llamacpp_fixture_builder import load_bridge_config, load_fixture_seeds
from pcss.llamacpp_target_artifacts import (
    CAPABILITY_SCHEMA_VERSION,
    MODEL_DESCRIPTOR_SCHEMA_VERSION,
    SEED_PLAN_SCHEMA_VERSION,
    build_capability_seed_context,
    build_observed_verifier_manifest,
)
from pcss.manifest import validate_manifest
from pcss.results import file_sha256
from pcss.verifier import TargetHostIdentity
from pcss.verifier_runner import _require_observed_target_manifest


ROOT = Path(__file__).parents[1]


def _target_identity() -> TargetHostIdentity:
    return TargetHostIdentity.from_observed(
        target_profile="macbook_pro_m4_24gb",
        operating_system="Darwin 27.0",
        architecture="arm64",
        hardware_model="MacBookPro16,8",
        chip="Apple M4 Pro",
        memory_bytes=24 * 1024**3,
    )


def _config(tmp_path: Path) -> Path:
    bridge = tmp_path / "pcss_llama_bridge"
    bridge.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    bridge.chmod(0o755)
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    model.write_bytes(b"real target model artifact for unit fixture")
    library.write_bytes(b"real target llama cpp library for unit fixture")
    payload = {
        "schema_version": "pcss.llamacpp-adapter-config/v1",
        "bridge_protocol": "pcss.llama-bridge/v1",
        "bridge_executable": str(bridge),
        "model_artifact": str(model),
        "llama_cpp_library": str(library),
        "llama_cpp_revision": "test-llama-cpp-revision",
        "runtime": {
            "n_ctx": 64,
            "n_batch": 16,
            "n_gpu_layers": 99,
            "flash_attn": "disabled",
            "seed": 17,
            "n_threads": 4,
            "n_threads_batch": 4,
            "chat_template": "qwen3",
            "thinking_mode": "disabled",
        },
        "request_timeout_seconds": 5,
    }
    path = tmp_path / "config.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def _plan(tmp_path: Path) -> Path:
    payload = {
        "schema_version": SEED_PLAN_SCHEMA_VERSION,
        "seed_id": "loaded-model-probe-token",
        "k_values": [2, 3],
        "provider_floor_k": 2,
        "forced_rejection_k": 3,
        "long_mixed": {"context_steps": 4, "k": 3, "rejection_position": 1},
    }
    path = tmp_path / "seed-plan.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def _capability(config, *, probe_token: int = 7) -> dict:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "operation": "capabilities",
        "status": "ok",
        "backend": "llama.cpp",
        "state_export": "model-architecture+memory-state/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "raw_token_ids": True,
        "llama_c_api_version": "not-exposed-by-stable-c-api/v1",
        "loaded_llama_cpp_library_path": str(config.llama_cpp_library),
        "supports_gpu_offload": True,
        "supports_mmap": True,
        "supports_mlock": False,
        "vocabulary_size": 29,
        "configuration": {
            "model_artifact": str(config.model_artifact),
            "llama_cpp_library": str(config.llama_cpp_library),
            "llama_cpp_revision": config.llama_cpp_revision,
            "runtime": dict(config.runtime),
        },
        "observed_context": {
            "actual_n_ctx": config.runtime["n_ctx"],
            "actual_n_batch": config.runtime["n_batch"],
            "actual_n_threads": config.runtime["n_threads"],
            "actual_n_threads_batch": config.runtime["n_threads_batch"],
        },
        "state_components": ["model_architecture_identity", "llama_memory"],
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "declared_identity_fields": "caller-declared-not-verified-by-bridge/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
        "sampler_rng_state": "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1",
        "probe_token": probe_token,
        "probe_state_sha256": "a" * 64,
        "probe_state_bytes": 128,
        "restored_state_sha256": "a" * 64,
        "restored_state_bytes": 128,
        "state_round_trip_byte_exact": True,
        "operations": ["capabilities", "reference", "verify", "profile"],
    }


@pytest.fixture(autouse=True)
def _deterministic_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pcss.llamacpp_target_artifacts.resolve_git_revision", lambda root: "f" * 40)
    monkeypatch.setattr(
        "pcss.llamacpp_target_artifacts.recorded_module_command",
        lambda module, arguments, root: f"PYTHONPATH=/fixture python -m {module}",
    )


def _seed_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "seeds.json", tmp_path / "seed.raw.json", tmp_path / "seed.report.md"


def test_capability_seed_is_nonempty_and_fixture_builder_compatible(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = load_bridge_config(config_path)
    plan_path = _plan(tmp_path)
    output, raw_result, report = _seed_artifacts(tmp_path)

    receipt = build_capability_seed_context(
        config_path=config_path,
        plan_path=plan_path,
        output_path=output,
        raw_result_path=raw_result,
        report_path=report,
        repository_root=ROOT,
        command_arguments=("seed",),
        target_observer=_target_identity,
        capability_runner=lambda loaded: _capability(loaded, probe_token=7),
    )

    assert receipt.evidence_class == "LOCAL_MEASURED"
    assert receipt.status == "SEED_CONTEXT_OBSERVED"
    assert json.loads(output.read_text(encoding="utf-8"))["seeds"] == [
        {"seed_id": "loaded-model-probe-token", "initial_context_tokens": [7]}
    ]
    assert load_fixture_seeds(output).contexts == (("loaded-model-probe-token", (7,)),)
    raw = json.loads(raw_result.read_text(encoding="utf-8"))
    assert raw["payload"]["capability"]["probe_token"] == 7
    assert raw["payload"]["claim_boundary"]["not_claimed"] == [
        "hardware_performance",
        "timing",
        "provider_economics",
        "live_speculative_decode",
    ]
    assert "Raw result" in report.read_text(encoding="utf-8")
    assert config.runtime_configuration_fingerprint


@pytest.mark.parametrize("mutator", ["missing-token", "out-of-range-token", "wrong-config", "wrong-library"])
def test_capability_seed_fails_closed_without_any_artifact(
    tmp_path: Path, mutator: str
) -> None:
    config_path = _config(tmp_path)
    plan_path = _plan(tmp_path)
    output, raw_result, report = _seed_artifacts(tmp_path)

    def malformed(config):
        response = _capability(config)
        if mutator == "missing-token":
            del response["probe_token"]
        elif mutator == "out-of-range-token":
            response["probe_token"] = response["vocabulary_size"]
        elif mutator == "wrong-config":
            response["configuration"]["runtime"] = {**response["configuration"]["runtime"], "n_ctx": 65}
        else:
            response["loaded_llama_cpp_library_path"] = str(config.model_artifact)
        return response

    with pytest.raises(ContractError):
        build_capability_seed_context(
            config_path=config_path,
            plan_path=plan_path,
            output_path=output,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=ROOT,
            command_arguments=("seed",),
            target_observer=_target_identity,
            capability_runner=malformed,
        )
    assert not output.exists()
    assert not raw_result.exists()
    assert not report.exists()


def test_capability_seed_refuses_overwrite_before_bridge_call(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    plan_path = _plan(tmp_path)
    output, raw_result, report = _seed_artifacts(tmp_path)
    output.write_text("prior", encoding="utf-8")
    calls = 0

    def should_not_run(config):
        nonlocal calls
        calls += 1
        return _capability(config)

    with pytest.raises(ContractError, match="overwrite"):
        build_capability_seed_context(
            config_path=config_path,
            plan_path=plan_path,
            output_path=output,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=ROOT,
            command_arguments=("seed",),
            target_observer=_target_identity,
            capability_runner=should_not_run,
        )
    assert calls == 0
    assert not raw_result.exists()
    assert not report.exists()


def _protocol(parameters: dict) -> dict:
    return {"fingerprint": sha256(parameters), "parameters": parameters}


def _descriptor(tmp_path: Path, config_path: Path, *, model_sha256: str | None = None) -> Path:
    config = load_bridge_config(config_path)
    model_hash = model_sha256 or file_sha256(config.model_artifact)
    binding_build = "b" * 64
    payload = {
        "schema_version": MODEL_DESCRIPTOR_SCHEMA_VERSION,
        "run_id": "qwen3-8b-q4km-target-gate",
        "model_artifact": {
            "artifact_uri": "hf://Qwen/Qwen3-8B-GGUF@revision/Qwen3-8B-Q4_K_M.gguf",
            "sha256": model_hash,
            "source": {
                "source_class": "huggingface",
                "locator": "Qwen/Qwen3-8B-GGUF",
                "revision": "revision",
                "artifact_name": "Qwen3-8B-Q4_K_M.gguf",
            },
        },
        "quantization": _protocol({"format": "gguf", "quantization": "Q4_K_M"}),
        "tokenizer": _protocol({"source": "model-artifact"}),
        "chat_template": _protocol({"template": config.runtime["chat_template"]}),
        "generation_protocol": _protocol(
            {
                "generation_mode": "greedy_argmax",
                "token_input": "caller-supplied-raw-token-ids/v1",
                "thinking_mode": "disabled",
            }
        ),
        "llama_cpp": _protocol(
            {
                "llama_cpp_revision": config.llama_cpp_revision,
                "llama_cpp_build_fingerprint": binding_build,
                "llama_cpp_library_sha256": file_sha256(config.llama_cpp_library),
            }
        ),
        "context_kv": _protocol(
            {
                "runtime_configuration_fingerprint": config.runtime_configuration_fingerprint,
                "runtime": dict(config.runtime),
            }
        ),
        "benchmark": _protocol({"measurement_status": "not_run", "parameters": {"suite": "future"}}),
    }
    path = tmp_path / "model-descriptor.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def _preflight(tmp_path: Path, config_path: Path, identity: TargetHostIdentity) -> Path:
    config = load_bridge_config(config_path)
    binding = {
        "schema_version": "pcss.verifier-adapter-binding/v2",
        "adapter_source_sha256": file_sha256(ROOT / "src" / "pcss" / "llamacpp_adapter.py"),
        "backend_binary_sha256": file_sha256(config.bridge_executable),
        "model_artifact_sha256": file_sha256(config.model_artifact),
        "llama_cpp_build_fingerprint": "b" * 64,
        "runtime_configuration_fingerprint": config.runtime_configuration_fingerprint,
    }
    payload = {
        "schema_version": "pcss.result/v1",
        "evidence_class": "NOT_RUN",
        "command": "PYTHONPATH=/fixture python -m pcss.verifier_preflight",
        "git_revision": "f" * 40,
        "input_fingerprints": {
            "adapter_spec": "a" * 64,
            "adapter_source": binding["adapter_source_sha256"],
            "backend_binary": binding["backend_binary_sha256"],
            "model_artifact": binding["model_artifact_sha256"],
            "llama_cpp_build": binding["llama_cpp_build_fingerprint"],
            "runtime_configuration": binding["runtime_configuration_fingerprint"],
        },
        "status": "NOT_RUN",
        "payload": {
            "schema_version": "pcss.m4-target-observation/v1",
            "observation_status": "observed",
            "adapter": "llamacpp-full-state-bridge",
            "declared_execution_class": "TARGET_BACKEND",
            "reason": None,
            "host_identity": identity.as_dict(),
            "adapter_binding": binding,
        },
    }
    path = tmp_path / "preflight.raw.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def _manifest_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "manifest.json", tmp_path / "manifest.raw.json", tmp_path / "manifest.report.md"


def test_observed_manifest_is_bound_to_preflight_config_and_source_descriptor(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    identity = _target_identity()
    preflight = _preflight(tmp_path, config_path, identity)
    descriptor = _descriptor(tmp_path, config_path)
    output, raw_result, report = _manifest_artifacts(tmp_path)

    receipt = build_observed_verifier_manifest(
        config_path=config_path,
        preflight_receipt_path=preflight,
        descriptor_path=descriptor,
        output_path=output,
        raw_result_path=raw_result,
        report_path=report,
        repository_root=ROOT,
        command_arguments=("manifest",),
        target_observer=_target_identity,
    )

    assert receipt.evidence_class == "NOT_RUN"
    assert receipt.status == "OBSERVED_MANIFEST_BUILT"
    built = json.loads(output.read_text(encoding="utf-8"))
    validate_manifest(built)
    _require_observed_target_manifest(built)
    assert built["evidence_class"] == "LOCAL_MEASURED"
    assert built["model_artifact"]["sha256"] == file_sha256(load_bridge_config(config_path).model_artifact)
    assert built["target"]["hardware"]["value"]["chip"] == "Apple M4 Pro"
    assert built["verifier_gate"]["host_identity"] == identity.as_dict()
    raw = json.loads(raw_result.read_text(encoding="utf-8"))
    assert raw["payload"]["source"]["locator"] == "Qwen/Qwen3-8B-GGUF"
    assert raw["payload"]["claim_boundary"]["not_claimed"][-1] == "verifier_correctness"
    assert "Raw result" in report.read_text(encoding="utf-8")


@pytest.mark.parametrize("mutator", ["wrong-model", "wrong-runtime", "other-host"])
def test_manifest_fails_closed_on_mismatched_target_binding_or_descriptor(
    tmp_path: Path, mutator: str
) -> None:
    config_path = _config(tmp_path)
    identity = _target_identity()
    preflight = _preflight(tmp_path, config_path, identity)
    descriptor = _descriptor(tmp_path, config_path, model_sha256=("c" * 64 if mutator == "wrong-model" else None))
    if mutator == "wrong-runtime":
        receipt = json.loads(preflight.read_text(encoding="utf-8"))
        receipt["payload"]["adapter_binding"]["runtime_configuration_fingerprint"] = "d" * 64
        receipt["input_fingerprints"]["runtime_configuration"] = "d" * 64
        preflight.write_text(canonical_json(receipt), encoding="utf-8")
    output, raw_result, report = _manifest_artifacts(tmp_path)
    def other_host_identity() -> TargetHostIdentity:
        return TargetHostIdentity.from_observed(
            target_profile="other-target",
            operating_system="Darwin 27.0",
            architecture="arm64",
            hardware_model="MacBookPro16,8",
            chip="Apple M4 Pro",
            memory_bytes=24 * 1024**3,
        )

    observer = _target_identity
    if mutator == "other-host":
        observer = other_host_identity

    with pytest.raises(ContractError):
        build_observed_verifier_manifest(
            config_path=config_path,
            preflight_receipt_path=preflight,
            descriptor_path=descriptor,
            output_path=output,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=ROOT,
            command_arguments=("manifest",),
            target_observer=observer,
        )
    assert not output.exists()
    assert not raw_result.exists()
    assert not report.exists()


def test_manifest_rejects_descriptor_benchmark_claim(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    preflight = _preflight(tmp_path, config_path, _target_identity())
    descriptor = _descriptor(tmp_path, config_path)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    parameters = {"measurement_status": "measured", "parameters": {"speedup": 2}}
    payload["benchmark"] = _protocol(parameters)
    descriptor.write_text(canonical_json(payload), encoding="utf-8")
    output, raw_result, report = _manifest_artifacts(tmp_path)

    with pytest.raises(ContractError, match="must remain not_run"):
        build_observed_verifier_manifest(
            config_path=config_path,
            preflight_receipt_path=preflight,
            descriptor_path=descriptor,
            output_path=output,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=ROOT,
            command_arguments=("manifest",),
            target_observer=_target_identity,
        )
    assert not output.exists()
    assert not raw_result.exists()
    assert not report.exists()
