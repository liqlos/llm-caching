from __future__ import annotations

from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.llamacpp_fixture_builder import BridgeConfig
from pcss.llamacpp_profiler import _analyze, _request, _validate_response


def _config(tmp_path: Path) -> BridgeConfig:
    bridge = tmp_path / "bridge"
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    config_path = tmp_path / "config.json"
    for path in (bridge, model, library, config_path):
        path.write_bytes(b"fixture")
    runtime = {
        "n_ctx": 128,
        "n_batch": 32,
        "n_gpu_layers": 99,
        "flash_attn": "auto",
        "seed": 0,
        "n_threads": 4,
        "n_threads_batch": 4,
        "chat_template": "caller-pretokenized/v1",
        "thinking_mode": "disabled",
    }
    return BridgeConfig(config_path, bridge, model, library, "build-1", runtime, 30, "f" * 64)


def _measurement(iteration: int, *, before: int = 200, batch: int = 100, after: int = 202) -> dict:
    return {
        "iteration": iteration,
        "restore_before_ns": 20,
        "serial_before_ns": before,
        "restore_batch_ns": 21,
        "batch_verify_ns": batch,
        "restore_after_ns": 22,
        "serial_after_ns": after,
        "snapshot_ns": 30,
    }


def _response(config: BridgeConfig) -> dict:
    return {
        "schema_version": "pcss.llama-bridge-profile/v1",
        "operation": "profile",
        "status": "ok",
        "backend": "llama.cpp",
        "measurement_clock": "std::chrono::steady_clock/nanoseconds/v1",
        "measurement_scope": "decode-and-synchronize/no-context-creation/v1",
        "claim_boundary": "isolated-perfect-draft-verifier-profile/not-live-pcss/v1",
        "seed_id": "seed-1",
        "warmup_repetitions": 1,
        "measurement_repetitions": 2,
        "prompt_state_sha256": "a" * 64,
        "prompt_state_bytes": 64,
        "profiles": [
            {
                "k": 2,
                "perfect_draft_tokens": [7, 8],
                "measurements": [_measurement(0), _measurement(1, before=204, after=206)],
            }
        ],
        "configuration": {
            "model_artifact": str(config.model_artifact),
            "llama_cpp_library": str(config.llama_cpp_library),
            "llama_cpp_revision": config.llama_cpp_revision,
            "runtime": dict(config.runtime),
        },
        "observed_context": {
            "actual_n_ctx": 128,
            "actual_n_batch": 32,
            "actual_n_threads": 4,
            "actual_n_threads_batch": 4,
        },
        "state_export": "model-architecture+memory-state/v2",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
    }


def test_profile_request_is_bounded_by_batch_and_repetitions(tmp_path: Path) -> None:
    config = _config(tmp_path)

    request = _request(
        config,
        seed_id="seed-1",
        context=(1,),
        k_values=(1, 2, 4),
        warmup_repetitions=1,
        measurement_repetitions=2,
    )

    assert request["operation"] == "profile"
    assert request["profile"]["k_values"] == [1, 2, 4]
    with pytest.raises(ContractError, match="n_batch"):
        _request(
            config,
            seed_id="seed-1",
            context=(1,),
            k_values=(64,),
            warmup_repetitions=1,
            measurement_repetitions=2,
        )


def test_profile_response_is_exact_and_analysis_uses_paired_iterations(tmp_path: Path) -> None:
    config = _config(tmp_path)
    profiles = _validate_response(
        _response(config),
        config=config,
        seed_id="seed-1",
        k_values=(2,),
        warmup_repetitions=1,
        measurement_repetitions=2,
    )

    analysis = _analyze(profiles, baseline_tolerance=0.05)

    assert analysis["decision"] == "PROFILE_VALID"
    assert analysis["rows"][0]["paired_decode_ratio_ci"]["paired_iteration_clusters"] == 2
    assert analysis["rows"][0]["paired_decode_ratio_ci"]["lower"] > 1.9


def test_pending_profile_accounts_for_anchor_and_requires_its_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = _request(
        config,
        seed_id="seed-1",
        context=(1,),
        k_values=(2,),
        warmup_repetitions=1,
        measurement_repetitions=2,
        commit_protocol="pending_native",
    )
    response = _response(config)
    response.update(
        {
            "schema_version": "pcss.llama-bridge-pending-profile/v1",
            "operation": "profile_pending",
            "measurement_scope": "pending-anchor-decode-and-synchronize/no-context-creation/v1",
            "claim_boundary": "isolated-exact-state-pending-token-profile/not-live-pcss/v1",
        }
    )

    assert request["operation"] == "profile_pending"
    profiles = _validate_response(
        response,
        config=config,
        seed_id="seed-1",
        k_values=(2,),
        warmup_repetitions=1,
        measurement_repetitions=2,
        commit_protocol="pending_native",
    )
    assert profiles[0]["k"] == 2

    constrained = _config(tmp_path)
    constrained.runtime["n_batch"] = 2
    with pytest.raises(ContractError, match="n_batch"):
        _request(
            constrained,
            seed_id="seed-1",
            context=(1,),
            k_values=(2,),
            warmup_repetitions=1,
            measurement_repetitions=2,
            commit_protocol="pending_native",
        )


def test_profile_rejects_nonchronological_or_nonpositive_measurements(tmp_path: Path) -> None:
    config = _config(tmp_path)
    response = _response(config)
    response["profiles"][0]["measurements"][1]["iteration"] = 3
    with pytest.raises(ContractError, match="chronological"):
        _validate_response(
            response,
            config=config,
            seed_id="seed-1",
            k_values=(2,),
            warmup_repetitions=1,
            measurement_repetitions=2,
        )

    response = _response(config)
    response["profiles"][0]["measurements"][0]["batch_verify_ns"] = 0
    with pytest.raises(ContractError):
        _validate_response(
            response,
            config=config,
            seed_id="seed-1",
            k_values=(2,),
            warmup_repetitions=1,
            measurement_repetitions=2,
        )


def test_profile_baseline_repeatability_gate_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    response = _response(config)
    for measurement in response["profiles"][0]["measurements"]:
        measurement["serial_after_ns"] = measurement["serial_before_ns"] * 2
    profiles = _validate_response(
        response,
        config=config,
        seed_id="seed-1",
        k_values=(2,),
        warmup_repetitions=1,
        measurement_repetitions=2,
    )

    analysis = _analyze(profiles, baseline_tolerance=0.05)

    assert analysis["decision"] == "BASELINE_REPEATABILITY_FAILED"
    assert not analysis["all_baselines_repeatable"]
