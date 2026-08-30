from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.llamacpp_fixture_builder import BridgeConfig
from pcss.llamacpp_live_benchmark import _pair, _request, _validate_response


def record(record_id: str, *, chronological_index: int, session_id: str) -> dict:
    return {
        "record_id": record_id,
        "chronological_index": chronological_index,
        "request_id": f"request-{record_id}",
        "session_id": session_id,
        "trusted_scope": {"repository_id": "repo"},
        "model_fingerprint": "model",
        "tokenizer_fingerprint": "tokenizer",
        "chat_template_fingerprint": "template",
        "generation_protocol_fingerprint": "protocol",
        "content": {
            "prompt_token_ids": [10, 11],
            "committed_token_ids": [20, 21, 22, 23],
            "committed_token_status": "complete",
        },
    }


def config() -> BridgeConfig:
    return BridgeConfig(
        config_path=Path("/tmp/config.json"),
        bridge_executable=Path("/tmp/bridge"),
        model_artifact=Path("/tmp/model.gguf"),
        llama_cpp_library=Path("/tmp/libllama.dylib"),
        llama_cpp_revision="revision",
        runtime={
            "n_ctx": 128, "n_batch": 8, "n_gpu_layers": 0, "flash_attn": "disabled", "seed": 0,
            "n_threads": 1, "n_threads_batch": 1, "chat_template": "none", "thinking_mode": "disabled",
        },
        request_timeout_seconds=30,
        runtime_configuration_fingerprint="0" * 64,
    )


def response_for(request: dict) -> dict:
    benchmark = request["benchmark"]
    runtime = request["runtime"]
    measurements = [
        {
            "iteration": index,
            "serial_before_ns": 120,
            "candidate_decode_ns": 100,
            "serial_after_ns": 122,
            "proposed_tokens": 3,
            "accepted_tokens": 3,
            "verification_rounds": 2,
            "serial_fallback_steps": 1,
        }
        for index in range(benchmark["measurement_repetitions"])
    ]
    return {
        "schema_version": "pcss.llama-bridge-pending-benchmark/v1",
        "operation": "benchmark_pending",
        "status": "ok",
        "backend": "llama.cpp",
        "case_id": benchmark["case_id"],
        "seed_id": request["seed"]["seed_id"],
        "k": benchmark["k"],
        "output_tokens": benchmark["output_tokens"],
        "warmup_repetitions": benchmark["warmup_repetitions"],
        "measurement_repetitions": benchmark["measurement_repetitions"],
        "authoritative_tokens": benchmark["candidate_tokens"][: benchmark["output_tokens"]],
        "candidate_prefix_tokens": benchmark["candidate_tokens"][: benchmark["output_tokens"]],
        "final_state_sha256": "a" * 64,
        "final_state_bytes": 100,
        "measurements": measurements,
        "measurement_clock": "std::chrono::steady_clock/nanoseconds/v1",
        "measurement_scope": "decode-and-synchronize/prefill-restore-provider-excluded/v1",
        "provider_integration": "static-trace-candidate/provider-lookup-excluded/v1",
        "exact_output_and_state": True,
        "configuration": runtime,
        "observed_context": {
            "actual_n_ctx": runtime["n_ctx"], "actual_n_batch": runtime["n_batch"],
            "actual_n_threads": runtime["n_threads"], "actual_n_threads_batch": runtime["n_threads_batch"],
        },
        "state_export": "model-architecture+memory-state/v2",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "generation_mode": "greedy_argmax",
        "greedy_tie_break": "lowest-token-id/v1",
        "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
        "chat_template_applied": False,
        "thinking_mode_applied": False,
        "claim_boundary": "controlled-static-candidate-live-decode/not-general-pcss/v1",
    }


def test_pair_requires_chronological_cross_session_exact_repeat() -> None:
    source = record("source", chronological_index=1, session_id="one")
    target = record("target", chronological_index=2, session_id="two")
    assert _pair([source, target], "source:target") == (source, target)

    target["content"]["committed_token_ids"][-1] = 99
    with pytest.raises(ContractError, match="exact committed-token repeat"):
        _pair([source, target], "source:target")


def test_response_contract_accepts_only_exact_target_output_and_state() -> None:
    source = record("source", chronological_index=1, session_id="one")
    target = record("target", chronological_index=2, session_id="two")
    request = _request(
        config(), source, target, k=2, output_tokens=4, warmup_repetitions=1, measurement_repetitions=2
    )
    response = response_for(request)
    assert len(_validate_response(response, request, config())["measurements"]) == 2

    response["authoritative_tokens"][-1] = 99
    with pytest.raises(ContractError, match="target output differs"):
        _validate_response(response, request, config())


def test_request_enforces_pending_batch_and_candidate_bounds() -> None:
    source = record("source", chronological_index=1, session_id="one")
    target = record("target", chronological_index=2, session_id="two")
    with pytest.raises(ContractError, match="batch boundary"):
        _request(config(), source, target, k=8, output_tokens=4, warmup_repetitions=0, measurement_repetitions=2)
    with pytest.raises(ContractError, match="candidate length"):
        _request(config(), source, target, k=2, output_tokens=5, warmup_repetitions=0, measurement_repetitions=2)
