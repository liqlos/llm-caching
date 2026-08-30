from pathlib import Path

from pcss.exact_prompt_external_trajectory import _request, _validate
from pcss.llamacpp_fixture_builder import BridgeConfig


def _config() -> BridgeConfig:
    return BridgeConfig(
        config_path=Path("/tmp/config"), bridge_executable=Path("/tmp/bridge"), model_artifact=Path("/tmp/model"),
        llama_cpp_library=Path("/tmp/library"), llama_cpp_revision="revision",
        runtime={"n_ctx": 128, "n_batch": 8, "n_gpu_layers": 0, "flash_attn": "disabled", "seed": 0, "n_threads": 1, "n_threads_batch": 1, "chat_template": "none", "thinking_mode": "disabled"},
        request_timeout_seconds=1, runtime_configuration_fingerprint="0" * 64,
    )


def test_external_trajectory_binds_archive_drafts_and_native_bonuses() -> None:
    source = {"content": {"committed_token_ids": [3, 4, 5]}}
    target = {"record_id": "target", "content": {"prompt_token_ids": [1, 2], "committed_token_ids": [3, 4, 5]}}
    request = _request(_config(), target, ((3, 4),), k=2)
    assert request["trajectory"]["rounds"] == [{"draft_tokens": [3, 4]}]
    response = {
        "operation": "trajectory_pending", "status": "ok", "all_rounds_match": True,
        "final_state_matches": True, "continuation_matches": True,
        "rounds": [{"mode": "external_draft", "verification_targets_match": True, "state_matches": True,
                    "accepted_draft_tokens": 2, "pending_token": 5, "verifier_state_sha256": "a" * 64, "state_bytes": 1}],
    }
    verified = _validate(response, source, target, ((3, 4),), k=2, output_tokens=3)
    assert verified["exact_output"] is True
    assert verified["output_tokens"] == [3, 4, 5]
