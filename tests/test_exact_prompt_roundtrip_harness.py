from __future__ import annotations

from pcss.exact_prompt_roundtrip_harness import _request, _validate_round
from pcss.llamacpp_fixture_builder import BridgeConfig
from pcss.provider import Proposal, TrustedScope
from pathlib import Path


def _config() -> BridgeConfig:
    return BridgeConfig(
        config_path=Path("/tmp/config"), bridge_executable=Path("/tmp/bridge"), model_artifact=Path("/tmp/model"),
        llama_cpp_library=Path("/tmp/library"), llama_cpp_revision="revision",
        runtime={"n_ctx": 128, "n_batch": 8, "n_gpu_layers": 0, "flash_attn": "disabled", "seed": 0, "n_threads": 1, "n_threads_batch": 1, "chat_template": "none", "thinking_mode": "disabled"},
        request_timeout_seconds=1, runtime_configuration_fingerprint="0" * 64,
    )


def test_roundtrip_request_and_validation_bind_committed_correction() -> None:
    target = {"record_id": "target"}
    request = _request(_config(), target, step=3, context_tokens=(1, 2), draft_tokens=(3, 4))
    assert request["operation"] == "verify_pending"
    assert request["seed"]["initial_context_tokens"] == [1, 2]
    scope = TrustedScope("p", "a", "r", "git", "edit", "block", None)
    proposal = Proposal("proposal", "request", (3, 4), "same_repository", scope, None, 4, 1.0, 1, "repository", 0)
    response = {
        "operation": "verify_pending", "status": "ok", "verification_targets_match": True,
        "pending_state_matches": True, "state_matches": True, "continuation_matches": True,
        "committed_tokens": [3, 4, 5], "reference_committed_tokens": [3, 4, 5],
        "accepted_draft_tokens": 2, "committed_state_sha256": "a" * 64, "committed_state_bytes": 1,
        "pending_state_sha256": "b" * 64, "pending_state_bytes": 1, "correction_token": None, "bonus_token": 5,
    }
    committed, accepted, state = _validate_round(response, proposal)
    assert committed == (3, 4, 5)
    assert accepted == 2
    assert state["bonus_token"] == 5
