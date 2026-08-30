from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcss.canonical import ContractError
from pcss.llamacpp_trajectory import _request, _validate_response, load_plan


def _plan(tmp_path: Path) -> tuple[Path, dict]:
    payload = {
        "schema_version": "pcss.pending-trajectory-plan/v1",
        "case_id": "mixed-k2",
        "k": 2,
        "rounds": [
            {"mode": "full_accept", "rejection_position": -1},
            {"mode": "reject", "rejection_position": 0},
            {"mode": "reject", "rejection_position": 1},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _config(tmp_path: Path):
    return SimpleNamespace(
        model_artifact=tmp_path / "model.gguf",
        llama_cpp_library=tmp_path / "libllama.dylib",
        llama_cpp_revision="test-revision",
        runtime={
            "n_ctx": 64,
            "n_batch": 8,
            "n_gpu_layers": 99,
            "flash_attn": "auto",
            "seed": 0,
            "n_threads": 2,
            "n_threads_batch": 2,
            "chat_template": "caller-pretokenized/v1",
            "thinking_mode": "disabled",
        },
    )


def _response(config, plan: dict) -> dict:
    rounds = []
    for index, expected in enumerate(plan["rounds"]):
        digest = f"{index + 1:064x}"
        rounds.append(
            {
                "round": index,
                "mode": expected["mode"],
                "rejection_position": expected["rejection_position"],
                "accepted_draft_tokens": (
                    plan["k"] if expected["mode"] == "full_accept" else expected["rejection_position"]
                ),
                "pending_token": 100 + index,
                "tail_trimmed": True,
                "state_matches": True,
                "verifier_state_sha256": digest,
                "reference_state_sha256": digest,
                "state_bytes": 4096,
            }
        )
    return {
        "schema_version": "pcss.llama-bridge-pending-trajectory/v1",
        "operation": "trajectory_pending",
        "status": "ok",
        "backend": "llama.cpp",
        "case_id": plan["case_id"],
        "seed_id": "seed",
        "k": plan["k"],
        "rounds_requested": len(rounds),
        "rounds_completed": len(rounds),
        "rounds": rounds,
        "all_rounds_match": True,
        "final_state_matches": True,
        "continuation_matches": True,
        "final_next_token": 200,
        "state_export": "model-architecture+memory-state/v2",
        "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
        "configuration": {
            "model_artifact": str(config.model_artifact),
            "llama_cpp_library": str(config.llama_cpp_library),
            "llama_cpp_revision": config.llama_cpp_revision,
            "runtime": config.runtime,
        },
        "observed_context": {
            "actual_n_ctx": 64,
            "actual_n_batch": 8,
            "actual_n_threads": 2,
            "actual_n_threads_batch": 2,
        },
        "claim_boundary": "exact-state correctness trajectory only; no performance measurement",
    }


def test_plan_and_request_preserve_the_explicit_mixed_schedule(tmp_path: Path) -> None:
    path, _ = _plan(tmp_path)
    plan = load_plan(path)
    request = _request(_config(tmp_path), seed_id="seed", context=(1, 2), plan=plan)

    assert request["operation"] == "trajectory_pending"
    assert request["trajectory"]["rounds"] == plan["rounds"]


def test_plan_rejects_out_of_range_rejection_position(tmp_path: Path) -> None:
    path, payload = _plan(tmp_path)
    payload["rounds"][1]["rejection_position"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="mode/rejection_position"):
        load_plan(path)


def test_response_validation_fails_closed_on_any_round_state_mismatch(tmp_path: Path) -> None:
    path, _ = _plan(tmp_path)
    plan = load_plan(path)
    config = _config(tmp_path)
    response = _response(config, plan)
    assert _validate_response(response, config=config, seed_id="seed", plan=plan)

    response["rounds"][1]["state_matches"] = False
    with pytest.raises(ContractError, match="state-match claim"):
        _validate_response(response, config=config, seed_id="seed", plan=plan)
