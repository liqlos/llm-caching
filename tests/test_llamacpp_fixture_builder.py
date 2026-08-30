from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError, canonical_json
from pcss.llamacpp_fixture_builder import (
    CONFIG_SCHEMA_VERSION,
    FIXTURE_SCHEMA_VERSION,
    SEEDS_SCHEMA_VERSION,
    build_target_fixtures,
    load_bridge_config,
)
from pcss.verifier import TargetHostIdentity
from pcss.verifier_runner import load_cases


def _target_identity() -> TargetHostIdentity:
    return TargetHostIdentity.from_observed(
        target_profile="macbook_pro_m4_24gb",
        operating_system="Darwin",
        architecture="arm64",
        hardware_model="MacBookPro16,8",
        chip="Apple M4 Pro",
        memory_bytes=24 * 1024**3,
    )


def _write_bridge(path: Path, *, mode: str = "ok") -> None:
    source = f'''#!/usr/bin/env python3
import hashlib
import json
import sys

MODE = {mode!r}
request = json.loads(sys.stdin.read())
context = request["seed"]["initial_context_tokens"]
steps = request["seed"]["steps"]
vocabulary_size = 29 if MODE != "one-token-vocabulary" else 1
prefill_offset = len(context) if MODE == "prefill-sensitive" else 0
target_tokens = [(context[-1] + prefill_offset + index + 1) % vocabulary_size for index in range(steps)]
state = hashlib.sha256(json.dumps(
    {{"context": context, "target_tokens": target_tokens}}, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
pending = request["operation"] == "reference_pending"
response = {{
    "schema_version": (
        "pcss.llama-bridge-pending-reference/v1" if pending else "pcss.llama-bridge-reference/v2"
    ),
    "operation": request["operation"],
    "status": "ok",
    "backend": "llama.cpp",
    "generation_mode": "greedy_argmax",
    "greedy_tie_break": "lowest-token-id/v1",
    "seed_id": request["seed"]["seed_id"],
    "target_tokens": target_tokens,
    "vocabulary_size": vocabulary_size,
    "state_export": "model-architecture+memory-state/v2",
    "mutable_state_digest": state,
    "completion_state_sha256": state,
    "completion_state_bytes": 64,
    "prompt_state_sha256": "a" * 64,
    "prompt_state_bytes": 32,
    "configuration": {{
        "model_artifact": request["model_artifact"],
        "llama_cpp_library": request["llama_cpp_library"],
        "llama_cpp_revision": request["llama_cpp_revision"],
        "runtime": request["runtime"],
    }},
    "observed_context": {{
        "actual_n_ctx": request["runtime"]["n_ctx"],
        "actual_n_batch": request["runtime"]["n_batch"],
        "actual_n_threads": request["runtime"]["n_threads"],
        "actual_n_threads_batch": request["runtime"]["n_threads_batch"],
    }},
    "tokenization_boundary": "caller-supplied-raw-token-ids/v1",
    "state_api": "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2",
    "sampler_rng_state": "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1",
    "declared_identity_fields": "caller-declared-not-verified-by-bridge/v1",
    "chat_template_applied": False,
    "thinking_mode_applied": False,
}}
if MODE == "malformed":
    del response["vocabulary_size"]
elif MODE == "wrong-model":
    response["configuration"]["model_artifact"] = "/different/model.gguf"
elif MODE == "wrong-length":
    response["target_tokens"] = target_tokens[:-1]
print(json.dumps(response, sort_keys=True, separators=(",", ":")))
'''
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _config(tmp_path: Path, bridge: Path) -> Path:
    model = tmp_path / "model.gguf"
    library = tmp_path / "libllama.dylib"
    model.write_bytes(b"target model bytes")
    library.write_bytes(b"target llama.cpp library bytes")
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "bridge_protocol": "pcss.llama-bridge/v1",
        "bridge_executable": str(bridge),
        "model_artifact": str(model),
        "llama_cpp_library": str(library),
        "llama_cpp_revision": "test-llama-revision",
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
    path.write_text(canonical_json(config), encoding="utf-8")
    return path


def _seeds(tmp_path: Path) -> Path:
    payload = {
        "schema_version": SEEDS_SCHEMA_VERSION,
        "k_values": [2, 3],
        "provider_floor_k": 2,
        "forced_rejection_k": 3,
        "long_mixed": {"context_steps": 4, "k": 3, "rejection_position": 1},
        "seeds": [{"seed_id": "alpha", "initial_context_tokens": [4, 8, 15]}],
    }
    path = tmp_path / "seeds.json"
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def _build(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    bridge_mode: str = "ok",
    commit_protocol: str = "serial",
):
    bridge = tmp_path / "fake-bridge"
    _write_bridge(bridge, mode=bridge_mode)
    config = _config(tmp_path, bridge)
    seeds = _seeds(tmp_path)
    monkeypatch.setattr("pcss.llamacpp_fixture_builder.resolve_git_revision", lambda root: "f" * 40)
    monkeypatch.setattr(
        "pcss.llamacpp_fixture_builder.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.llamacpp_fixture_builder",
    )
    fixture_directory = tmp_path / "fixtures"
    raw_result = tmp_path / "fixture-build.raw.json"
    report = tmp_path / "fixture-build.report.md"
    receipt = build_target_fixtures(
        config_path=config,
        seeds_path=seeds,
        fixture_directory=fixture_directory,
        raw_result_path=raw_result,
        report_path=report,
        repository_root=Path.cwd(),
        command_arguments=("--config", str(config)),
        commit_protocol=commit_protocol,
        target_observer=_target_identity,
    )
    return receipt, config, seeds, fixture_directory, raw_result, report


def test_pending_reference_protocol_is_explicit_in_fixture_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, _, _, fixture_directory, _, _ = _build(
        tmp_path,
        monkeypatch=monkeypatch,
        commit_protocol="pending_native",
    )

    assert receipt.payload["claim_boundary"]["fixture_source"] == (
        "pending-token-serial-greedy-memory-state/v1"
    )
    fixture = json.loads((fixture_directory / "long-mixed.v4.json").read_text(encoding="utf-8"))
    assert fixture["provenance"]["reference_method"] == "pending-token-serial-greedy-memory-state/v1"


def test_builds_all_v4_fixture_suites_from_real_bridge_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt, config_path, _, fixture_directory, raw_result, report = _build(tmp_path, monkeypatch=monkeypatch)
    config = load_bridge_config(config_path)
    model_hash = hashlib.sha256((tmp_path / "model.gguf").read_bytes()).hexdigest()

    assert receipt.evidence_class == "LOCAL_MEASURED"
    assert receipt.status == "FIXTURES_BUILT"
    assert raw_result.is_file()
    assert report.is_file()
    assert "Raw result" in report.read_text(encoding="utf-8")
    raw_payload = json.loads(raw_result.read_text(encoding="utf-8"))
    assert raw_payload["payload"]["claim_boundary"]["not_claimed"] == [
        "hardware_performance",
        "timing",
        "provider_economics",
        "live_speculative_decode",
    ]

    suite_cases = {}
    for suite, filename in {
        "k_sweep": "k-sweep.v4.json",
        "perfect_draft": "perfect-draft.v4.json",
        "provider_floor": "provider-floor.v4.json",
        "forced_rejection": "forced-rejection.v4.json",
        "long_mixed": "long-mixed.v4.json",
    }.items():
        fixture = fixture_directory / filename
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        assert payload["schema_version"] == FIXTURE_SCHEMA_VERSION
        assert payload["provenance"] == {
            "schema_version": "pcss.target-verifier-fixture-provenance/v2",
            "reference_method": "serial-greedy-memory-state/v2",
            "model_artifact_sha256": model_hash,
            "runtime_configuration_fingerprint": config.runtime_configuration_fingerprint,
        }
        suite_cases[suite] = load_cases(fixture)

    assert all(case.seed_id == "alpha" for case in suite_cases["k_sweep"])
    assert all(case.seed_id == "alpha" for case in suite_cases["perfect_draft"])
    assert suite_cases["provider_floor"][0].seed_id == "alpha"
    assert all(case.seed_id == "alpha" for case in suite_cases["forced_rejection"])

    assert [case.k for case in suite_cases["k_sweep"]] == [2, 3]
    perfect = suite_cases["perfect_draft"]
    assert all(case.include_bonus_on_full_accept for case in perfect)
    assert all(case.draft_tokens == case.target_tokens[: case.k] for case in perfect)
    assert all(len(case.expected_committed_tokens) == case.k + 1 for case in perfect)

    provider = suite_cases["provider_floor"]
    assert len(provider) == 1
    assert provider[0].draft_tokens[0] != provider[0].target_tokens[0]
    assert provider[0].expected_committed_tokens == provider[0].target_tokens[:1]

    forced = suite_cases["forced_rejection"]
    assert len(forced) == 3
    for position, case in enumerate(forced):
        assert case.draft_tokens[:position] == case.target_tokens[:position]
        assert case.draft_tokens[position] != case.target_tokens[position]
        assert case.expected_committed_tokens == case.target_tokens[: position + 1]

    mixed = suite_cases["long_mixed"]
    assert len(mixed) == 1
    assert len(mixed[0].initial_context_tokens) == 7
    assert mixed[0].seed_id == "alpha"
    assert mixed[0].draft_tokens[0] == mixed[0].target_tokens[0]
    assert mixed[0].draft_tokens[1] != mixed[0].target_tokens[1]


def test_long_mixed_reanchors_to_the_reconstructed_prefill_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, fixture_directory, _, _ = _build(tmp_path, monkeypatch=monkeypatch, bridge_mode="prefill-sensitive")

    mixed = load_cases(fixture_directory / "long-mixed.v4.json")

    assert len(mixed) == 1
    assert mixed[0].seed_id == "alpha"
    assert mixed[0].initial_context_tokens == (4, 8, 15, 19, 20, 21, 22)
    assert mixed[0].target_tokens == (1, 2, 3)


@pytest.mark.parametrize("bridge_mode", ["malformed", "wrong-model", "wrong-length"])
def test_malformed_or_mismatched_bridge_reference_fails_closed_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bridge_mode: str
) -> None:
    bridge = tmp_path / "fake-bridge"
    _write_bridge(bridge, mode=bridge_mode)
    config = _config(tmp_path, bridge)
    seeds = _seeds(tmp_path)
    monkeypatch.setattr("pcss.llamacpp_fixture_builder.resolve_git_revision", lambda root: "f" * 40)
    monkeypatch.setattr(
        "pcss.llamacpp_fixture_builder.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.llamacpp_fixture_builder",
    )
    fixture_directory = tmp_path / "fixtures"
    raw_result = tmp_path / "fixture-build.raw.json"
    report = tmp_path / "fixture-build.report.md"

    with pytest.raises(ContractError):
        build_target_fixtures(
            config_path=config,
            seeds_path=seeds,
            fixture_directory=fixture_directory,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=Path.cwd(),
            command_arguments=(),
            target_observer=_target_identity,
        )

    assert not fixture_directory.exists()
    assert not raw_result.exists()
    assert not report.exists()


def test_one_token_vocabulary_cannot_fabricate_forced_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = tmp_path / "fake-bridge"
    _write_bridge(bridge, mode="one-token-vocabulary")
    config = _config(tmp_path, bridge)
    seeds = _seeds(tmp_path)
    monkeypatch.setattr("pcss.llamacpp_fixture_builder.resolve_git_revision", lambda root: "f" * 40)
    monkeypatch.setattr(
        "pcss.llamacpp_fixture_builder.recorded_module_command",
        lambda module, arguments, root: "PYTHONPATH=/fixture python -m pcss.llamacpp_fixture_builder",
    )
    fixture_directory = tmp_path / "fixtures"
    raw_result = tmp_path / "fixture-build.raw.json"
    report = tmp_path / "fixture-build.report.md"

    with pytest.raises(ContractError, match="distinct vocabulary-valid"):
        build_target_fixtures(
            config_path=config,
            seeds_path=seeds,
            fixture_directory=fixture_directory,
            raw_result_path=raw_result,
            report_path=report,
            repository_root=Path.cwd(),
            command_arguments=(),
            target_observer=_target_identity,
        )

    assert not fixture_directory.exists()
    assert not raw_result.exists()
    assert not report.exists()


def test_seed_contexts_have_no_caller_supplied_target_outputs(tmp_path: Path) -> None:
    bridge = tmp_path / "fake-bridge"
    _write_bridge(bridge)
    seeds = _seeds(tmp_path)
    payload = json.loads(seeds.read_text(encoding="utf-8"))
    payload["seeds"][0]["target_tokens"] = [1, 2]
    seeds.write_text(canonical_json(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="invalid keys"):
        build_target_fixtures(
            config_path=_config(tmp_path, bridge),
            seeds_path=seeds,
            fixture_directory=tmp_path / "fixtures",
            raw_result_path=tmp_path / "result.json",
            report_path=tmp_path / "report.md",
            repository_root=Path.cwd(),
            command_arguments=(),
            target_observer=_target_identity,
        )
