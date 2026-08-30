from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss import verifier_runner
from pcss.canonical import ContractError
from pcss.verifier import (
    AdapterAvailability,
    AdapterOutcome,
    ScriptedFixtureAdapter,
    TargetExecutionPermit,
    TargetHostIdentity,
    VerifierCase,
    fixture_state_digest,
)
from pcss.verifier_runner import _authorize_target_backend, _require_fixture_target_compatibility, execute_suite, load_cases


def unfilled_manifest() -> dict:
    return {
        "schema_version": "pcss.manifest/v1",
        "run_id": "fixture-run",
        "evidence_class": "NOT_RUN",
        "target": {"hardware": {"observation_status": "not_observed", "value": None}, "operating_system": {"observation_status": "not_observed", "value": None}},
        "model_artifact": {"observation_status": "not_observed", "artifact_uri": None, "sha256": None},
        **{field: {"observation_status": "not_observed", "fingerprint": None, "parameters": None} for field in ("quantization", "tokenizer", "chat_template", "generation_protocol", "llama_cpp", "context_kv", "benchmark")},
        "verifier_gate": {
            "observation_status": "not_observed",
            "host_identity": None,
            "adapter_binding": None,
        },
    }


def fixture_payload() -> dict:
    committed = (1, 2)
    return {
        "schema_version": "pcss.m4-verifier-cases/v1",
        "cases": [{
            "case_id": "fixture-case",
            "draft_tokens": [1, 9],
            "target_tokens": [1, 2],
            "expected_committed_tokens": [1, 2],
            "expected_state_digest": fixture_state_digest(committed),
            "k": 2,
            "include_bonus_on_full_accept": False,
        }],
    }


def fixture_payload_v3() -> dict:
    return {
        "schema_version": "pcss.m4-verifier-cases/v3",
        "provenance": {
            "schema_version": "pcss.target-verifier-fixture-provenance/v1",
            "reference_method": "serial-greedy-full-state/v1",
            "model_artifact_sha256": "a" * 64,
            "runtime_configuration_fingerprint": "b" * 64,
        },
        "cases": [{
            "case_id": "context-case",
            "seed_id": "raw-seed-alpha",
            "initial_context_tokens": [10, 11],
            "draft_tokens": [12],
            "target_tokens": [12],
            "expected_committed_tokens": [12],
            "expected_state_digest": "state",
            "k": 1,
            "include_bonus_on_full_accept": False,
        }],
    }


def test_runner_emits_offline_fixture_receipt_without_hardware_claim(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(fixture_payload()))
    manifest.write_text(json.dumps(unfilled_manifest()))
    receipt = execute_suite(
        adapter=ScriptedFixtureAdapter(),
        suite="forced-rejection",
        fixture_path=fixture,
        manifest_path=manifest,
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--suite", "forced-rejection"),
    )
    assert receipt.evidence_class == "OFFLINE_FIXTURE"
    assert receipt.status == "OFFLINE_FIXTURE"


def test_fixture_loader_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "pcss.m4-verifier-cases/v1", "cases": []}))
    with pytest.raises(ContractError, match="at least one case"):
        load_cases(path)


def test_v3_fixture_binds_context_seed_and_reference_runtime(tmp_path: Path) -> None:
    path = tmp_path / "v3.json"
    path.write_text(json.dumps(fixture_payload_v3()))
    cases = load_cases(path)
    assert cases[0].initial_context_tokens == (10, 11)
    assert cases[0].seed_id == "raw-seed-alpha"
    assert cases[0].reference_provenance is not None
    _require_fixture_target_compatibility(
        cases,
        {
            "model_artifact_sha256": "a" * 64,
            "runtime_configuration_fingerprint": "b" * 64,
        },
    )


def test_v3_fixture_rejects_empty_context_or_different_runtime(tmp_path: Path) -> None:
    payload = fixture_payload_v3()
    payload["cases"][0]["initial_context_tokens"] = []
    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps(payload))
    with pytest.raises(ContractError, match="non-empty initial_context_tokens"):
        load_cases(empty_path)

    path = tmp_path / "v3.json"
    path.write_text(json.dumps(fixture_payload_v3()))
    with pytest.raises(ContractError, match="runtime configuration"):
        _require_fixture_target_compatibility(
            load_cases(path),
            {
                "model_artifact_sha256": "a" * 64,
                "runtime_configuration_fingerprint": "c" * 64,
            },
        )


def test_v2_target_fixture_without_explicit_seed_is_not_target_compatible(tmp_path: Path) -> None:
    payload = fixture_payload_v3()
    payload["schema_version"] = "pcss.m4-verifier-cases/v2"
    del payload["cases"][0]["seed_id"]
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ContractError, match="explicit seed_id"):
        _require_fixture_target_compatibility(
            load_cases(path),
            {
                "model_artifact_sha256": "a" * 64,
                "runtime_configuration_fingerprint": "b" * 64,
            },
        )


def _target_sysctl_value(name: str) -> str:
    return {
        "hw.memsize": str(24 * 1024**3),
        "hw.model": "Mac16,8",
        "machdep.cpu.brand_string": "Apple M4 Pro",
    }[name]


def test_target_host_accepts_apple_model_identifier_when_product_class_is_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verifier_runner, "_sysctl_value", _target_sysctl_value)
    monkeypatch.setattr(verifier_runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(verifier_runner.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        verifier_runner,
        "_system_profiler_hardware_profile",
        lambda: {"machine_model": "Mac16,8", "machine_name": "MacBook Pro"},
    )

    identity = verifier_runner.observe_target_host()

    assert identity.hardware_model == "Mac16,8"
    assert identity.chip == "Apple M4 Pro"


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"machine_model": "Mac16,8", "machine_name": "Mac mini"}, "MacBook Pro hardware class"),
        ({"machine_model": "Mac16,7", "machine_name": "MacBook Pro"}, "model identifiers must agree"),
    ],
)
def test_target_host_rejects_unverified_product_class_or_inconsistent_identifier(
    monkeypatch: pytest.MonkeyPatch,
    profile: dict[str, str],
    message: str,
) -> None:
    monkeypatch.setattr(verifier_runner, "_sysctl_value", _target_sysctl_value)
    monkeypatch.setattr(verifier_runner.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(verifier_runner.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(verifier_runner, "_system_profiler_hardware_profile", lambda: profile)

    with pytest.raises(ContractError, match=message):
        verifier_runner.observe_target_host()


class NamedAdapter:
    name = "target-adapter"


def _target_authorization_inputs() -> tuple[TargetHostIdentity, dict[str, str], dict]:
    host = TargetHostIdentity.from_observed(
        target_profile="contract-test",
        operating_system="Darwin",
        architecture="arm64",
        hardware_model="Mac16,8",
        chip="Apple M4 Pro",
        memory_bytes=24 * 1024**3,
    )
    binding = {
        "schema_version": "pcss.verifier-adapter-binding/v2",
        "adapter_source_sha256": "a" * 64,
        "backend_binary_sha256": "b" * 64,
        "model_artifact_sha256": "c" * 64,
        "llama_cpp_build_fingerprint": "d" * 64,
        "runtime_configuration_fingerprint": "e" * 64,
    }
    manifest = {
        "verifier_gate": {"host_identity": host.as_dict(), "adapter_binding": binding},
        "model_artifact": {"sha256": binding["model_artifact_sha256"]},
        "llama_cpp": {
            "fingerprint": "f" * 64,
            "parameters": {"llama_cpp_build_fingerprint": binding["llama_cpp_build_fingerprint"]},
        },
    }
    return host, binding, manifest


def test_target_authorization_compares_the_explicit_llama_cpp_build_field() -> None:
    host, binding, manifest = _target_authorization_inputs()

    permit = _authorize_target_backend(NamedAdapter(), manifest, host, binding)

    assert permit.adapter_name == "target-adapter"


def test_target_authorization_rejects_a_different_explicit_llama_cpp_build_field() -> None:
    host, binding, manifest = _target_authorization_inputs()
    manifest["llama_cpp"]["parameters"]["llama_cpp_build_fingerprint"] = "0" * 64

    with pytest.raises(ContractError, match="llama.cpp build fingerprint"):
        _authorize_target_backend(NamedAdapter(), manifest, host, binding)


class AuthorizedFailingAdapter:
    name = "authorized-failing-target"

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("TARGET_BACKEND", "prepared target adapter")

    def verify(self, case: VerifierCase):
        raise ContractError("native bridge execution failure")


class AuthorizedMismatchAdapter:
    name = "authorized-mismatch-target"

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("TARGET_BACKEND", "native mismatch receipt")

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        return AdapterOutcome(
            accepted_draft_tokens=1,
            correction_token=2,
            bonus_token=None,
            committed_tokens=case.expected_committed_tokens,
            mutable_state_digest=case.expected_state_digest,
            reconstruction_matches=False,
        )


def _allow_target_execution(
    monkeypatch: pytest.MonkeyPatch, adapter_name: str
) -> None:
    host, binding, _ = _target_authorization_inputs()
    permit = TargetExecutionPermit(
        adapter_name=adapter_name,
        host_fingerprint=host.fingerprint,
        adapter_binding_fingerprint="f" * 64,
    )
    monkeypatch.setattr(verifier_runner, "_require_observed_target_manifest", lambda value: None)
    monkeypatch.setattr(verifier_runner, "capture_target_observation", lambda adapter: (host, binding))
    monkeypatch.setattr(verifier_runner, "_require_fixture_target_compatibility", lambda cases, observed: None)
    monkeypatch.setattr(verifier_runner, "_authorize_target_backend", lambda *args: permit)
    monkeypatch.setattr(verifier_runner, "resolve_git_revision", lambda root: "f" * 40)
    monkeypatch.setattr(verifier_runner, "recorded_module_command", lambda module, arguments, root: "test-command")


def test_runner_marks_unstructured_target_execution_error_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(fixture_payload()))
    manifest.write_text(json.dumps(unfilled_manifest()))
    _allow_target_execution(monkeypatch, AuthorizedFailingAdapter.name)

    receipt = execute_suite(
        adapter=AuthorizedFailingAdapter(),
        suite="forced-rejection",
        fixture_path=fixture,
        manifest_path=manifest,
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--suite", "forced-rejection"),
    )

    assert receipt.evidence_class == "NOT_RUN"
    assert receipt.status == "NOT_RUN"
    assert receipt.payload["metadata"]["target_execution"]["status"] == "incomplete"
    assert "native bridge execution failure" in receipt.payload["reason"]


def test_runner_records_valid_target_mismatch_as_measured_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(fixture_payload()))
    manifest.write_text(json.dumps(unfilled_manifest()))
    _allow_target_execution(monkeypatch, AuthorizedMismatchAdapter.name)

    receipt = execute_suite(
        adapter=AuthorizedMismatchAdapter(),
        suite="forced-rejection",
        fixture_path=fixture,
        manifest_path=manifest,
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--suite", "forced-rejection"),
    )

    assert receipt.evidence_class == "LOCAL_MEASURED"
    assert receipt.status == "FAILED"
    assert receipt.payload["cases"][0]["failures"] == ["serial_batched_state_mismatch"]


class SelfDeclaredTargetAdapter:
    name = "self-declared-target"

    def __init__(self) -> None:
        self.availability_calls = 0
        self.verify_calls = 0

    def availability(self) -> AdapterAvailability:
        self.availability_calls += 1
        return AdapterAvailability("TARGET_BACKEND", "untrusted self-declaration")

    def verify(self, case: VerifierCase):
        self.verify_calls += 1
        raise AssertionError("target verification must not begin without runner authorization")


def test_target_self_declaration_is_not_run_before_host_or_verification(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(fixture_payload()))
    manifest.write_text(json.dumps(unfilled_manifest()))
    adapter = SelfDeclaredTargetAdapter()

    receipt = execute_suite(
        adapter=adapter,
        suite="forced-rejection",
        fixture_path=fixture,
        manifest_path=manifest,
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--suite", "forced-rejection"),
    )

    assert receipt.evidence_class == "NOT_RUN"
    assert receipt.status == "NOT_RUN"
    assert adapter.availability_calls == 1
    assert adapter.verify_calls == 0
    assert receipt.payload["metadata"]["target_authorization"]["status"] == "rejected"


class FlippingFixtureAdapter(ScriptedFixtureAdapter):
    def __init__(self) -> None:
        self.availability_calls = 0

    def availability(self) -> AdapterAvailability:
        self.availability_calls += 1
        if self.availability_calls == 1:
            return AdapterAvailability("OFFLINE_FIXTURE", "initial deterministic fixture")
        return AdapterAvailability("TARGET_BACKEND", "unexpected later claim")


def test_runner_freezes_execution_class_after_initial_availability(tmp_path: Path) -> None:
    fixture = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    fixture.write_text(json.dumps(fixture_payload()))
    manifest.write_text(json.dumps(unfilled_manifest()))
    adapter = FlippingFixtureAdapter()

    receipt = execute_suite(
        adapter=adapter,
        suite="forced-rejection",
        fixture_path=fixture,
        manifest_path=manifest,
        repository_root=Path(__file__).resolve().parents[1],
        command_arguments=("--suite", "forced-rejection"),
    )

    assert receipt.evidence_class == "OFFLINE_FIXTURE"
    assert adapter.availability_calls == 1
