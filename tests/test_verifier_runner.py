from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.verifier import AdapterAvailability, ScriptedFixtureAdapter, VerifierCase, fixture_state_digest
from pcss.verifier_runner import execute_suite, load_cases


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
