from __future__ import annotations

import json
from pathlib import Path

import pytest

import pcss.verifier_preflight as preflight
from pcss.canonical import ContractError
from pcss.verifier import AdapterAvailability, AdapterOutcome, ScriptedFixtureAdapter, VerifierCase


ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def deterministic_receipt_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "resolve_git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(
        preflight,
        "recorded_module_command",
        lambda module, arguments, root: f"python -m {module}",
    )


class CountingUnavailableAdapter:
    name = "counting-unavailable"

    def __init__(self) -> None:
        self.verify_calls = 0

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("NOT_RUN", "backend absent")

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        self.verify_calls += 1
        raise AssertionError("preflight must never call verify")


class SelfAssertedTargetWithoutBinding:
    name = "self-asserted-target"

    def __init__(self) -> None:
        self.verify_calls = 0

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("TARGET_BACKEND", "self-asserted")

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        self.verify_calls += 1
        raise AssertionError("preflight must never call verify")


def make_receipt(adapter: object, spec: str = "tests:adapter"):
    return preflight.preflight_target_observation(
        adapter=adapter,  # type: ignore[arg-type]
        adapter_spec=spec,
        repository_root=ROOT,
        command_arguments=("--adapter", spec),
    )


def test_unavailable_adapter_emits_not_run_without_capture_or_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = CountingUnavailableAdapter()

    def capture_should_not_run(object: object) -> object:
        raise AssertionError("non-target adapter must not capture target facts")

    monkeypatch.setattr(preflight, "capture_target_observation", capture_should_not_run)
    receipt = make_receipt(adapter)
    assert receipt.evidence_class == "NOT_RUN"
    assert receipt.status == "NOT_RUN"
    assert receipt.payload["schema_version"] == "pcss.m4-target-observation/v1"
    assert receipt.payload["observation_status"] == "not_run"
    assert receipt.payload["host_identity"] is None
    assert receipt.payload["adapter_binding"] is None
    assert adapter.verify_calls == 0
    assert set(receipt.input_fingerprints) == {"adapter_spec"}


def test_offline_fixture_cannot_be_promoted_to_target_observation() -> None:
    receipt = make_receipt(ScriptedFixtureAdapter(), "pcss.verifier:ScriptedFixtureAdapter")
    assert receipt.payload["declared_execution_class"] == "OFFLINE_FIXTURE"
    assert receipt.payload["observation_status"] == "not_run"
    assert receipt.evidence_class == "NOT_RUN"


def test_self_asserted_target_without_binding_is_not_run_and_never_verifies() -> None:
    adapter = SelfAssertedTargetWithoutBinding()
    receipt = make_receipt(adapter)
    assert receipt.payload["declared_execution_class"] == "TARGET_BACKEND"
    assert receipt.payload["observation_status"] == "not_run"
    assert "target_binding" in receipt.payload["reason"]
    assert adapter.verify_calls == 0


def test_preflight_writer_is_immutable_and_preserves_not_run_payload(tmp_path: Path) -> None:
    receipt = make_receipt(CountingUnavailableAdapter())
    output = tmp_path / "target-observation.json"
    digest = preflight.write_target_observation(output, receipt)
    payload = json.loads(output.read_text())
    assert len(digest) == 64
    assert payload["schema_version"] == "pcss.result/v1"
    assert payload["evidence_class"] == "NOT_RUN"
    assert payload["payload"]["schema_version"] == "pcss.m4-target-observation/v1"
    with pytest.raises(ContractError, match="already exists"):
        preflight.write_target_observation(output, receipt)
