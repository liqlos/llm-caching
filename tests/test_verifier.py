from __future__ import annotations

import json
from pathlib import Path

from pcss.verifier import (
    AdapterAvailability,
    ScriptedFixtureAdapter,
    TargetAdapterBinding,
    TargetFixtureProvenance,
    UnavailableAdapter,
    VerifierCase,
    VerifierHarness,
    fixture_state_digest,
)


def case(
    case_id: str,
    draft: tuple[int, ...],
    target: tuple[int, ...],
    committed: tuple[int, ...],
    *,
    bonus: bool = False,
    state: str | None = None,
) -> VerifierCase:
    return VerifierCase(
        case_id=case_id,
        draft_tokens=draft,
        target_tokens=target,
        expected_committed_tokens=committed,
        expected_state_digest=fixture_state_digest(committed) if state is None else state,
        k=len(draft),
        include_bonus_on_full_accept=bonus,
    )


def test_missing_backend_is_honest_not_run() -> None:
    receipt = VerifierHarness(UnavailableAdapter("Qwen/llama.cpp adapter unavailable")).k_sweep(
        (case("unused", (1,), (1,), (1,)),)
    )
    assert receipt.status == "NOT_RUN"
    assert receipt.execution_class == "NOT_RUN"
    assert receipt.cases == ()


def test_offline_fixture_handles_bonus_correction_and_state_comparison() -> None:
    harness = VerifierHarness(ScriptedFixtureAdapter())
    full = case("full", (1, 2), (1, 2, 3), (1, 2, 3), bonus=True)
    reject = case("reject", (1, 9, 3), (1, 2, 3), (1, 2))
    receipt = harness.k_sweep((full, reject))
    assert receipt.status == "OFFLINE_FIXTURE"
    assert receipt.cases[0].bonus_token == 3
    assert receipt.cases[1].correction_token == 2
    assert receipt.cases[1].accepted_draft_tokens == 1
    assert json.loads(receipt.as_json())["execution_class"] == "OFFLINE_FIXTURE"


def test_forced_rejection_and_long_mixed_trajectories_are_fixture_only() -> None:
    harness = VerifierHarness(ScriptedFixtureAdapter())
    cases = (
        case("reject-0", (9, 2, 3), (1, 2, 3), (1,)),
        case("reject-1", (1, 9, 3), (1, 2, 3), (1, 2)),
        case("reject-2", (1, 2, 9), (1, 2, 3), (1, 2, 3)),
    )
    forced = harness.forced_rejection_every_position(cases)
    mixed = harness.long_mixed_trajectories(cases)
    assert forced.status == "OFFLINE_FIXTURE"
    assert mixed.status == "OFFLINE_FIXTURE"
    assert all(result.state_matches for result in forced.cases)


def test_state_mismatch_is_a_failure_not_a_hardware_pass() -> None:
    harness = VerifierHarness(ScriptedFixtureAdapter())
    receipt = harness.provider_floor((case("bad", (1,), (1,), (1,), state="wrong"),))
    assert receipt.status == "FAILED"
    assert receipt.cases[0].failures == ("mutable_state_mismatch",)


def test_fixture_is_explicitly_offline_not_a_hardware_receipt() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "verifier" / "offline_token_state_cases.json"
    payload = json.loads(fixture.read_text())
    assert payload["evidence_class"] == "OFFLINE_FIXTURE"
    assert "not a target backend gate" in payload["purpose"]


class SelfDeclaredTargetAdapter:
    name = "self-declared-target"

    def __init__(self) -> None:
        self.verify_calls = 0

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("TARGET_BACKEND", "untrusted self-declaration")

    def verify(self, case: VerifierCase):
        self.verify_calls += 1
        raise AssertionError("a target adapter needs a runner-minted permit")


def test_target_adapter_cannot_run_through_harness_without_permit() -> None:
    adapter = SelfDeclaredTargetAdapter()
    receipt = VerifierHarness(adapter).k_sweep((case("unused", (1,), (1,), (1,)),))
    assert receipt.status == "NOT_RUN"
    assert receipt.execution_class == "NOT_RUN"
    assert adapter.verify_calls == 0


def test_target_binding_requires_a_hashed_immutable_runtime_configuration() -> None:
    binding = TargetAdapterBinding(
        schema_version="pcss.verifier-adapter-binding/v2",
        adapter_source_path="/tmp/adapter.py",
        backend_binary_path="/tmp/bridge",
        model_artifact_path="/tmp/model.gguf",
        llama_cpp_build_fingerprint="a" * 64,
        runtime_configuration_fingerprint="b" * 64,
    )
    assert binding.runtime_configuration_fingerprint == "b" * 64


def test_context_bearing_target_fixture_provenance_is_strict() -> None:
    provenance = TargetFixtureProvenance(
        schema_version="pcss.target-verifier-fixture-provenance/v1",
        reference_method="serial-greedy-full-state/v1",
        model_artifact_sha256="a" * 64,
        runtime_configuration_fingerprint="b" * 64,
    )
    case_with_context = VerifierCase(
        case_id="context",
        initial_context_tokens=(1, 2),
        draft_tokens=(3,),
        target_tokens=(3,),
        expected_committed_tokens=(3,),
        expected_state_digest="state",
        k=1,
        reference_provenance=provenance,
    )
    assert case_with_context.initial_context_tokens == (1, 2)
    assert case_with_context.reference_provenance is provenance
