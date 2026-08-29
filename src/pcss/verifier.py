"""Target-agnostic verification-harness contracts for future backend gates.

The only executable adapter included here is an *offline fixture* adapter used
to test the protocol itself.  `UnavailableAdapter` is the default boundary for
the absent Qwen/llama.cpp/M4 backend and always returns `NOT_RUN`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol

from .canonical import ContractError, canonical_json, require_int, require_string, sha256

HarnessStatus = Literal["NOT_RUN", "OFFLINE_FIXTURE", "VERIFIED", "FAILED"]
ExecutionClass = Literal["NOT_RUN", "OFFLINE_FIXTURE", "TARGET_BACKEND"]


def _tokens(value: tuple[int, ...], field: str, *, allow_empty: bool = False) -> None:
    if not allow_empty and not value:
        raise ContractError(f"{field} must not be empty")
    for index, token in enumerate(value):
        require_int(token, f"{field}[{index}]", minimum=0)


def _sha256(value: str, field: str) -> str:
    value = require_string(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class AdapterAvailability:
    execution_class: ExecutionClass
    reason: str

    def __post_init__(self) -> None:
        if self.execution_class not in {"NOT_RUN", "OFFLINE_FIXTURE", "TARGET_BACKEND"}:
            raise ContractError("availability.execution_class is invalid")
        require_string(self.reason, "availability.reason")


@dataclass(frozen=True, slots=True)
class TargetHostIdentity:
    """Runner-observed future target-host facts, never filled by baseline fixtures."""

    schema_version: str
    target_profile: str
    operating_system: str
    architecture: str
    hardware_model: str
    chip: str
    memory_bytes: int
    fingerprint: str

    @classmethod
    def from_observed(
        cls,
        *,
        target_profile: str,
        operating_system: str,
        architecture: str,
        hardware_model: str,
        chip: str,
        memory_bytes: int,
    ) -> "TargetHostIdentity":
        payload = {
            "schema_version": "pcss.host-identity/v1",
            "target_profile": target_profile,
            "operating_system": operating_system,
            "architecture": architecture,
            "hardware_model": hardware_model,
            "chip": chip,
            "memory_bytes": memory_bytes,
        }
        return cls(**payload, fingerprint=sha256(payload))

    def __post_init__(self) -> None:
        if self.schema_version != "pcss.host-identity/v1":
            raise ContractError("unsupported host identity schema")
        for field in ("target_profile", "operating_system", "architecture", "hardware_model", "chip"):
            require_string(getattr(self, field), f"host_identity.{field}")
        require_int(self.memory_bytes, "host_identity.memory_bytes", minimum=1)
        if self.fingerprint != sha256(
            {
                "schema_version": self.schema_version,
                "target_profile": self.target_profile,
                "operating_system": self.operating_system,
                "architecture": self.architecture,
                "hardware_model": self.hardware_model,
                "chip": self.chip,
                "memory_bytes": self.memory_bytes,
            }
        ):
            raise ContractError("host identity fingerprint does not match its observed fields")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_profile": self.target_profile,
            "operating_system": self.operating_system,
            "architecture": self.architecture,
            "hardware_model": self.hardware_model,
            "chip": self.chip,
            "memory_bytes": self.memory_bytes,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TargetAdapterBinding:
    """Paths and build identity for a future real backend; hashes are runner-computed."""

    schema_version: str
    adapter_source_path: str
    backend_binary_path: str
    model_artifact_path: str
    llama_cpp_build_fingerprint: str
    runtime_configuration_fingerprint: str

    def __post_init__(self) -> None:
        if self.schema_version != "pcss.verifier-adapter-binding/v2":
            raise ContractError("unsupported target adapter binding schema")
        for field in ("adapter_source_path", "backend_binary_path", "model_artifact_path"):
            require_string(getattr(self, field), f"target_adapter_binding.{field}")
        _sha256(self.llama_cpp_build_fingerprint, "target_adapter_binding.llama_cpp_build_fingerprint")
        _sha256(
            self.runtime_configuration_fingerprint,
            "target_adapter_binding.runtime_configuration_fingerprint",
        )


@dataclass(frozen=True, slots=True)
class TargetExecutionPermit:
    """Runner-minted authorization needed before a target adapter can verify cases."""

    adapter_name: str
    host_fingerprint: str
    adapter_binding_fingerprint: str

    def __post_init__(self) -> None:
        require_string(self.adapter_name, "target_execution_permit.adapter_name")
        _sha256(self.host_fingerprint, "target_execution_permit.host_fingerprint")
        _sha256(self.adapter_binding_fingerprint, "target_execution_permit.adapter_binding_fingerprint")


@dataclass(frozen=True, slots=True)
class VerifierCase:
    """A token-only fixture with expected commit and full mutable-state digest."""

    case_id: str
    draft_tokens: tuple[int, ...]
    target_tokens: tuple[int, ...]
    expected_committed_tokens: tuple[int, ...]
    expected_state_digest: str
    k: int
    include_bonus_on_full_accept: bool = False

    def __post_init__(self) -> None:
        require_string(self.case_id, "case.case_id")
        _tokens(self.draft_tokens, "case.draft_tokens")
        _tokens(self.target_tokens, "case.target_tokens")
        _tokens(self.expected_committed_tokens, "case.expected_committed_tokens")
        require_string(self.expected_state_digest, "case.expected_state_digest")
        require_int(self.k, "case.k", minimum=1)
        if self.k != len(self.draft_tokens):
            raise ContractError("case.k must equal the draft token count")


@dataclass(frozen=True, slots=True)
class AdapterOutcome:
    accepted_draft_tokens: int
    correction_token: int | None
    bonus_token: int | None
    committed_tokens: tuple[int, ...]
    mutable_state_digest: str

    def __post_init__(self) -> None:
        require_int(self.accepted_draft_tokens, "outcome.accepted_draft_tokens", minimum=0)
        for field_name in ("correction_token", "bonus_token"):
            value = getattr(self, field_name)
            if value is not None:
                require_int(value, f"outcome.{field_name}", minimum=0)
        _tokens(self.committed_tokens, "outcome.committed_tokens", allow_empty=True)
        require_string(self.mutable_state_digest, "outcome.mutable_state_digest")


class VerifierAdapter(Protocol):
    name: str

    def availability(self) -> AdapterAvailability: ...

    def verify(self, case: VerifierCase) -> AdapterOutcome: ...


class TargetBoundVerifierAdapter(VerifierAdapter, Protocol):
    """Additional contract required for a runner-authorized TARGET_BACKEND claim."""

    def target_binding(self) -> TargetAdapterBinding: ...


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    status: HarnessStatus
    accepted_draft_tokens: int | None
    committed_tokens: tuple[int, ...] | None
    correction_token: int | None
    bonus_token: int | None
    state_matches: bool | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HarnessReceipt:
    suite: str
    adapter: str
    status: HarnessStatus
    execution_class: ExecutionClass
    reason: str | None
    cases: tuple[CaseResult, ...]
    metadata: dict[str, object]

    def as_json(self) -> str:
        return canonical_json(asdict(self))


class UnavailableAdapter:
    """Explicit default: no backend present is `NOT_RUN`, never a fixture pass."""

    name = "unavailable"

    def __init__(self, reason: str = "no target backend adapter configured") -> None:
        self.reason = reason

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("NOT_RUN", self.reason)

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        raise RuntimeError("unavailable adapter must not verify")


def fixture_state_digest(committed_tokens: tuple[int, ...]) -> str:
    """Stable stand-in for a complete mutable-state snapshot in unit fixtures."""
    return sha256(
        {
            "fixture_state_version": 1,
            "token_position": len(committed_tokens),
            "attention_marker": list(committed_tokens),
            "recurrent_marker": list(reversed(committed_tokens)),
        }
    )


class ScriptedFixtureAdapter:
    """Protocol-only target semantics; cannot satisfy a hardware acceptance gate."""

    name = "scripted_offline_fixture"

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability("OFFLINE_FIXTURE", "deterministic token/state fixture only")

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        accepted = 0
        for proposed, target in zip(case.draft_tokens, case.target_tokens):
            if proposed != target:
                break
            accepted += 1
        correction: int | None = None
        bonus: int | None = None
        if accepted < len(case.draft_tokens):
            correction = case.target_tokens[accepted]
            committed = case.draft_tokens[:accepted] + (correction,)
        else:
            committed = case.draft_tokens
            if case.include_bonus_on_full_accept and len(case.target_tokens) > len(case.draft_tokens):
                bonus = case.target_tokens[len(case.draft_tokens)]
                committed += (bonus,)
        return AdapterOutcome(
            accepted_draft_tokens=accepted,
            correction_token=correction,
            bonus_token=bonus,
            committed_tokens=committed,
            mutable_state_digest=fixture_state_digest(committed),
        )


class VerifierHarness:
    """Runs token/state fixtures without claiming target-runtime correctness."""

    def __init__(self, adapter: VerifierAdapter, *, target_execution_permit: TargetExecutionPermit | None = None) -> None:
        self.adapter = adapter
        self.target_execution_permit = target_execution_permit

    def _run(self, suite: str, cases: tuple[VerifierCase, ...], metadata: dict[str, object] | None = None) -> HarnessReceipt:
        availability = self.adapter.availability()
        if availability.execution_class == "NOT_RUN":
            return HarnessReceipt(
                suite=suite,
                adapter=self.adapter.name,
                status="NOT_RUN",
                execution_class="NOT_RUN",
                reason=availability.reason,
                cases=(),
                metadata=metadata or {},
            )
        if availability.execution_class == "TARGET_BACKEND" and (
            self.target_execution_permit is None
            or self.target_execution_permit.adapter_name != self.adapter.name
        ):
            return HarnessReceipt(
                suite=suite,
                adapter=self.adapter.name,
                status="NOT_RUN",
                execution_class="NOT_RUN",
                reason="TARGET_BACKEND requires a runner-minted target execution permit",
                cases=(),
                metadata=metadata or {},
            )

        results: list[CaseResult] = []
        for case in cases:
            outcome = self.adapter.verify(case)
            failures: list[str] = []
            if outcome.committed_tokens != case.expected_committed_tokens:
                failures.append("committed_tokens_mismatch")
            state_matches = outcome.mutable_state_digest == case.expected_state_digest
            if not state_matches:
                failures.append("mutable_state_mismatch")
            if outcome.accepted_draft_tokens > len(case.draft_tokens):
                failures.append("accepted_count_out_of_range")
            status: HarnessStatus = (
                "FAILED"
                if failures
                else "VERIFIED"
                if availability.execution_class == "TARGET_BACKEND"
                else "OFFLINE_FIXTURE"
            )
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    status=status,
                    accepted_draft_tokens=outcome.accepted_draft_tokens,
                    committed_tokens=outcome.committed_tokens,
                    correction_token=outcome.correction_token,
                    bonus_token=outcome.bonus_token,
                    state_matches=state_matches,
                    failures=tuple(failures),
                )
            )
        overall: HarnessStatus = (
            "FAILED"
            if any(item.status == "FAILED" for item in results)
            else "VERIFIED"
            if availability.execution_class == "TARGET_BACKEND"
            else "OFFLINE_FIXTURE"
        )
        return HarnessReceipt(
            suite=suite,
            adapter=self.adapter.name,
            status=overall,
            execution_class=availability.execution_class,
            reason=availability.reason,
            cases=tuple(results),
            metadata=metadata or {},
        )

    def k_sweep(self, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
        return self._run("k_sweep", cases, {"k_values": sorted({case.k for case in cases})})

    def perfect_draft(self, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
        invalid = tuple(case.case_id for case in cases if case.draft_tokens != case.target_tokens[: case.k])
        receipt = self._run("perfect_draft", cases, {"invalid_perfect_draft_cases": list(invalid)})
        if invalid and receipt.status != "NOT_RUN":
            return replace(receipt, status="FAILED")
        return receipt

    def provider_floor(self, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
        return self._run("provider_floor", cases, {"timing_claim": "not available from offline fixture"})

    def forced_rejection_every_position(self, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
        positions = sorted({case.k - 1 for case in cases})
        return self._run("forced_rejection_every_position", cases, {"draft_lengths": positions})

    def long_mixed_trajectories(self, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
        return self._run("long_mixed_trajectories", cases, {"case_count": len(cases)})
