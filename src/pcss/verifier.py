"""Target-agnostic verification-harness contracts for future backend gates.

The only executable adapter included here is an *offline fixture* adapter used
to test the protocol itself.  `UnavailableAdapter` is the default boundary for
the absent Qwen/llama.cpp/M4 backend and always returns `NOT_RUN`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol

from .canonical import ContractError, canonical_json, require_int, require_string, sha256

HarnessStatus = Literal["NOT_RUN", "OFFLINE_FIXTURE", "FAILED"]
ExecutionClass = Literal["NOT_RUN", "OFFLINE_FIXTURE"]


def _tokens(value: tuple[int, ...], field: str, *, allow_empty: bool = False) -> None:
    if not allow_empty and not value:
        raise ContractError(f"{field} must not be empty")
    for index, token in enumerate(value):
        require_int(token, f"{field}[{index}]", minimum=0)


@dataclass(frozen=True, slots=True)
class AdapterAvailability:
    execution_class: ExecutionClass
    reason: str

    def __post_init__(self) -> None:
        require_string(self.reason, "availability.reason")


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

    def __init__(self, adapter: VerifierAdapter) -> None:
        self.adapter = adapter

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
            status: HarnessStatus = "FAILED" if failures else "OFFLINE_FIXTURE"
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
        overall: HarnessStatus = "FAILED" if any(item.status == "FAILED" for item in results) else "OFFLINE_FIXTURE"
        return HarnessReceipt(
            suite=suite,
            adapter=self.adapter.name,
            status=overall,
            execution_class="OFFLINE_FIXTURE",
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
