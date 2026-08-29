"""A runtime-neutral, deliberately small PCSS provider boundary.

Nothing in this module invokes a model or opens a store.  It only defines the
request/proposal lifecycle that a later lookup plane may implement behind a
target decoder after the target-specific correctness gates have run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import monotonic_ns
from typing import Callable, Literal, Protocol

from .canonical import ContractError, canonical_json, require_int, require_string

SourceClass = Literal[
    "current_context",
    "current_source",
    "same_session",
    "same_repository",
    "personal_persistent_history",
    "imported_corpus",
]
ProposalStatus = Literal["proposal", "abstain"]
FeedbackStatus = Literal["all_accepted", "rejected", "cancelled", "not_verified"]
SOURCE_CLASSES = frozenset(
    {
        "current_context",
        "current_source",
        "same_session",
        "same_repository",
        "personal_persistent_history",
        "imported_corpus",
    }
)


@dataclass(frozen=True, slots=True)
class Fingerprints:
    """Non-text namespaces that make a candidate compatible with a request."""

    model: str
    tokenizer: str
    chat_template: str
    generation_protocol: str

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            require_string(value, f"fingerprints.{field_name}")


@dataclass(frozen=True, slots=True)
class TrustedScope:
    """Immutable routing labels created outside model text."""

    principal_namespace: str
    application: str
    repository: str | None
    revision_family: str | None
    operation_class: str
    workflow_block: str | None
    schema_or_tool_version: str | None
    widening_level: int = 0

    def __post_init__(self) -> None:
        require_string(self.principal_namespace, "scope.principal_namespace")
        require_string(self.application, "scope.application")
        require_string(self.operation_class, "scope.operation_class")
        require_int(self.widening_level, "scope.widening_level", minimum=0)
        if self.widening_level > 4:
            raise ContractError("scope.widening_level must be <= 4")
        for field_name in ("repository", "revision_family", "workflow_block", "schema_or_tool_version"):
            value = getattr(self, field_name)
            if value is not None:
                require_string(value, f"scope.{field_name}")


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    fingerprints: Fingerprints
    scope: TrustedScope
    source_tiers: tuple[SourceClass, ...]
    deadline_ns: int
    probe_budget: int

    def __post_init__(self) -> None:
        require_string(self.request_id, "request_id")
        require_int(self.deadline_ns, "deadline_ns", minimum=0)
        require_int(self.probe_budget, "probe_budget", minimum=0)
        if not self.source_tiers:
            raise ContractError("source_tiers must not be empty")
        if len(set(self.source_tiers)) != len(self.source_tiers):
            raise ContractError("source_tiers must not contain duplicates")
        if any(source not in SOURCE_CLASSES for source in self.source_tiers):
            raise ContractError("source_tiers contains an unknown source class")


@dataclass(frozen=True, slots=True)
class Cursor:
    segment_id: str
    sequence_id: str
    token_offset: int

    def __post_init__(self) -> None:
        require_string(self.segment_id, "cursor.segment_id")
        require_string(self.sequence_id, "cursor.sequence_id")
        require_int(self.token_offset, "cursor.token_offset", minimum=0)


@dataclass(frozen=True, slots=True)
class Proposal:
    proposal_id: str
    request_id: str
    tokens: tuple[int, ...]
    source_class: SourceClass
    scope: TrustedScope
    cursor: Cursor | None
    matched_suffix_tokens: int
    score: float
    ambiguity: int
    provenance_family: str
    ready_ns: int
    shadow_only: bool = True

    def __post_init__(self) -> None:
        require_string(self.proposal_id, "proposal.proposal_id")
        require_string(self.request_id, "proposal.request_id")
        if not self.tokens:
            raise ContractError("proposal.tokens must not be empty")
        for index, token in enumerate(self.tokens):
            require_int(token, f"proposal.tokens[{index}]", minimum=0)
        require_int(self.matched_suffix_tokens, "proposal.matched_suffix_tokens", minimum=0)
        require_int(self.ambiguity, "proposal.ambiguity", minimum=0)
        require_int(self.ready_ns, "proposal.ready_ns", minimum=0)
        require_string(self.provenance_family, "proposal.provenance_family")
        if self.source_class not in SOURCE_CLASSES:
            raise ContractError("proposal.source_class is invalid")
        if not self.shadow_only:
            raise ContractError("live provider proposals are deferred until target gates pass")


@dataclass(frozen=True, slots=True)
class Abstention:
    request_id: str
    reason: str
    ready_ns: int
    status: ProposalStatus = "abstain"

    def __post_init__(self) -> None:
        require_string(self.request_id, "abstention.request_id")
        require_string(self.reason, "abstention.reason")
        require_int(self.ready_ns, "abstention.ready_ns", minimum=0)


@dataclass(frozen=True, slots=True)
class Feedback:
    proposal_id: str | None
    status: FeedbackStatus
    accepted_tokens: int
    committed_tokens: tuple[int, ...]
    verification_latency_ns: int | None
    source_attribution: SourceClass | None

    def __post_init__(self) -> None:
        if self.proposal_id is not None:
            require_string(self.proposal_id, "feedback.proposal_id")
        require_int(self.accepted_tokens, "feedback.accepted_tokens", minimum=0)
        if self.accepted_tokens > len(self.committed_tokens):
            raise ContractError("feedback.accepted_tokens cannot exceed committed_tokens")
        for index, token in enumerate(self.committed_tokens):
            require_int(token, f"feedback.committed_tokens[{index}]", minimum=0)
        if self.verification_latency_ns is not None:
            require_int(self.verification_latency_ns, "feedback.verification_latency_ns", minimum=0)
        if self.source_attribution is not None and self.source_attribution not in SOURCE_CLASSES:
            raise ContractError("feedback.source_attribution is invalid")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event: Literal["begin_request", "proposal", "abstain", "feedback", "end_request", "cancel"]
    request_id: str
    at_ns: int
    payload: dict[str, object]

    def as_json(self) -> str:
        return canonical_json(asdict(self))


class Provider(Protocol):
    def begin_request(self, request: RequestContext) -> None: ...

    def propose(
        self,
        request_id: str,
        *,
        tail_tokens: tuple[int, ...],
        max_tokens: int,
        deadline_ns: int,
    ) -> Proposal | Abstention: ...

    def feedback(self, request_id: str, feedback: Feedback) -> None: ...

    def end_request(self, request_id: str) -> None: ...


@dataclass(slots=True)
class _OpenRequest:
    context: RequestContext
    cancelled: bool = False


class BaseProvider:
    """Lifecycle checks and JSON-ready telemetry common to all test providers."""

    name = "base"

    def __init__(self, *, clock_ns: Callable[[], int] = monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._requests: dict[str, _OpenRequest] = {}
        self.telemetry: list[TelemetryEvent] = []

    def _emit(self, event: TelemetryEvent) -> None:
        self.telemetry.append(event)

    def begin_request(self, request: RequestContext) -> None:
        if request.request_id in self._requests:
            raise ContractError(f"request already exists: {request.request_id}")
        self._requests[request.request_id] = _OpenRequest(request)
        self._emit(TelemetryEvent("begin_request", request.request_id, self._clock_ns(), {"provider": self.name}))

    def _request(self, request_id: str) -> _OpenRequest:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise ContractError(f"unknown or ended request: {request_id}") from error

    def cancel(self, request_id: str) -> None:
        request = self._request(request_id)
        request.cancelled = True
        self._emit(TelemetryEvent("cancel", request_id, self._clock_ns(), {"provider": self.name}))

    def _preflight(
        self,
        request_id: str,
        *,
        max_tokens: int,
        deadline_ns: int,
    ) -> Abstention | None:
        request = self._request(request_id)
        require_int(max_tokens, "max_tokens", minimum=1)
        require_int(deadline_ns, "deadline_ns", minimum=0)
        now = self._clock_ns()
        if request.cancelled:
            return Abstention(request_id, "cancelled", now)
        if now > deadline_ns or now > request.context.deadline_ns:
            return Abstention(request_id, "deadline_exceeded", now)
        return None

    def _record_result(self, result: Proposal | Abstention) -> Proposal | Abstention:
        if isinstance(result, Proposal):
            payload: dict[str, object] = {
                "provider": self.name,
                "proposal_id": result.proposal_id,
                "token_count": len(result.tokens),
                "source_class": result.source_class,
                "widening_level": result.scope.widening_level,
                "matched_suffix_tokens": result.matched_suffix_tokens,
                "provenance_family": result.provenance_family,
                "shadow_only": result.shadow_only,
            }
            event = "proposal"
        else:
            payload = {"provider": self.name, "reason": result.reason}
            event = "abstain"
        self._emit(TelemetryEvent(event, result.request_id, result.ready_ns, payload))
        return result

    def feedback(self, request_id: str, feedback: Feedback) -> None:
        self._request(request_id)
        self._emit(
            TelemetryEvent(
                "feedback",
                request_id,
                self._clock_ns(),
                {
                    "provider": self.name,
                    "proposal_id": feedback.proposal_id,
                    "status": feedback.status,
                    "accepted_tokens": feedback.accepted_tokens,
                    "committed_token_count": len(feedback.committed_tokens),
                    "verification_latency_ns": feedback.verification_latency_ns,
                    "source_attribution": feedback.source_attribution,
                },
            )
        )

    def end_request(self, request_id: str) -> None:
        self._request(request_id)
        del self._requests[request_id]
        self._emit(TelemetryEvent("end_request", request_id, self._clock_ns(), {"provider": self.name}))


class NoOpProvider(BaseProvider):
    """A no-op provider control: every request abstains immediately."""

    name = "no_op"

    def propose(self, request_id: str, *, tail_tokens: tuple[int, ...], max_tokens: int, deadline_ns: int) -> Proposal | Abstention:
        result = self._preflight(request_id, max_tokens=max_tokens, deadline_ns=deadline_ns)
        return self._record_result(result or Abstention(request_id, "no_provider", self._clock_ns()))


class AlwaysMissProvider(NoOpProvider):
    """Infrastructure control that proves an intentional provider miss path."""

    name = "always_miss"

    def propose(self, request_id: str, *, tail_tokens: tuple[int, ...], max_tokens: int, deadline_ns: int) -> Proposal | Abstention:
        result = self._preflight(request_id, max_tokens=max_tokens, deadline_ns=deadline_ns)
        return self._record_result(result or Abstention(request_id, "forced_miss", self._clock_ns()))


@dataclass(frozen=True, slots=True)
class ShadowCandidate:
    tail_tokens: tuple[int, ...]
    tokens: tuple[int, ...]
    source_class: SourceClass
    scope: TrustedScope
    fingerprints: Fingerprints
    cursor: Cursor | None = None
    matched_suffix_tokens: int = 0
    score: float = 0.0
    ambiguity: int = 0
    provenance_family: str = "fixture"

    def __post_init__(self) -> None:
        _ = Proposal(
            proposal_id="validation",
            request_id="validation",
            tokens=self.tokens,
            source_class=self.source_class,
            scope=self.scope,
            cursor=self.cursor,
            matched_suffix_tokens=self.matched_suffix_tokens,
            score=self.score,
            ambiguity=self.ambiguity,
            provenance_family=self.provenance_family,
            ready_ns=0,
        )


class DeterministicShadowProvider(BaseProvider):
    """A deterministic lookup-plane fixture; proposals are telemetry-only."""

    name = "deterministic_shadow"

    def __init__(self, candidates: tuple[ShadowCandidate, ...], *, clock_ns: Callable[[], int] = monotonic_ns) -> None:
        super().__init__(clock_ns=clock_ns)
        self._candidates = {candidate.tail_tokens: candidate for candidate in candidates}

    def propose(self, request_id: str, *, tail_tokens: tuple[int, ...], max_tokens: int, deadline_ns: int) -> Proposal | Abstention:
        preflight = self._preflight(request_id, max_tokens=max_tokens, deadline_ns=deadline_ns)
        if preflight is not None:
            return self._record_result(preflight)
        request = self._request(request_id).context
        candidate = self._candidates.get(tail_tokens)
        now = self._clock_ns()
        if candidate is None:
            return self._record_result(Abstention(request_id, "no_exact_candidate", now))
        if candidate.fingerprints != request.fingerprints:
            return self._record_result(Abstention(request_id, "incompatible_fingerprints", now))
        if candidate.source_class not in request.source_tiers:
            return self._record_result(Abstention(request_id, "source_tier_not_allowed", now))
        if candidate.scope.principal_namespace != request.scope.principal_namespace:
            return self._record_result(Abstention(request_id, "scope_not_authorized", now))
        if candidate.scope.widening_level > request.scope.widening_level:
            return self._record_result(Abstention(request_id, "widening_not_authorized", now))
        proposed = candidate.tokens[:max_tokens]
        if not proposed:
            return self._record_result(Abstention(request_id, "empty_candidate", now))
        return self._record_result(
            Proposal(
                proposal_id=f"shadow:{request_id}:{len(self.telemetry)}",
                request_id=request_id,
                tokens=proposed,
                source_class=candidate.source_class,
                scope=candidate.scope,
                cursor=candidate.cursor,
                matched_suffix_tokens=candidate.matched_suffix_tokens,
                score=candidate.score,
                ambiguity=candidate.ambiguity,
                provenance_family=candidate.provenance_family,
                ready_ns=now,
            )
        )
