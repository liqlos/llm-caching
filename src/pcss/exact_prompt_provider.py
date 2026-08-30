"""Read-only exact-prompt archive provider for the runtime-neutral shadow ABI."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic_ns
from typing import Callable

from .canonical import ContractError
from .exact_prompt_archive import ExactPromptArchive
from .provider import (
    Abstention,
    BaseProvider,
    Cursor,
    Fingerprints,
    Proposal,
    RequestContext,
    SourceClass,
    TrustedScope,
    Feedback,
)


@dataclass(frozen=True, slots=True)
class ExactPromptDescriptor:
    """Trusted metadata supplied by a chronology-aware request router."""

    tail_tokens: tuple[int, ...]
    source_record_id: str
    source_class: SourceClass
    scope: TrustedScope
    fingerprints: Fingerprints
    provenance_family: str
    matched_suffix_tokens: int


@dataclass(slots=True)
class _CursorState:
    descriptor: ExactPromptDescriptor
    next_token_offset: int
    invalid: bool = False


class ExactPromptArchiveShadowProvider(BaseProvider):
    """One full-prompt archive lookup per proposal; never emits a live proposal."""

    name = "exact_prompt_archive_shadow"

    def __init__(
        self,
        archive: ExactPromptArchive,
        descriptors: tuple[ExactPromptDescriptor, ...],
        *,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        super().__init__(clock_ns=clock_ns)
        if not descriptors:
            raise ContractError("exact-prompt archive provider requires at least one descriptor")
        if len({descriptor.tail_tokens for descriptor in descriptors}) != len(descriptors):
            raise ContractError("exact-prompt archive provider descriptors must have unique prompt keys")
        self._archive = archive
        self._descriptors = {descriptor.tail_tokens: descriptor for descriptor in descriptors}
        self._cursors: dict[str, _CursorState] = {}

    def propose(
        self,
        request_id: str,
        *,
        tail_tokens: tuple[int, ...],
        max_tokens: int,
        deadline_ns: int,
    ) -> Proposal | Abstention:
        preflight = self._preflight(request_id, max_tokens=max_tokens, deadline_ns=deadline_ns)
        if preflight is not None:
            return self._record_result(preflight)
        request = self._request(request_id).context
        state = self._cursors.get(request_id)
        descriptor = state.descriptor if state is not None else self._descriptors.get(tail_tokens)
        now = self._clock_ns()
        if state is not None and state.invalid:
            return self._record_result(Abstention(request_id, "authoritative_prefix_diverged", now))
        if descriptor is None:
            return self._record_result(Abstention(request_id, "no_exact_prompt_descriptor", now))
        if descriptor.fingerprints != request.fingerprints:
            return self._record_result(Abstention(request_id, "incompatible_fingerprints", now))
        if descriptor.source_class not in request.source_tiers:
            return self._record_result(Abstention(request_id, "source_tier_not_allowed", now))
        if descriptor.scope.principal_namespace != request.scope.principal_namespace:
            return self._record_result(Abstention(request_id, "scope_not_authorized", now))
        if descriptor.scope.widening_level > request.scope.widening_level:
            return self._record_result(Abstention(request_id, "widening_not_authorized", now))
        if state is None:
            hits = self._archive.lookup(tail_tokens)
            if not hits:
                return self._record_result(Abstention(request_id, "no_exact_prompt_archive_hit", now))
            if len(hits) != 1:
                return self._record_result(Abstention(request_id, "ambiguous_exact_prompt_archive_hit", now))
            hit = hits[0]
            if hit.record_id != descriptor.source_record_id:
                return self._record_result(Abstention(request_id, "source_provenance_mismatch", now))
            state = _CursorState(descriptor, hit.prompt_token_count)
            self._cursors[request_id] = state
        tokens = self._archive.archive.open_cursor(
            descriptor.source_record_id, position=state.next_token_offset
        ).read(max_tokens)
        if not tokens:
            return self._record_result(Abstention(request_id, "empty_archive_cursor", now))
        return self._record_result(
            Proposal(
                proposal_id=f"exact-prompt:{request_id}:{len(self.telemetry)}",
                request_id=request_id,
                tokens=tokens,
                source_class=descriptor.source_class,
                scope=descriptor.scope,
                cursor=Cursor("immutable-exact-prompt", descriptor.source_record_id, state.next_token_offset),
                matched_suffix_tokens=descriptor.matched_suffix_tokens if state.next_token_offset == len(descriptor.tail_tokens) else 0,
                score=float(descriptor.matched_suffix_tokens),
                ambiguity=1,
                provenance_family=descriptor.provenance_family,
                ready_ns=now,
            )
        )

    def feedback(self, request_id: str, feedback: Feedback) -> None:
        super().feedback(request_id, feedback)
        state = self._cursors.get(request_id)
        if state is None or feedback.status in {"cancelled", "not_verified"}:
            return
        if not feedback.committed_tokens:
            state.invalid = True
            return
        try:
            expected = self._archive.archive.read_tokens(
                state.descriptor.source_record_id, state.next_token_offset, len(feedback.committed_tokens)
            )
        except ContractError:
            state.invalid = True
            return
        if expected != feedback.committed_tokens:
            state.invalid = True
            return
        state.next_token_offset += len(feedback.committed_tokens)

    def end_request(self, request_id: str) -> None:
        super().end_request(request_id)
        self._cursors.pop(request_id, None)
