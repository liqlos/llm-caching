from __future__ import annotations

import json

import pytest

from pcss.provider import (
    AlwaysMissProvider,
    Cursor,
    DeterministicShadowProvider,
    Feedback,
    Fingerprints,
    NoOpProvider,
    RequestContext,
    ShadowCandidate,
    TrustedScope,
)


class Clock:
    def __init__(self, now: int = 100) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def context() -> RequestContext:
    return RequestContext(
        request_id="r1",
        fingerprints=Fingerprints("model", "tokenizer", "template", "protocol"),
        scope=TrustedScope("principal", "app", "repo", "revision", "generation", "block", "v1"),
        source_tiers=("current_context", "same_repository"),
        deadline_ns=200,
        probe_budget=1,
    )


def test_noop_and_always_miss_are_explicit_controls() -> None:
    clock = Clock()
    noop = NoOpProvider(clock_ns=clock)
    noop.begin_request(context())
    assert noop.propose("r1", tail_tokens=(1,), max_tokens=2, deadline_ns=150).reason == "no_provider"

    miss = AlwaysMissProvider(clock_ns=clock)
    miss.begin_request(context())
    assert miss.propose("r1", tail_tokens=(1,), max_tokens=2, deadline_ns=150).reason == "forced_miss"
    assert json.loads(miss.telemetry[-1].as_json())["event"] == "abstain"


def test_shadow_provider_checks_scope_fingerprints_deadline_and_cancellation() -> None:
    clock = Clock()
    request = context()
    candidate = ShadowCandidate(
        tail_tokens=(7, 8),
        tokens=(9, 10, 11),
        source_class="same_repository",
        scope=request.scope,
        fingerprints=request.fingerprints,
        cursor=Cursor("seg", "seq", 3),
        matched_suffix_tokens=2,
        provenance_family="independent-run-a",
    )
    provider = DeterministicShadowProvider((candidate,), clock_ns=clock)
    provider.begin_request(request)
    proposal = provider.propose("r1", tail_tokens=(7, 8), max_tokens=2, deadline_ns=150)
    assert proposal.tokens == (9, 10)
    assert proposal.shadow_only is True

    provider.feedback(
        "r1",
        Feedback("proposal", "rejected", 1, (9, 42), 10, "same_repository"),
    )
    provider.cancel("r1")
    assert provider.propose("r1", tail_tokens=(7, 8), max_tokens=2, deadline_ns=150).reason == "cancelled"
    provider.end_request("r1")

    expired = DeterministicShadowProvider((candidate,), clock_ns=Clock(201))
    expired.begin_request(context())
    assert expired.propose("r1", tail_tokens=(7, 8), max_tokens=2, deadline_ns=250).reason == "deadline_exceeded"


def test_shadow_provider_fails_closed_on_incompatible_fingerprints() -> None:
    clock = Clock()
    request = context()
    incompatible = ShadowCandidate(
        tail_tokens=(1,),
        tokens=(2,),
        source_class="same_repository",
        scope=request.scope,
        fingerprints=Fingerprints("other", "tokenizer", "template", "protocol"),
    )
    provider = DeterministicShadowProvider((incompatible,), clock_ns=clock)
    provider.begin_request(request)
    assert provider.propose("r1", tail_tokens=(1,), max_tokens=1, deadline_ns=150).reason == "incompatible_fingerprints"


def test_provider_rejects_unknown_source_classes() -> None:
    with pytest.raises(ValueError, match="unknown source class"):
        RequestContext(
            request_id="r",
            fingerprints=Fingerprints("model", "tokenizer", "template", "protocol"),
            scope=TrustedScope("principal", "app", None, None, "generation", None, None),
            source_tiers=("not-a-tier",),  # type: ignore[arg-type]
            deadline_ns=1,
            probe_budget=1,
        )
