from __future__ import annotations

import pytest

from pcss.metrics import (
    CapacityValue,
    PracticalOutcome,
    accepted_prefix_value,
    active_capture_ratio,
    capacity_curve,
    chain_survival,
    clustered_bootstrap_ci,
    opportunity,
    oracle_reuse_ratio,
    request_cluster_values,
    source_attribution,
    viable_opportunity_recovery_rate,
)
from pcss.replay import Candidate, Opportunity, ScopeIdentity, SourceTier


SCOPE = ScopeIdentity(
    principal_namespace="private/local",
    application_id="app-a",
    repository_id="repo-a",
    workflow_stage="edit",
    semantic_block_id="block-a",
    session_id="session-a",
    revision_family="git",
    path_hash="path-a",
    before_blob="blob-a",
)


def candidate(tier: SourceTier, prefix: int, *, source: str) -> Candidate:
    return Candidate(
        tier=tier,
        source_record_id=source,
        source_cursor=2,
        matched_suffix_length=2,
        continuation=tuple(range(prefix)),
        accepted_prefix_length=prefix,
        source_token_bytes=16,
        source_scope=SCOPE,
        provenance_family="repository",
        source_class="visible_code",
    )


def fixture_opportunities() -> tuple[Opportunity, Opportunity]:
    first = Opportunity(
        record_id="request-one",
        position=0,
        target_remaining_tokens=5,
        candidates=(
            candidate(SourceTier.CURRENT_CONTEXT, 1, source="context"),
            candidate(SourceTier.PERSONAL_PERSISTENT_HISTORY, 3, source="history"),
        ),
    )
    second = Opportunity(
        record_id="request-two",
        position=0,
        target_remaining_tokens=5,
        candidates=(candidate(SourceTier.SAME_REPOSITORY, 2, source="repo"),),
    )
    return first, second


def test_opportunity_chain_survival_and_source_credit() -> None:
    first, second = fixture_opportunities()

    summary = opportunity((first, second), minimum_prefix=2)
    survival = chain_survival((first, second), max_prefix=3)
    attribution = source_attribution((first, second))

    assert summary.positions == 2
    assert summary.positive_positions == 2
    assert summary.accepted_prefix_value == 5
    assert accepted_prefix_value((first, second), persistent_unique=True) == 4
    assert oracle_reuse_ratio((first, second), minimum_prefix=3) == 0.5
    assert [(point.prefix_length, point.survival_rate) for point in survival] == [
        (1, 1.0),
        (2, 1.0),
        (3, 0.5),
    ]
    assert attribution.accepted_prefix_value == 5
    assert dict(attribution.tokens_by_tier) == {
        SourceTier.CURRENT_CONTEXT: 1,
        SourceTier.PERSONAL_PERSISTENT_HISTORY: 2,
        SourceTier.SAME_REPOSITORY: 2,
    }


def test_vorr_and_acr_keep_oracle_and_practical_denominators_separate() -> None:
    first, second = fixture_opportunities()
    outcomes = (
        PracticalOutcome(
            "request-one",
            0,
            3,
            source_tier=SourceTier.PERSONAL_PERSISTENT_HISTORY,
        ),
        PracticalOutcome(
            "request-two",
            0,
            2,
            source_tier=SourceTier.SAME_REPOSITORY,
            on_time=False,
        ),
    )

    vorr = viable_opportunity_recovery_rate((first, second), outcomes, minimum_prefix=2)
    acr = active_capture_ratio((first, second), outcomes)

    assert (vorr.oracle_positions, vorr.recovered_positions, vorr.rate) == (2, 1, 0.5)
    # First request captures two tokens beyond its current-context source;
    # second is excluded because it was not on time.
    assert acr.practical_marginal_value == 2
    assert acr.full_archive_oracle_marginal_value == 4
    assert acr.ratio == 0.5


def test_capacity_curve_reports_offline_value_per_accounted_byte() -> None:
    curve = capacity_curve(
        (
            CapacityValue(100, 2.0, 4.0),
            CapacityValue(200, 4.0, 4.0),
        )
    )

    assert curve[0].active_capture_ratio == 0.5
    assert curve[0].marginal_value_rate is None
    assert curve[1].active_capture_ratio == 1.0
    assert curve[1].marginal_value_rate == pytest.approx(0.02)


def test_clustered_bootstrap_resamples_requests_not_token_positions() -> None:
    interval = clustered_bootstrap_ci(
        {"one-request": (0.0, 1.0)},
        bootstrap_replicates=100,
        seed=7,
    )

    assert interval.request_clusters == 1
    assert interval.estimate == 0.5
    assert interval.lower == 0.5
    assert interval.upper == 0.5


def test_request_clusters_follow_trace_request_id_not_record_id() -> None:
    first, second = fixture_opportunities()
    first = Opportunity(
        first.record_id,
        first.position,
        first.target_remaining_tokens,
        first.candidates,
        request_id="shared-request",
    )
    second = Opportunity(
        second.record_id,
        second.position,
        second.target_remaining_tokens,
        second.candidates,
        request_id="shared-request",
    )

    clusters = request_cluster_values((first, second))

    assert set(clusters) == {"shared-request"}
    assert clusters["shared-request"] == (3.0, 2.0)
