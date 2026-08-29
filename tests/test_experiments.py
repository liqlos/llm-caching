from __future__ import annotations

import pytest

from pcss.experiments import (
    ConsensusObservation,
    DecayedNegativeEvidence,
    NegativeEvidenceEvent,
    calibrate_consensus,
    compare_edit_protocols,
    compare_scope_policies,
    consensus_by_provenance,
    current_source_reanchor_oracle,
    patch_edit_representation,
    simulate_decayed_negative_evidence,
)
from pcss.replay import (
    Candidate,
    Fingerprints,
    Opportunity,
    ReplayRecord,
    ReplayResult,
    ScopeIdentity,
    SourceTier,
    build_census,
)


FINGERPRINTS = Fingerprints("model", "tokenizer", "template", "protocol")


def scope(repository: str, block: str) -> ScopeIdentity:
    return ScopeIdentity(
        principal_namespace="private/local",
        application_id="app",
        repository_id=repository,
        workflow_stage="edit",
        semantic_block_id=block,
        session_id=None,
        revision_family="git",
        path_hash=f"path-{repository}-{block}",
        before_blob=None,
    )


TARGET_SCOPE = scope("repo-a", "block-a")


def candidate(
    source: str,
    source_scope: ScopeIdentity,
    tier: SourceTier,
    suffix: int,
    accepted: int,
    family: str = "repository",
) -> Candidate:
    return Candidate(
        tier=tier,
        source_record_id=source,
        source_cursor=3,
        matched_suffix_length=suffix,
        continuation=tuple(range(accepted)),
        accepted_prefix_length=accepted,
        source_token_bytes=16,
        source_scope=source_scope,
        provenance_family=family,
        source_class="visible_code",
    )


def one_position_result(candidates: tuple[Candidate, ...]) -> ReplayResult:
    target = ReplayRecord(
        record_id="target",
        timestamp_ns=1,
        chronological_index=1,
        prompt_tokens=(1,),
        committed_tokens=(2,),
        fingerprints=FINGERPRINTS,
        scope=TARGET_SCOPE,
        source_class="visible_code",
        provenance_family="repository",
        eligible_for_persistent_store=True,
    )
    opportunity = Opportunity("target", 0, 8, candidates)
    return ReplayResult((target,), (opportunity,), build_census((target,)))


def test_scope_routes_hold_byte_and_probe_budget_constant() -> None:
    result = one_position_result(
        (
            candidate(
                "global", scope("repo-other", "block-other"), SourceTier.PERSONAL_PERSISTENT_HISTORY, 10, 5
            ),
            candidate("repository", scope("repo-a", "block-other"), SourceTier.SAME_REPOSITORY, 9, 3),
            candidate("block", scope("repo-a", "block-a"), SourceTier.SAME_REPOSITORY, 8, 2),
        )
    )

    compared = {item.policy.name: item for item in compare_scope_policies(result, byte_budget=16, probe_budget=1)}

    assert compared["global"].selected_prefix_value == 5
    assert compared["repository_isolation"].selected_prefix_value == 3
    assert compared["block_isolation"].selected_prefix_value == 2
    assert compared["bounded_hierarchical_widening"].selected_prefix_value == 2
    for experiment in compared.values():
        position = experiment.positions[0]
        assert position.probes_used <= experiment.policy.probe_budget
        assert position.source_bytes_used <= experiment.policy.byte_budget


def test_current_source_reanchor_and_patch_representation_are_offline_oracles() -> None:
    before = (1, 2, 3, 4, 5)
    after = (1, 2, 9, 4, 5)

    reuse = current_source_reanchor_oracle(before, after)
    patch = patch_edit_representation(before, after)
    comparison = compare_edit_protocols(before, after)

    assert reuse.positive_positions >= 2
    assert any(anchor.output_position == 4 and anchor.accepted_prefix_length == 1 for anchor in reuse.reanchors)
    assert patch.copied_output_tokens == 4
    assert patch.edited_output_tokens == 1
    assert dict((value.protocol, value.accepted_prefix_value) for value in comparison.values)[
        "full_regeneration"
    ] == 0
    assert dict((value.protocol, value.accepted_prefix_value) for value in comparison.values)[
        "patch_edit_representation"
    ] == 4


def test_consensus_shrinks_duplicate_provenance_support_and_calibrates() -> None:
    duplicated = (
        candidate("a1", scope("repo-a", "block-a"), SourceTier.SAME_REPOSITORY, 4, 2, "family-a"),
        candidate("a2", scope("repo-a", "block-a"), SourceTier.SAME_REPOSITORY, 3, 2, "family-a"),
        candidate("b", scope("repo-a", "block-a"), SourceTier.SAME_REPOSITORY, 3, 2, "family-b"),
    )
    # Make the independent family agree on the same actual continuation.
    duplicated = tuple(
        Candidate(
            **{**item.__dict__, "continuation": (7, 8) if item.source_record_id != "a2" else (7, 8, 9)}
        )
        for item in duplicated
    )
    independent = (duplicated[0], duplicated[2])

    consensus = consensus_by_provenance(duplicated)
    no_duplicate = consensus_by_provenance(independent)
    calibration = calibrate_consensus(
        (
            ConsensusObservation(0.1, 1, 0),
            ConsensusObservation(0.9, 2, 3),
        ),
        bins=2,
    )

    assert consensus.raw_support == 3
    assert consensus.independent_support == 2
    assert consensus.consensus_prefix == (7, 8)
    assert consensus.support_shrunk_score < no_duplicate.support_shrunk_score
    assert calibration[0].positive_rate == 0.0
    assert calibration[1].positive_rate == 1.0


def test_decayed_negative_evidence_recovers_after_a_success() -> None:
    simulator = DecayedNegativeEvidence(half_life_events=10.0, suppress_at=1.5, max_entries=2)

    first = simulator.observe(NegativeEvidenceEvent("anchor", 0, 0))
    second = simulator.observe(NegativeEvidenceEvent("anchor", 1, 0))
    assert first.score_after == 1.0
    assert second.score_after > 1.5
    assert simulator.should_suppress("anchor", 1)
    recovered = simulator.observe(NegativeEvidenceEvent("anchor", 2, 1))
    assert recovered.score_after < 1.5
    assert not simulator.should_suppress("anchor", 2)

    points = simulate_decayed_negative_evidence(
        (NegativeEvidenceEvent("one", 0, 0), NegativeEvidenceEvent("two", 1, 0)),
        max_entries=1,
    )
    assert len(points) == 2
