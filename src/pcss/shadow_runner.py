"""Run the provider ABI in post-hoc shadow mode over chronological traces."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .canonical import ContractError
from .metrics import clustered_bootstrap_ci
from .offline_runner import load_trace_fixture
from .provider import (
    Abstention,
    Cursor,
    DeterministicShadowProvider,
    Feedback,
    Fingerprints,
    Proposal,
    RequestContext,
    ShadowCandidate,
    TrustedScope,
)
from .replay import Candidate, ReplayRecord, SourceTier, continuation_prefix_length, replay_chronologically
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)


_PERSISTENT_TIERS = (
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
)
_TIER_RANK = {tier: index for index, tier in enumerate(_PERSISTENT_TIERS)}


def _rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    return (
        -candidate.matched_suffix_length,
        _TIER_RANK[candidate.tier],
        candidate.source_token_bytes,
        candidate.source_record_id,
        candidate.source_cursor,
    )


def _fingerprints(record: ReplayRecord) -> Fingerprints:
    return Fingerprints(
        model=record.fingerprints.model,
        tokenizer=record.fingerprints.tokenizer,
        chat_template=record.fingerprints.chat_template,
        generation_protocol=record.fingerprints.generation_protocol,
    )


def _scope(record: ReplayRecord, *, widening_level: int = 4) -> TrustedScope:
    scope = record.scope
    return TrustedScope(
        principal_namespace=scope.principal_namespace or "unknown",
        application=scope.application_id or "unknown",
        repository=scope.repository_id,
        revision_family=scope.revision_family,
        operation_class=scope.workflow_stage or "unknown",
        workflow_block=scope.semantic_block_id,
        schema_or_tool_version=None,
        widening_level=widening_level,
    )


def _candidate_scope(candidate: Candidate) -> TrustedScope:
    scope = candidate.source_scope
    return TrustedScope(
        principal_namespace=scope.principal_namespace or "unknown",
        application=scope.application_id or "unknown",
        repository=scope.repository_id,
        revision_family=scope.revision_family,
        operation_class=scope.workflow_stage or "unknown",
        workflow_block=scope.semantic_block_id,
        schema_or_tool_version=None,
        widening_level=_TIER_RANK[candidate.tier],
    )


def run_shadow(
    *,
    trace_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    min_suffix: int = 4,
    k: int = 2,
) -> ResultReceipt:
    if min_suffix < 1 or k < 1:
        raise ContractError("shadow min_suffix and k must be positive")
    records = load_trace_fixture(trace_path)
    replay = replay_chronologically(records, min_suffix=min_suffix, max_continuation=k)
    record_by_id = {record.record_id: record for record in replay.records}
    opportunities_by_record: dict[str, list[Any]] = defaultdict(list)
    for opportunity in replay.opportunities:
        opportunities_by_record[opportunity.record_id].append(opportunity)

    proposed_positions = 0
    accepted_positions = 0
    accepted_tokens = 0
    abstentions = 0
    persistent_oracle_positions = 0
    persistent_oracle_tokens = 0
    cluster_values: dict[str, tuple[float, ...]] = {}
    source_tokens: dict[str, int] = defaultdict(int)
    telemetry: list[dict[str, Any]] = []

    for record_id, opportunities in opportunities_by_record.items():
        target = record_by_id[record_id]
        selected_by_position: dict[int, Candidate] = {}
        shadow_candidates: list[ShadowCandidate] = []
        candidate_by_tail: dict[tuple[int, ...], Candidate] = {}
        for opportunity in opportunities:
            persistent = [candidate for candidate in opportunity.candidates if candidate.tier in _PERSISTENT_TIERS]
            if persistent:
                persistent_oracle_positions += int(max(candidate.accepted_prefix_length for candidate in persistent) > 0)
                persistent_oracle_tokens += max(candidate.accepted_prefix_length for candidate in persistent)
            eligible = [candidate for candidate in persistent if candidate.matched_suffix_length >= min_suffix]
            if not eligible:
                continue
            selected = min(eligible, key=_rank)
            query = target.prompt_tokens + target.committed_tokens[: opportunity.position]
            tail = query[-selected.matched_suffix_length :]
            if tail in candidate_by_tail:
                selected_by_position[opportunity.position] = candidate_by_tail[tail]
                continue
            candidate_by_tail[tail] = selected
            selected_by_position[opportunity.position] = selected
            shadow_candidates.append(
                ShadowCandidate(
                    tail_tokens=tail,
                    tokens=selected.continuation[:k],
                    source_class=selected.tier.value,
                    scope=_candidate_scope(selected),
                    fingerprints=_fingerprints(target),
                    cursor=Cursor("chronological-replay", selected.source_record_id, selected.source_cursor),
                    matched_suffix_tokens=selected.matched_suffix_length,
                    score=float(selected.matched_suffix_length),
                    ambiguity=sum(
                        1 for candidate in eligible if candidate.matched_suffix_length == selected.matched_suffix_length
                    ),
                    provenance_family=selected.provenance_family,
                )
            )
        provider = DeterministicShadowProvider(tuple(shadow_candidates))
        request_id = target.request_id or target.record_id
        provider.begin_request(
            RequestContext(
                request_id=request_id,
                fingerprints=_fingerprints(target),
                scope=_scope(target),
                source_tiers=tuple(tier.value for tier in _PERSISTENT_TIERS),
                deadline_ns=10**30,
                probe_budget=1,
            )
        )
        cluster_accepted = 0
        cluster_proposed = 0
        for opportunity in opportunities:
            selected = selected_by_position.get(opportunity.position)
            if selected is None:
                abstentions += 1
                continue
            query = target.prompt_tokens + target.committed_tokens[: opportunity.position]
            tail = query[-selected.matched_suffix_length :]
            result = provider.propose(request_id, tail_tokens=tail, max_tokens=k, deadline_ns=10**30)
            if isinstance(result, Abstention):
                abstentions += 1
                continue
            if not isinstance(result, Proposal):
                raise ContractError("shadow provider returned an unsupported result")
            proposed_positions += 1
            cluster_proposed += 1
            accepted = continuation_prefix_length(
                target.committed_tokens[opportunity.position :],
                result.tokens,
            )
            accepted_tokens += accepted
            cluster_accepted += accepted
            accepted_positions += int(accepted > 0)
            source_tokens[result.source_class] += accepted
            provider.feedback(
                request_id,
                Feedback(
                    proposal_id=result.proposal_id,
                    status="all_accepted" if accepted == len(result.tokens) else "rejected",
                    accepted_tokens=accepted,
                    committed_tokens=result.tokens,
                    verification_latency_ns=None,
                    source_attribution=result.source_class,
                ),
            )
        provider.end_request(request_id)
        telemetry.extend(
            {"event": event.event, "request_id": event.request_id, "at_ns": event.at_ns, **event.payload}
            for event in provider.telemetry
        )
        cluster_values[request_id] = (cluster_accepted / cluster_proposed if cluster_proposed else 0.0,)

    interval = clustered_bootstrap_ci(cluster_values, bootstrap_replicates=2_000, seed=17)
    analysis = {
        "min_suffix": min_suffix,
        "k": k,
        "positions": len(replay.opportunities),
        "proposed_positions": proposed_positions,
        "accepted_positions": accepted_positions,
        "accepted_tokens": accepted_tokens,
        "abstentions": abstentions,
        "proposal_acceptance_rate": accepted_positions / proposed_positions if proposed_positions else 0.0,
        "mean_accepted_tokens_per_proposal": accepted_tokens / proposed_positions if proposed_positions else 0.0,
        "persistent_oracle_positions": persistent_oracle_positions,
        "persistent_oracle_tokens": persistent_oracle_tokens,
        "vorr": accepted_positions / persistent_oracle_positions if persistent_oracle_positions else 0.0,
        "acr": accepted_tokens / persistent_oracle_tokens if persistent_oracle_tokens else 0.0,
        "accepted_tokens_by_source": dict(sorted(source_tokens.items())),
        "clustered_mean_accepted_tokens_per_proposal_ci": {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence_level": interval.confidence_level,
            "request_clusters": interval.request_clusters,
        },
    }
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.shadow_runner", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"trace": file_sha256(trace_path)},
        status="VERIFIED_OFFLINE",
        payload={
            "kind": "provider_abi_posthoc_shadow",
            "analysis": analysis,
            "telemetry": telemetry,
            "future_tokens_used_for_retrieval": replay.future_tokens_used_for_retrieval,
            "runtime_claim": "post-hoc shadow acceptance only; target output was not changed and no speedup is claimed",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS provider ABI post-hoc shadow")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--min-suffix", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    run_shadow(
        trace_path=args.trace,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root.resolve(),
        command_arguments=arguments,
        min_suffix=args.min_suffix,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
