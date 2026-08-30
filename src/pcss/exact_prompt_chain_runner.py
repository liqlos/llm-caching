"""Chronological exact-prompt provider-chain replay over a sealed archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import ContractError
from .exact_prompt_archive import open_exact_prompt_archive
from .exact_prompt_live_harness import _fingerprints, _scope
from .exact_prompt_provider import ExactPromptArchiveShadowProvider, ExactPromptDescriptor
from .llamacpp_live_benchmark import _pair
from .offline_runner import load_trace_fixture
from .provider import Abstention, Feedback, Proposal, RequestContext
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)


def _run_pair(archive, source: Mapping[str, Any], target: Mapping[str, Any], *, k: int) -> dict[str, Any]:
    prompt = tuple(target["content"]["prompt_token_ids"])
    authoritative = tuple(target["content"]["committed_token_ids"])
    fingerprints = _fingerprints(target)
    scope = _scope(target, widening_level=1)
    descriptor = ExactPromptDescriptor(
        tail_tokens=prompt, source_record_id=source["record_id"], source_class="same_repository",
        scope=scope, fingerprints=fingerprints, provenance_family=source["source"]["provenance_family"],
        matched_suffix_tokens=len(prompt),
    )
    provider = ExactPromptArchiveShadowProvider(archive, (descriptor,))
    request = RequestContext(
        request_id=target["request_id"], fingerprints=fingerprints, scope=scope,
        source_tiers=("same_repository",), deadline_ns=10**30, probe_budget=1,
    )
    provider.begin_request(request)
    position = 0
    rounds: list[dict[str, Any]] = []
    while position < len(authoritative):
        proposal = provider.propose(
            request.request_id, tail_tokens=prompt if position == 0 else (), max_tokens=k, deadline_ns=10**30
        )
        if isinstance(proposal, Abstention):
            rounds.append({"position": position, "status": "abstain", "reason": proposal.reason})
            break
        if not isinstance(proposal, Proposal):
            raise ContractError("exact-prompt provider returned an unsupported result")
        expected = authoritative[position : position + len(proposal.tokens)]
        accepted = 0
        while accepted < len(proposal.tokens) and proposal.tokens[accepted] == expected[accepted]:
            accepted += 1
        if accepted == len(proposal.tokens):
            committed_count = min(len(proposal.tokens) + 1, len(authoritative) - position)
            feedback_status = "all_accepted"
        else:
            committed_count = max(1, accepted + 1)
            feedback_status = "rejected"
        committed = authoritative[position : position + committed_count]
        provider.feedback(
            request.request_id,
            Feedback(
                proposal_id=proposal.proposal_id, status=feedback_status, accepted_tokens=accepted,
                committed_tokens=committed, verification_latency_ns=None, source_attribution="same_repository",
            ),
        )
        rounds.append(
            {
                "position": position, "status": feedback_status, "proposal_tokens": list(proposal.tokens),
                "accepted_tokens": accepted, "committed_tokens": list(committed),
            }
        )
        position += len(committed)
    provider.end_request(request.request_id)
    telemetry = [
        {"event": event.event, "request_id": event.request_id, "at_ns": event.at_ns, **event.payload}
        for event in provider.telemetry
    ]
    return {
        "source_record_id": source["record_id"], "target_record_id": target["record_id"],
        "request_id": target["request_id"], "rounds": rounds, "final_committed_tokens": position,
        "complete": position == len(authoritative), "telemetry": telemetry,
    }


def run_exact_prompt_chain(
    *,
    trace_path: Path,
    archive_root: Path,
    pairs: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    k: int = 2,
) -> ResultReceipt:
    if k < 1 or not pairs or len(set(pairs)) != len(pairs):
        raise ContractError("exact-prompt chain requires positive k and unique pairs")
    records = load_trace_fixture(trace_path)
    archive = open_exact_prompt_archive(archive_root)
    cases = []
    for pair_spec in pairs:
        source, target = _pair(records, pair_spec, pair_mode="exact_repeat")
        if source["chronological_index"] >= target["chronological_index"]:
            raise ContractError("provider chain source must precede target")
        cases.append(_run_pair(archive, source, target, k=k))
    if not all(case["complete"] for case in cases):
        raise ContractError("exact-prompt chain did not commit every authoritative target token")
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.exact_prompt_chain_runner", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"trace": file_sha256(trace_path), "archive_manifest": file_sha256(archive_root / "archive-manifest.json")},
        status="EXACT_PROMPT_CHAIN_VERIFIED_OFFLINE",
        payload={
            "kind": "sealed_exact_prompt_provider_chain_shadow",
            "k": k,
            "case_count": len(cases),
            "cases": cases,
            "chain_rounds": sum(len(case["rounds"]) for case in cases),
            "future_tokens_used_for_retrieval": False,
            "runtime_claim": "post-hoc provider chain replay only; no target output was changed and no speedup is claimed",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS sealed exact-prompt provider chain")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_exact_prompt_chain(
        trace_path=parsed.trace, archive_root=parsed.archive_root, pairs=parsed.pair,
        output_path=parsed.output, report_path=parsed.report, repository_root=parsed.repository_root,
        command_arguments=arguments, k=parsed.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
