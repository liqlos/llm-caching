"""Bounded provider-ABI microprofile with an explicit empty-call control.

The result measures Python provider boundary overhead only.  It is not attached
to target decoding and is not an end-to-end latency or speedup measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from time import monotonic_ns, perf_counter_ns
from typing import Callable, Sequence

from .canonical import ContractError
from .manifest import validate_manifest
from .metrics import clustered_bootstrap_ci
from .provider import (
    AlwaysMissProvider,
    Cursor,
    DeterministicShadowProvider,
    Fingerprints,
    NoOpProvider,
    RequestContext,
    ShadowCandidate,
    TrustedScope,
)
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .verifier import TargetHostIdentity
from .verifier_runner import observe_target_host


def _request(request_id: str) -> RequestContext:
    return RequestContext(
        request_id=request_id,
        fingerprints=Fingerprints("model", "tokenizer", "template", "greedy-protocol"),
        scope=TrustedScope("principal", "application", "repository", "revision", "generation", "block", "v1"),
        source_tiers=("current_context", "same_repository"),
        deadline_ns=monotonic_ns() + 60_000_000_000,
        probe_budget=1,
    )


def _factory(mode: str, request: RequestContext):
    if mode == "no-op":
        return NoOpProvider()
    if mode == "always-miss":
        return AlwaysMissProvider()
    if mode == "shadow-hit":
        candidate = ShadowCandidate(
            tail_tokens=(7, 8),
            tokens=(9, 10, 11, 12),
            source_class="same_repository",
            scope=request.scope,
            fingerprints=request.fingerprints,
            cursor=Cursor("segment", "sequence", 0),
            matched_suffix_tokens=2,
            provenance_family="microprofile-fixture",
        )
        return DeterministicShadowProvider((candidate,))
    raise ContractError(f"unknown provider profile mode: {mode}")


def _empty_call(*, tail_tokens: tuple[int, ...], max_tokens: int, deadline_ns: int) -> None:
    del tail_tokens, max_tokens, deadline_ns


def _profile_mode(mode: str, *, blocks: int, calls_per_block: int, warmup_calls: int) -> dict:
    if blocks < 2 or calls_per_block < 1 or warmup_calls < 0:
        raise ContractError("provider profile requires at least two non-empty blocks and non-negative warmups")
    rows: list[dict[str, float | int]] = []
    for block in range(blocks):
        request = _request(f"{mode}-{block}")
        provider = _factory(mode, request)
        provider.begin_request(request)
        deadline = request.deadline_ns
        for _ in range(warmup_calls):
            provider.propose(request.request_id, tail_tokens=(7, 8), max_tokens=4, deadline_ns=deadline)

        def measure_control() -> int:
            started = perf_counter_ns()
            for _ in range(calls_per_block):
                _empty_call(tail_tokens=(7, 8), max_tokens=4, deadline_ns=deadline)
            return perf_counter_ns() - started

        control_before_ns = measure_control()

        provider_started = perf_counter_ns()
        for _ in range(calls_per_block):
            provider.propose(request.request_id, tail_tokens=(7, 8), max_tokens=4, deadline_ns=deadline)
        provider_ns = perf_counter_ns() - provider_started
        control_after_ns = measure_control()
        provider.end_request(request.request_id)
        paired_control_ns = (control_before_ns + control_after_ns) / 2.0
        rows.append(
            {
                "block": block,
                "control_before_ns_per_call": control_before_ns / calls_per_block,
                "control_after_ns_per_call": control_after_ns / calls_per_block,
                "paired_control_ns_per_call": paired_control_ns / calls_per_block,
                "provider_ns_per_call": provider_ns / calls_per_block,
                "net_ns_per_call": (provider_ns - paired_control_ns) / calls_per_block,
            }
        )

    clusters = {f"block-{row['block']}": (float(row["net_ns_per_call"]),) for row in rows}
    interval = clustered_bootstrap_ci(clusters, bootstrap_replicates=2_000, seed=sum(ord(c) for c in mode))
    return {
        "mode": mode,
        "blocks": blocks,
        "calls_per_block": calls_per_block,
        "warmup_calls_per_block": warmup_calls,
        "raw_blocks": rows,
        "paired_control_mean_ns_per_call": fmean(float(row["paired_control_ns_per_call"]) for row in rows),
        "provider_mean_ns_per_call": fmean(float(row["provider_ns_per_call"]) for row in rows),
        "net_overhead_ci_ns_per_call": {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence_level": interval.confidence_level,
            "bootstrap_replicates": interval.bootstrap_replicates,
            "block_clusters": interval.request_clusters,
        },
    }


def run_provider_profile(
    *,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    blocks: int,
    calls_per_block: int,
    warmup_calls: int,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
) -> ResultReceipt:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    if manifest["evidence_class"] != "LOCAL_MEASURED":
        raise ContractError("provider profile requires a LOCAL_MEASURED observed manifest")
    host = target_observer()
    if manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("provider profile host differs from observed manifest")
    profiles = [
        _profile_mode(mode, blocks=blocks, calls_per_block=calls_per_block, warmup_calls=warmup_calls)
        for mode in ("no-op", "always-miss", "shadow-hit")
    ]
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.provider_profiler", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"manifest": file_sha256(manifest_path)},
        status="PROVIDER_PROFILE_VALID",
        payload={
            "kind": "provider_abi_microprofile",
            "host_identity": host.as_dict(),
            "profiles": profiles,
            "runtime_claim": "provider ABI microprofile only; not attached to model decoding",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS provider ABI microprofile")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--calls-per-block", type=int, default=1_000)
    parser.add_argument("--warmup-calls", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    run_provider_profile(
        manifest_path=args.manifest,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root,
        command_arguments=arguments,
        blocks=args.blocks,
        calls_per_block=args.calls_per_block,
        warmup_calls=args.warmup_calls,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
