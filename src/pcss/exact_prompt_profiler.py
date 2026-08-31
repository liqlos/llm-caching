"""Profile a sealed exact-prompt archive without attaching it to decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, require_int
from .exact_prompt_archive import ExactPromptArchive, open_exact_prompt_archive
from .importer import import_jsonl
from .manifest import validate_manifest
from .metrics import clustered_bootstrap_ci
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


def _tokens(record: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = record["content"][field]
    if not isinstance(value, list):
        raise ContractError(f"trace {field} must be a token list")
    return tuple(require_int(token, f"trace {field}", minimum=0) for token in value)


def _profile_request(
    archive: ExactPromptArchive,
    record: Mapping[str, Any],
    *,
    k: int,
    calls: int,
    warmup_calls: int,
    expect_hit: bool,
) -> dict[str, Any]:
    prompt = _tokens(record, "prompt_token_ids")
    expected = _tokens(record, "committed_token_ids")[:k]
    if len(expected) != k:
        raise ContractError("profile target has fewer committed tokens than k")
    hits = archive.lookup(prompt)
    if expect_hit:
        if len(hits) != 1 or hits[0].cursor.read(k) != expected:
            raise ContractError("exact-prompt archive did not produce one exact continuation")
    elif hits:
        raise ContractError("exact-prompt archive miss control unexpectedly found a candidate")
    for _ in range(warmup_calls):
        warm = archive.lookup(prompt)
        if expect_hit and (len(warm) != 1 or warm[0].cursor.read(k) != expected):
            raise ContractError("exact-prompt archive warmup changed its exact continuation")
        if not expect_hit and warm:
            raise ContractError("exact-prompt archive warmup miss found a candidate")

    def control() -> int:
        started = perf_counter_ns()
        for _ in range(calls):
            _ = prompt, expected
        return perf_counter_ns() - started

    before = control()
    started = perf_counter_ns()
    for _ in range(calls):
        hit = archive.lookup(prompt)
        if expect_hit and (len(hit) != 1 or hit[0].cursor.read(k) != expected):
            raise ContractError("exact-prompt archive changed continuation during measurement")
        if not expect_hit and hit:
            raise ContractError("exact-prompt archive miss found a candidate during measurement")
    lookup_ns = perf_counter_ns() - started
    after = control()
    paired_control = (before + after) / 2.0
    return {
        "record_id": record["record_id"],
        "calls": calls,
        "warmup_calls": warmup_calls,
        "hit_count": 1 if expect_hit else 0,
        "candidate_tokens": list(expected) if expect_hit else [],
        "control_before_ns_per_call": before / calls,
        "control_after_ns_per_call": after / calls,
        "paired_control_ns_per_call": paired_control / calls,
        "lookup_and_cursor_ns_per_call": lookup_ns / calls,
        "net_lookup_and_cursor_ns_per_call": (lookup_ns - paired_control) / calls,
    }


def run_exact_prompt_profile(
    *,
    manifest_path: Path,
    trace_path: Path,
    archive_root: Path,
    target_record_ids: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    k: int,
    calls: int,
    warmup_calls: int,
    expect_hit: bool = True,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
) -> ResultReceipt:
    if k < 1 or calls < 1 or warmup_calls < 0:
        raise ContractError("exact-prompt profile k/calls/warmup must be non-negative with positive k/calls")
    if not target_record_ids or len(set(target_record_ids)) != len(target_record_ids):
        raise ContractError("exact-prompt profile requires unique target record ids")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    host = target_observer()
    if manifest["evidence_class"] != "LOCAL_MEASURED" or manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("exact-prompt profile requires the matching LOCAL_MEASURED target manifest")
    imported = import_jsonl(trace_path, require_uniform_fingerprints=True)
    by_id = {record["record_id"]: record for record in imported.records}
    missing = [record_id for record_id in target_record_ids if record_id not in by_id]
    if missing:
        raise ContractError(f"profile target record is absent: {missing[0]!r}")
    archive = open_exact_prompt_archive(archive_root)
    rows = [
        _profile_request(
            archive, by_id[record_id], k=k, calls=calls, warmup_calls=warmup_calls, expect_hit=expect_hit
        )
        for record_id in target_record_ids
    ]
    clusters = {row["record_id"]: (float(row["net_lookup_and_cursor_ns_per_call"]),) for row in rows}
    interval = clustered_bootstrap_ci(clusters, bootstrap_replicates=2_000, seed=31)
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.exact_prompt_profiler", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "manifest": file_sha256(manifest_path),
            "trace": file_sha256(trace_path),
            "archive_manifest": file_sha256(archive_root / "archive-manifest.json"),
        },
        status="EXACT_PROMPT_LOOKUP_PROFILE_VALID",
        payload={
            "kind": "immutable_exact_prompt_lookup_microprofile",
            "host_identity": host.as_dict(),
            "k": k,
            "expect_hit": expect_hit,
            "rows": rows,
            "net_lookup_and_cursor_ci_ns_per_call": {
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence_level": interval.confidence_level,
                "request_clusters": interval.request_clusters,
            },
            "archive_byte_accounting": archive.archive.byte_accounting(),
            "runtime_claim": "warm immutable lookup plus cursor-read microprofile only; not attached to model decoding or a speedup claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS immutable exact-prompt lookup microprofile")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--target-record-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--calls", type=int, default=1_000)
    parser.add_argument("--warmup-calls", type=int, default=100)
    parser.add_argument("--expect-miss", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_exact_prompt_profile(
        manifest_path=parsed.manifest,
        trace_path=parsed.trace,
        archive_root=parsed.archive_root,
        target_record_ids=parsed.target_record_id,
        output_path=parsed.output,
        report_path=parsed.report,
        repository_root=parsed.repository_root,
        command_arguments=arguments,
        k=parsed.k,
        calls=parsed.calls,
        warmup_calls=parsed.warmup_calls,
        expect_hit=not parsed.expect_miss,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
