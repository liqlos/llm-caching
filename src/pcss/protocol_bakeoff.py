"""Offline protocol bakeoff over target-generated chronological edit traces."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import ContractError
from .experiments import compare_edit_protocols
from .metrics import clustered_bootstrap_ci
from .offline_runner import load_trace_fixture
from .replay import group_opportunities_by_record, replay_chronologically
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)


def _record(records: Sequence[Mapping[str, Any]], record_id: str) -> Mapping[str, Any]:
    matches = [record for record in records if record["record_id"] == record_id]
    if len(matches) != 1:
        raise ContractError(f"trace must contain exactly one record_id {record_id!r}")
    return matches[0]


def run_protocol_bakeoff(
    *,
    trace_path: Path,
    pairs: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    max_continuation: int = 32,
) -> ResultReceipt:
    if not pairs or len(pairs) != len(set(pairs)):
        raise ContractError("protocol bakeoff requires unique source:target pairs")
    records = load_trace_fixture(trace_path)
    replay = replay_chronologically(records, min_suffix=1, max_continuation=max_continuation)
    opportunities = group_opportunities_by_record(replay)
    cases = []
    coverage_by_protocol: dict[str, dict[str, list[float]]] = {}
    for pair_spec in pairs:
        parts = pair_spec.split(":")
        if len(parts) != 2 or not all(parts):
            raise ContractError("protocol pair must be SOURCE_RECORD_ID:TARGET_RECORD_ID")
        source, target = (_record(records, record_id) for record_id in parts)
        if source["chronological_index"] >= target["chronological_index"]:
            raise ContractError("protocol source must precede target chronologically")
        if source["trusted_scope"]["repository_id"] != target["trusted_scope"]["repository_id"]:
            raise ContractError("protocol pair must remain in one repository")
        before = source["content"]["committed_token_ids"]
        after = target["content"]["committed_token_ids"]
        comparison = compare_edit_protocols(
            before,
            after,
            persistent_opportunities=opportunities.get(target["record_id"], ()),
            max_continuation=max_continuation,
        )
        values = [asdict(value) for value in comparison.values]
        for value in values:
            coverage_by_protocol.setdefault(value["protocol"], {})[target["request_id"]] = [
                value["directly_covered_output_tokens"] / comparison.output_tokens
            ]
        cases.append(
            {
                "source_record_id": source["record_id"],
                "target_record_id": target["record_id"],
                "request_id": target["request_id"],
                "output_tokens": comparison.output_tokens,
                "protocol_values": values,
                "current_source": asdict(comparison.current_source),
                "patch": asdict(comparison.patch),
            }
        )
    coverage = {
        protocol: asdict(clustered_bootstrap_ci(values_by_request))
        for protocol, values_by_request in sorted(coverage_by_protocol.items())
    }
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.protocol_bakeoff", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"trace": file_sha256(trace_path)},
        status="VERIFIED_OFFLINE",
        payload={
            "kind": "chronological_edit_protocol_bakeoff",
            "case_count": len(cases),
            "cases": cases,
            "direct_output_coverage_clustered_ci": coverage,
            "full_regeneration_baseline_tokens": sum(case["output_tokens"] for case in cases),
            "oracle_boundary": (
                "patch operations and accepted-prefix values use completed target output only for offline scoring; "
                "they are not an online proposer or a speedup measurement"
            ),
            "persistent_credit_rule": "persistent_cross_session receives only persistent_unique_prefix after stronger source tiers",
            "runtime_claim": "none; model-free chronological protocol comparison",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS edit protocol bakeoff")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-continuation", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_protocol_bakeoff(
        trace_path=parsed.trace,
        pairs=parsed.pair,
        output_path=parsed.output,
        report_path=parsed.report,
        repository_root=parsed.repository_root,
        command_arguments=arguments,
        max_continuation=parsed.max_continuation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
