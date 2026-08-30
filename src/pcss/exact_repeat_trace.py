"""Derive a visible cross-session exact-repeat trace control.

The control reuses only already committed visible tokens. It does not invoke a
model or pretend that a copied trace is a new generation; its receipt records
the source trace and the deterministic record-id selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import ContractError, canonical_json, sha256
from .importer import import_jsonl
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .trace import normalize_trace_record, trace_stream_fingerprint, validate_trace_sequence


def _selected_records(trace_path: Path, record_ids: Sequence[str]) -> list[dict[str, Any]]:
    if trace_path.suffix != ".jsonl":
        raise ContractError("exact-repeat control requires a JSONL trace")
    imported = import_jsonl(trace_path, require_uniform_fingerprints=True)
    if not imported.records:
        raise ContractError("exact-repeat control requires a non-empty trace")
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ContractError("exact-repeat control requires unique record ids")
    by_id = {record["record_id"]: dict(record) for record in imported.records}
    if len(by_id) != len(imported.records):
        raise ContractError("trace contains duplicate record ids")
    missing = [record_id for record_id in record_ids if record_id not in by_id]
    if missing:
        raise ContractError(f"exact-repeat record id is absent: {missing[0]!r}")
    records = [by_id[record_id] for record_id in record_ids]
    if any(record["content"]["committed_token_status"] != "complete" for record in records):
        raise ContractError("exact-repeat control requires complete committed tokens")
    return records


def derive_exact_repeat_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Create chronologically later, cross-session duplicates without hidden data."""

    if not records:
        raise ContractError("exact-repeat control requires at least one source record")
    source_records = [normalize_trace_record(record) for record in records]
    if len({record["record_id"] for record in source_records}) != len(source_records):
        raise ContractError("exact-repeat source record ids must be unique")
    ordered = sorted(source_records, key=lambda record: (record["chronological_index"], record["timestamp_ns"]))
    first_timestamp = min(record["timestamp_ns"] for record in ordered)
    stride = 1_000_000_000
    result: list[dict[str, Any]] = []
    for index, source in enumerate(ordered, start=1):
        copied = dict(source)
        copied["chronological_index"] = index
        copied["timestamp_ns"] = first_timestamp + index * stride
        result.append(normalize_trace_record(copied))
    for index, source in enumerate(ordered, start=1):
        target = normalize_trace_record(source)
        target["record_id"] = f"{source['record_id']}--exact-repeat-target"
        target["request_id"] = f"{source['request_id']}--exact-repeat"
        target["session_id"] = f"{source['session_id']}--exact-repeat"
        lineage = dict(target["artifact_lineage"])
        lineage["parent_revision"] = source["artifact_lineage"]["revision"]
        lineage["revision"] = f"{source['artifact_lineage']['revision']}--exact-repeat"
        lineage["path_hash"] = None
        lineage["before_blob"] = None
        lineage["after_blob"] = None
        target["artifact_lineage"] = lineage
        target["chronological_index"] = len(ordered) + index
        target["timestamp_ns"] = first_timestamp + (len(ordered) + index) * stride
        result.append(normalize_trace_record(target))
    validate_trace_sequence(result, require_uniform_fingerprints=True)
    return result


def derive_exact_repeat_trace(
    *,
    trace_path: Path,
    record_ids: Sequence[str],
    output_path: Path,
    raw_result_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    sources = _selected_records(trace_path, record_ids)
    records = derive_exact_repeat_records(sources)
    if output_path.exists():
        raise ContractError(f"refuse to overwrite exact-repeat trace: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.exact_repeat_trace", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "source_trace": file_sha256(trace_path),
            "selected_record_ids": sha256(list(record_ids)),
        },
        status="EXACT_REPEAT_TRACE_DERIVED",
        payload={
            "kind": "visible_cross_session_exact_repeat_control/v1",
            "source_record_count": len(sources),
            "derived_record_count": len(records),
            "output_trace_sha256": file_sha256(output_path),
            "output_trace_fingerprint": trace_stream_fingerprint(records),
            "hidden_reasoning": "excluded-by-source-trace-contract/v1",
            "not_claimed": ["new_model_generation", "provider_lookup", "runtime_speedup"],
        },
    )
    write_raw_result(raw_result_path, receipt)
    write_compact_report(report_path, raw_result=raw_result_path, title="PCSS exact-repeat trace control")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--record-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    parsed = _parser().parse_args(arguments)
    derive_exact_repeat_trace(
        trace_path=parsed.trace,
        record_ids=parsed.record_id,
        output_path=parsed.output,
        raw_result_path=parsed.raw_result,
        report_path=parsed.report,
        repository_root=parsed.repository_root,
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
