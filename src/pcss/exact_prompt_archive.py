"""Read-only full-prompt cache built on the sealed u32 token archive.

This is a narrow admission control, not a general suffix retriever: a hit is
usable only when the complete visible prompt equals one archived prompt and the
fixed-key candidate has one unambiguous continuation cursor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .archive import (
    REFERENCE_KEY_TOKENS,
    ArchiveCursor,
    ArchiveRecord,
    ImmutableTokenArchive,
    TokenRecord,
    build_immutable_archive,
    open_immutable_archive,
)
from .canonical import ContractError, canonical_json, require_int
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
from .trace import trace_record_fingerprint


_METADATA_KEYS = frozenset({"prompt_token_ids", "committed_token_offset", "trace_record_fingerprint"})


@dataclass(frozen=True, slots=True)
class ExactPromptHit:
    record_id: str
    prompt_token_count: int
    cursor: ArchiveCursor


def _tokens(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{field} must be a non-empty token list")
    result: list[int] = []
    for index, token in enumerate(value):
        result.append(require_int(token, f"{field}[{index}]", minimum=0))
    return tuple(result)


def _metadata(record: Mapping[str, Any], prompt: tuple[int, ...]) -> dict[str, Any]:
    return {
        "prompt_token_ids": list(prompt),
        "committed_token_offset": len(prompt),
        "trace_record_fingerprint": trace_record_fingerprint(record),
    }


def token_records_from_trace(records: Iterable[Mapping[str, Any]]) -> tuple[TokenRecord, ...]:
    """Convert complete visible trace records without adding future tokens."""

    result: list[TokenRecord] = []
    for record in records:
        content = record["content"]
        if content["committed_token_status"] != "complete":
            raise ContractError("exact-prompt archive requires complete committed token streams")
        if record["reasoning_retention"] != "exclude":
            raise ContractError("exact-prompt archive excludes hidden reasoning")
        prompt = _tokens(content["prompt_token_ids"], "trace.content.prompt_token_ids")
        committed = _tokens(content["committed_token_ids"], "trace.content.committed_token_ids")
        result.append(TokenRecord(record["record_id"], prompt + committed, _metadata(record, prompt)))
    if not result:
        raise ContractError("exact-prompt archive requires at least one record")
    return tuple(result)


class ExactPromptArchive:
    """Fail-closed full-prompt lookup over an immutable token archive."""

    def __init__(self, archive: ImmutableTokenArchive) -> None:
        self._archive = archive
        self._records = {record.record_id: record for record in archive.records}
        for record in archive.records:
            metadata = record.metadata
            if set(metadata) != _METADATA_KEYS:
                raise ContractError("exact-prompt archive record metadata is incompatible")
            prompt = _tokens(metadata["prompt_token_ids"], "archive.metadata.prompt_token_ids")
            offset = require_int(metadata["committed_token_offset"], "archive.metadata.committed_token_offset", minimum=1)
            if offset != len(prompt) or offset >= record.token_count:
                raise ContractError("exact-prompt archive record offset is invalid")
            fingerprint = metadata["trace_record_fingerprint"]
            if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
                raise ContractError("exact-prompt archive trace fingerprint is invalid")

    @property
    def archive(self) -> ImmutableTokenArchive:
        return self._archive

    def lookup(self, prompt_tokens: Iterable[int]) -> tuple[ExactPromptHit, ...]:
        prompt = tuple(require_int(token, "prompt_tokens", minimum=0) for token in prompt_tokens)
        if len(prompt) < REFERENCE_KEY_TOKENS:
            return ()
        references = self._archive.find_references(prompt[-REFERENCE_KEY_TOKENS:])
        hits: list[ExactPromptHit] = []
        for reference in references:
            record = self._records[reference.record_id]
            metadata = record.metadata
            archived_prompt = _tokens(metadata["prompt_token_ids"], "archive.metadata.prompt_token_ids")
            offset = int(metadata["committed_token_offset"])
            if archived_prompt != prompt or reference.token_offset != offset - REFERENCE_KEY_TOKENS:
                continue
            hits.append(ExactPromptHit(record.record_id, offset, self._archive.open_cursor(record.record_id, position=offset)))
        return tuple(hits)


def build_exact_prompt_archive(root: str | Path, records: Iterable[Mapping[str, Any]]) -> ExactPromptArchive:
    return ExactPromptArchive(build_immutable_archive(root, token_records_from_trace(records)))


def open_exact_prompt_archive(root: str | Path) -> ExactPromptArchive:
    return ExactPromptArchive(open_immutable_archive(root))


def build_trace_archive(
    *,
    trace_path: Path,
    record_ids: Sequence[str],
    archive_root: Path,
    raw_result_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    imported = import_jsonl(trace_path, require_uniform_fingerprints=True)
    if not record_ids or len(set(record_ids)) != len(record_ids):
        raise ContractError("exact-prompt archive requires unique selected record ids")
    by_id = {record["record_id"]: record for record in imported.records}
    missing = [record_id for record_id in record_ids if record_id not in by_id]
    if missing:
        raise ContractError(f"selected record is absent from trace: {missing[0]!r}")
    archive = build_exact_prompt_archive(archive_root, (by_id[record_id] for record_id in record_ids))
    receipt = make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.exact_prompt_archive", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "trace": file_sha256(trace_path),
            "selected_record_ids": hashlib.sha256(canonical_json(list(record_ids)).encode()).hexdigest(),
        },
        status="EXACT_PROMPT_ARCHIVE_BUILT",
        payload={
            "kind": "immutable_exact_prompt_archive/v1",
            "archive_manifest_sha256": file_sha256(archive_root / "archive-manifest.json"),
            "archive_byte_accounting": archive.archive.byte_accounting(),
            "record_count": len(record_ids),
            "not_claimed": ["runtime_speedup", "online_append", "production_storage"],
        },
    )
    write_raw_result(raw_result_path, receipt)
    write_compact_report(report_path, raw_result=raw_result_path, title="PCSS immutable exact-prompt archive")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--record-id", action="append", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    build_trace_archive(
        trace_path=parsed.trace,
        record_ids=parsed.record_id,
        archive_root=parsed.archive_root,
        raw_result_path=parsed.raw_result,
        report_path=parsed.report,
        repository_root=parsed.repository_root,
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
