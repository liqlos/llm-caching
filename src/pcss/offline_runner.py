"""Reproducible, model-free PCSS replay receipt command.

This command deliberately runs only chronological oracle, immutable-archive,
and logical-capacity work.  Its result is a simulator receipt, never a target
decode, verifier, timing, or speedup claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from .archive import TokenRecord, build_immutable_archive
from .capacity import (
    ActivationBundle,
    ArchiveSpanReference,
    CapacityObservation,
    CapacityPolicy,
    CapacitySimulator,
)
from .canonical import ContractError
from .experiments import compare_scope_policies
from .importer import import_jsonl
from .metrics import (
    accepted_prefix_value,
    active_capture_ratio,
    chain_survival,
    clustered_bootstrap_ci,
    opportunity,
    oracle_reuse_ratio,
    request_cluster_values,
    source_attribution,
    viable_opportunity_recovery_rate,
)
from .replay import ReplayResult, replay_chronologically
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


PCSS_MASTER_RFC_FILENAME = "PCSS_MASTER_RFC_EN.md"
PCSS_MASTER_RFC_SHA256 = "1dfdc69b055af5059021d130e216ed702a9636fb585cc5c597be783382d5e357"


def _json_value(value: Any) -> Any:
    """Convert contract dataclasses and enums without changing their values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return value


def load_trace_fixture(path: Path) -> list[dict[str, Any]]:
    """Load a visible trace fixture through the same fail-closed trace contract."""

    if path.suffix == ".jsonl":
        result = import_jsonl(path, require_uniform_fingerprints=True)
        if not result.records:
            raise ContractError("offline trace fixture must not be empty")
        return [dict(record) for record in result.records]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read trace fixture: {path}") from error
    if not isinstance(value, list) or not value:
        raise ContractError("offline trace fixture must be a non-empty JSON array")
    if not all(isinstance(record, dict) for record in value):
        raise ContractError("offline trace fixture must contain JSON objects")
    records = [normalize_trace_record(record) for record in value]
    validate_trace_sequence(records, require_uniform_fingerprints=True)
    return records


def _validate_rfc_source(path: Path, *, expected_sha256: str) -> str:
    """Bind a receipt to the versioned master RFC rather than an arbitrary file."""

    if path.name != PCSS_MASTER_RFC_FILENAME:
        raise ContractError(f"RFC source must be named {PCSS_MASTER_RFC_FILENAME}: {path}")
    if not path.is_file():
        raise ContractError(f"RFC source file is required for the receipt: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ContractError("RFC source fingerprint does not match the pinned PCSS master RFC")
    return actual_sha256


def _archive_validation(result: ReplayResult) -> dict[str, Any]:
    """Build and read one temporary sealed archive; no storage timing is collected."""

    token_records = tuple(
        TokenRecord(
            record_id=record.record_id,
            tokens=record.committed_tokens,
            metadata={
                "timestamp_ns": record.timestamp_ns,
                "chronological_index": record.chronological_index,
                "source_class": record.source_class,
                "provenance_family": record.provenance_family,
            },
        )
        for record in result.records
    )
    with TemporaryDirectory(prefix="pcss-offline-archive-") as temporary_root:
        archive = build_immutable_archive(Path(temporary_root) / "archive", token_records)
        cursor_reads = {
            record.record_id: list(archive.open_cursor(record.record_id).read(len(record.committed_tokens)))
            for record in result.records
        }
        return {
            "kind": "sealed_temporary_archive_validation_no_storage_timing",
            "record_count": len(archive.records),
            "token_count": archive.token_count,
            "reference_count": archive.reference_count,
            "posting_limit": archive.posting_limit,
            "byte_accounting": archive.byte_accounting(),
            "cursor_reads_match_input": all(
                tuple(cursor_reads[record.record_id]) == record.committed_tokens for record in result.records
            ),
        }


def _capacity_simulation(result: ReplayResult, common_archive_bytes: int) -> dict[str, Any]:
    """Run two causal logical views over the fixed replay stream, without timing."""

    bundles = tuple(
        ActivationBundle(
            bundle_id=record.record_id,
            archive_span=ArchiveSpanReference(record.record_id, 0, len(record.committed_tokens)),
            active_bytes=max(8, 4 * len(record.committed_tokens)),
            ghost_bytes=4,
        )
        for record in result.records
    )
    record_epochs = {record.record_id: index for index, record in enumerate(result.records)}
    observations: list[CapacityObservation] = []
    ordinal_by_epoch: dict[int, int] = {}
    for item in result.opportunities:
        epoch = record_epochs[item.record_id]
        ordinal = ordinal_by_epoch.get(epoch, 0)
        ordinal_by_epoch[epoch] = ordinal + 1
        persistent = tuple(
            candidate
            for candidate in item.candidates
            if candidate.source_record_id in record_epochs and candidate.tier.value
            in {
                "same_session",
                "same_repository",
                "personal_persistent_history",
                "imported_corpus",
            }
        )
        selected = min(
            persistent,
            key=lambda candidate: (
                -candidate.accepted_prefix_length,
                -candidate.matched_suffix_length,
                candidate.source_record_id,
                candidate.source_cursor,
                candidate.tier.value,
            ),
            default=None,
        )
        source_record_id = selected.source_record_id if selected is not None else item.record_id
        observations.append(
            CapacityObservation(
                observation_id=f"{item.record_id}:{item.position}",
                epoch=epoch,
                ordinal=ordinal,
                bundle_id=source_record_id,
                unique_value=item.persistent_unique_prefix if selected is not None else 0,
            )
        )
    simulator = CapacitySimulator(bundles, common_archive_bytes=common_archive_bytes)
    simulations = {
        policy.value: simulator.run_geometric(
            observations,
            first_capacity_bytes=64,
            max_capacity_bytes=256,
            policy=policy,
        ).as_dict()
        for policy in (CapacityPolicy.HISTORICAL_VALUE_DENSITY, CapacityPolicy.RECENCY_WINDOW_CONTROL)
    }
    return {
        "kind": "offline_causal_capacity_simulator_no_runtime_timing",
        "fixed_stream_observation_count": len(observations),
        "simulations": simulations,
    }


def _payload(result: ReplayResult) -> dict[str, Any]:
    archive_validation = _archive_validation(result)
    opportunities = result.opportunities
    cluster_values = request_cluster_values(opportunities)
    # No provider/verifier outcome is supplied.  These zero-recovery values
    # retain the denominators while explicitly avoiding a fictional live run.
    empty_outcomes = ()
    return {
        "kind": "model_free_chronological_replay",
        "runtime_claim": "none",
        "future_tokens_used_for_retrieval": result.future_tokens_used_for_retrieval,
        "trace_records": len(result.records),
        "proposal_positions": len(opportunities),
        "replay_census": _json_value(result.census),
        "oracle_metrics": {
            "opportunity": _json_value(opportunity(opportunities)),
            "accepted_prefix_value": accepted_prefix_value(opportunities),
            "chain_survival": _json_value(chain_survival(opportunities)),
            "source_attribution": _json_value(source_attribution(opportunities)),
            "orr": oracle_reuse_ratio(opportunities),
            "persistent_unique_opportunity": _json_value(
                opportunity(opportunities, persistent_unique=True)
            ),
            "persistent_unique_orr": oracle_reuse_ratio(opportunities, persistent_unique=True),
        },
        "clustered_confidence_interval": _json_value(
            clustered_bootstrap_ci(cluster_values, bootstrap_replicates=200, seed=0)
        ),
        "active_metrics": {
            "status": "NOT_RUN",
            "reason": "no live target decode, provider deadline, or verifier outcome was collected",
            "vorr_with_no_outcomes": _json_value(
                viable_opportunity_recovery_rate(opportunities, empty_outcomes)
            ),
            "acr_with_no_outcomes": _json_value(active_capture_ratio(opportunities, empty_outcomes)),
        },
        "scope_experiments": _json_value(
            compare_scope_policies(result, byte_budget=4096, probe_budget=2)
        ),
        "archive_validation": archive_validation,
        "capacity_simulator": _capacity_simulation(
            result, archive_validation["byte_accounting"]["total"]
        ),
    }


def execute_offline_fixture(
    *,
    fixture_path: Path,
    rfc_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    expected_rfc_sha256: str = PCSS_MASTER_RFC_SHA256,
) -> ResultReceipt:
    """Execute the host-independent replay and make one simulator receipt."""

    resolved_root = repository_root.resolve()
    rfc_fingerprint = _validate_rfc_source(rfc_path, expected_sha256=expected_rfc_sha256)
    records = load_trace_fixture(fixture_path)
    result = replay_chronologically(records)
    if result.future_tokens_used_for_retrieval:
        raise ContractError("offline replay future-leakage invariant failed")
    return make_receipt(
        evidence_class="SIMULATOR",
        command=recorded_module_command("pcss.offline_runner", command_arguments, resolved_root),
        git_revision=resolve_git_revision(resolved_root),
        input_fingerprints={
            "rfc": rfc_fingerprint,
            "trace_fixture": file_sha256(fixture_path),
        },
        status="VERIFIED_OFFLINE",
        payload={
            "trace_stream_fingerprint": trace_stream_fingerprint(records),
            **_payload(result),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a model-free chronological PCSS replay and write a simulator receipt."
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--rfc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    receipt = execute_offline_fixture(
        fixture_path=args.fixture,
        rfc_path=args.rfc,
        repository_root=args.repository_root,
        command_arguments=command_arguments,
    )
    write_raw_result(args.output, receipt)
    if args.report is not None:
        write_compact_report(args.report, raw_result=args.output, title="PCSS offline chronological replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
