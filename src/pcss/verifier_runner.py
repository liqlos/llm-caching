"""Fail-closed command boundary for a future target-backend verifier run.

This module intentionally ships no Qwen/llama.cpp adapter.  A later adapter
must be supplied as an importable factory and explicitly identify itself as a
``TARGET_BACKEND`` before a correctness receipt can be labelled local measured
evidence.  It produces no throughput or speedup metric.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError
from .manifest import validate_manifest
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .verifier import AdapterAvailability, VerifierAdapter, VerifierCase, VerifierHarness


def _case(payload: Mapping[str, Any]) -> VerifierCase:
    required = {
        "case_id",
        "draft_tokens",
        "target_tokens",
        "expected_committed_tokens",
        "expected_state_digest",
        "k",
        "include_bonus_on_full_accept",
    }
    if set(payload) != required:
        raise ContractError("verifier case has missing or unknown fields")
    return VerifierCase(
        case_id=payload["case_id"],
        draft_tokens=tuple(payload["draft_tokens"]),
        target_tokens=tuple(payload["target_tokens"]),
        expected_committed_tokens=tuple(payload["expected_committed_tokens"]),
        expected_state_digest=payload["expected_state_digest"],
        k=payload["k"],
        include_bonus_on_full_accept=payload["include_bonus_on_full_accept"],
    )


def load_cases(path: Path) -> tuple[VerifierCase, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read verifier fixture: {path}") from error
    if set(payload) != {"schema_version", "cases"} or payload["schema_version"] != "pcss.m4-verifier-cases/v1":
        raise ContractError("verifier fixture must use pcss.m4-verifier-cases/v1")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ContractError("verifier fixture must contain at least one case")
    cases = tuple(_case(item) for item in payload["cases"])
    if len({case.case_id for case in cases}) != len(cases):
        raise ContractError("verifier case ids must be unique")
    return cases


def _load_adapter(spec: str) -> VerifierAdapter:
    if spec.count(":") != 1:
        raise ContractError("adapter factory must be MODULE:CALLABLE")
    module_name, factory_name = spec.split(":", 1)
    try:
        factory: Callable[[], VerifierAdapter] = getattr(importlib.import_module(module_name), factory_name)
        adapter = factory()
    except (ImportError, AttributeError, TypeError) as error:
        raise ContractError(f"unable to load adapter factory {spec}") from error
    for required in ("name", "availability", "verify"):
        if not hasattr(adapter, required):
            raise ContractError(f"adapter from {spec} is missing {required}")
    return adapter


def _require_observed_target_manifest(manifest: Mapping[str, Any]) -> None:
    target = manifest["target"]
    if target["hardware"]["observation_status"] != "observed" or target["operating_system"]["observation_status"] != "observed":
        raise ContractError("TARGET_BACKEND receipts require observed hardware and OS manifest fields")
    for field in (
        "model_artifact",
        "quantization",
        "tokenizer",
        "chat_template",
        "generation_protocol",
        "llama_cpp",
        "context_kv",
        "benchmark",
    ):
        if manifest[field]["observation_status"] != "observed":
            raise ContractError(f"TARGET_BACKEND receipts require observed manifest.{field}")


def execute_suite(
    *,
    adapter: VerifierAdapter,
    suite: str,
    fixture_path: Path,
    manifest_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    cases = load_cases(fixture_path)
    availability: AdapterAvailability = adapter.availability()
    if availability.execution_class == "TARGET_BACKEND":
        _require_observed_target_manifest(manifest)

    harness = VerifierHarness(adapter)
    suites = {
        "k-sweep": harness.k_sweep,
        "perfect-draft": harness.perfect_draft,
        "provider-floor": harness.provider_floor,
        "forced-rejection": harness.forced_rejection_every_position,
        "long-mixed": harness.long_mixed_trajectories,
    }
    try:
        receipt = suites[suite](cases)
    except KeyError as error:
        raise ContractError(f"unknown verifier suite: {suite}") from error

    evidence_class = {
        "TARGET_BACKEND": "LOCAL_MEASURED",
        "OFFLINE_FIXTURE": "OFFLINE_FIXTURE",
        "NOT_RUN": "NOT_RUN",
    }[receipt.execution_class]
    return make_receipt(
        evidence_class=evidence_class,
        command=recorded_module_command("pcss.verifier_runner", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"fixture": file_sha256(fixture_path), "manifest": file_sha256(manifest_path)},
        status=receipt.status,
        payload=json.loads(receipt.as_json()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one future PCSS verifier suite through an external adapter.")
    parser.add_argument("--adapter", required=True, help="external MODULE:CALLABLE factory returning a VerifierAdapter")
    parser.add_argument("--suite", required=True, choices=("k-sweep", "perfect-draft", "provider-floor", "forced-rejection", "long-mixed"))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    receipt = execute_suite(
        adapter=_load_adapter(args.adapter),
        suite=args.suite,
        fixture_path=args.fixture,
        manifest_path=args.manifest,
        repository_root=args.repository_root.resolve(),
        command_arguments=command_arguments,
    )
    write_raw_result(args.output, receipt)
    if args.report is not None:
        write_compact_report(args.report, raw_result=args.output, title=f"PCSS verifier {args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
