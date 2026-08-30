"""Derive a fail-closed verifier-economics verdict from measured receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from .canonical import ContractError, require_exact_keys, require_mapping
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


_RECEIPT_FIELDS = {
    "schema_version",
    "evidence_class",
    "command",
    "git_revision",
    "input_fingerprints",
    "status",
    "payload",
}


def _read_receipt(path: Path, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read {field} receipt") from error
    payload = require_mapping(payload, f"{field} receipt")
    require_exact_keys(payload, _RECEIPT_FIELDS, f"{field} receipt")
    return dict(payload)


def _provider_upper_ns(receipt: Mapping[str, Any]) -> float:
    if receipt["evidence_class"] != "LOCAL_MEASURED" or receipt["status"] != "PROVIDER_PROFILE_VALID":
        raise ContractError("provider receipt is not a valid local measurement")
    payload = require_mapping(receipt["payload"], "provider receipt.payload")
    if payload.get("kind") != "provider_abi_microprofile":
        raise ContractError("provider receipt has an unsupported kind")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ContractError("provider receipt profiles are invalid")
    matches = [item for item in profiles if isinstance(item, dict) and item.get("mode") == "always-miss"]
    if len(matches) != 1:
        raise ContractError("provider receipt must contain exactly one always-miss profile")
    interval = require_mapping(matches[0].get("net_overhead_ci_ns_per_call"), "always-miss interval")
    upper = interval.get("upper")
    if type(upper) not in {int, float} or not 0 <= float(upper):
        raise ContractError("always-miss upper confidence bound is invalid")
    return float(upper)


def _derive(
    profile_receipt: Mapping[str, Any],
    verifier_receipt: Mapping[str, Any],
    provider_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    if profile_receipt["evidence_class"] != "LOCAL_MEASURED" or profile_receipt["status"] != "PROFILE_VALID":
        raise ContractError("verifier profile receipt is not a valid local measurement")
    if verifier_receipt["evidence_class"] != "LOCAL_MEASURED" or verifier_receipt["status"] != "VERIFIED":
        raise ContractError("verifier correctness receipt is not verified local evidence")
    if profile_receipt["git_revision"] != verifier_receipt["git_revision"]:
        raise ContractError("verifier profile and correctness receipts use different revisions")
    profile_payload = require_mapping(profile_receipt["payload"], "verifier profile receipt.payload")
    verifier_payload = require_mapping(verifier_receipt["payload"], "verifier receipt.payload")
    provider_payload = require_mapping(provider_receipt["payload"], "provider receipt.payload")
    if profile_payload.get("host_identity") != provider_payload.get("host_identity"):
        raise ContractError("verifier and provider profiles use different target hosts")
    if verifier_payload.get("execution_class") != "TARGET_BACKEND":
        raise ContractError("verifier correctness receipt is not target-backend evidence")
    raw_profile = require_mapping(profile_payload.get("raw_profile"), "verifier profile raw payload")
    profiles = raw_profile.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ContractError("verifier profile contains no measured K rows")
    provider_upper = _provider_upper_ns(provider_receipt)

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        profile = require_mapping(profile, "verifier profile row")
        k = profile.get("k")
        measurements = profile.get("measurements")
        if type(k) is not int or k < 1 or not isinstance(measurements, list) or len(measurements) < 2:
            raise ContractError("verifier profile row is invalid")
        batch_only_clusters: dict[str, tuple[float, ...]] = {}
        canonical_clusters: dict[str, tuple[float, ...]] = {}
        serial_values: list[float] = []
        batch_values: list[float] = []
        for index, measurement in enumerate(measurements):
            measurement = require_mapping(measurement, "verifier profile measurement")
            serial = fmean((float(measurement["serial_before_ns"]), float(measurement["serial_after_ns"])))
            batch = float(measurement["batch_verify_ns"])
            if not serial > 0 or not batch > 0:
                raise ContractError("verifier profile timings must be positive")
            serial_values.append(serial)
            batch_values.append(batch)
            batch_only_clusters[f"iteration-{index}"] = (serial / (batch + provider_upper),)
            canonical_clusters[f"iteration-{index}"] = (serial / (batch + serial + provider_upper),)
        batch_interval = clustered_bootstrap_ci(batch_only_clusters, bootstrap_replicates=2_000, seed=k)
        canonical_interval = clustered_bootstrap_ci(canonical_clusters, bootstrap_replicates=2_000, seed=10_000 + k)
        rows.append(
            {
                "k": k,
                "paired_iteration_clusters": len(measurements),
                "serial_mean_ns": fmean(serial_values),
                "batch_verify_mean_ns": fmean(batch_values),
                "always_miss_provider_upper_95_ns": provider_upper,
                "batch_only_perfect_draft_ratio_ci": {
                    "estimate": batch_interval.estimate,
                    "lower": batch_interval.lower,
                    "upper": batch_interval.upper,
                    "confidence_level": batch_interval.confidence_level,
                },
                "current_serial_canonical_commit_ratio_ci": {
                    "estimate": canonical_interval.estimate,
                    "lower": canonical_interval.lower,
                    "upper": canonical_interval.upper,
                    "confidence_level": canonical_interval.confidence_level,
                },
            }
        )
    no_go = all(row["current_serial_canonical_commit_ratio_ci"]["upper"] < 1.0 for row in rows)
    if not no_go:
        raise ContractError("canonical-commit no-go invariant unexpectedly failed")
    return {
        "decision": "NO_GO_CURRENT_SERIAL_CANONICAL_COMMIT",
        "rows": rows,
        "reason": (
            "the verified exact-state path performs batch verification and then serially reconstructs every committed "
            "token; even perfect drafts therefore cost T_batch + T_serial before other runtime overhead"
        ),
        "not_claimed": [
            "live_pcss_speedup",
            "profitable_k",
            "workload_acceptance_distribution",
            "thermal_qualification",
            "transfer_to_other_models_or_backends",
        ],
    }


def run_economics(
    *,
    profile_path: Path,
    verifier_path: Path,
    provider_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    profile = _read_receipt(profile_path, "verifier profile")
    verifier = _read_receipt(verifier_path, "verifier correctness")
    provider = _read_receipt(provider_path, "provider profile")
    analysis = _derive(profile, verifier, provider)
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.economics", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "provider_profile": file_sha256(provider_path),
            "verifier_correctness": file_sha256(verifier_path),
            "verifier_profile": file_sha256(profile_path),
        },
        status=analysis["decision"],
        payload={"kind": "derived_verifier_economics", "analysis": analysis},
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS verifier economics")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    run_economics(
        profile_path=args.profile,
        verifier_path=args.verifier,
        provider_path=args.provider,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root,
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
