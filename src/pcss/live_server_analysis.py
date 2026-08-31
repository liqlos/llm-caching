"""Strict offline analysis of a frozen live-server benchmark receipt.

This module never launches a server, changes a benchmark protocol, or searches
for a favourable slice.  It accepts only the complete raw matrix produced by
``pcss.live_server_benchmark`` and writes one new immutable JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable, Mapping, Sequence

from .live_server_benchmark import (
    ARM_NAMES,
    CAPTURE_RECEIPT_SCHEMA_V1,
    CAPTURE_RECEIPT_SCHEMA_V2,
    CAPTURE_SOURCE_COMMITMENT_SCHEMA,
    CURRENT_SOURCE_MATCHING_MODE,
    CURRENT_SOURCE_MISMATCH_CONTROL_MODE,
    CURRENT_SOURCE_REQUEST_KEY,
    REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
    RESULT_SCHEMA,
    _CAPTURE_RECEIPT_V1_KEYS,
    _CAPTURE_RECEIPT_V2_KEYS,
    _CAPTURE_SOURCE_COMMITMENT_KEYS,
    _canonical_json_sha256,
    _current_source_metric_attestation,
    _exact_output_gate,
    _policy_metric_attestation,
    _policy_binding,
    _validate_manifest,
    _validate_redacted_current_source_workload,
    _validate_workload,
)


ANALYSIS_SCHEMA = "pcss.live-server-benchmark-analysis/v1"
BOOTSTRAP_REPLICATES = 10_000
_RATIO_DIRECTION = "baseline_over_arm_gt_1_is_faster"
_MISS_METRICS = (
    "pcss_lookups_total",
    "pcss_persistent_hits_total",
    "pcss_current_context_hits_total",
    "pcss_accepted_tokens_total",
    "pcss_persistent_misses_total",
    "pcss_abstentions_total",
)
_CURRENT_SOURCE_MISS_METRICS = (
    "pcss_current_source_lookups_total",
    "pcss_current_source_hits_total",
    "pcss_current_source_rejections_total",
)


class AnalysisContractError(ValueError):
    """Raised when raw evidence cannot support the frozen analysis."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_digest(value: object, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnalysisContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AnalysisContractError(f"{label} keys differ from schema")
    return value


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AnalysisContractError(f"{label} must be a positive integer")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisContractError(f"cannot read {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisContractError(f"{label} must be a JSON object")
    return value, raw


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        qualifier = "positive " if positive else "non-negative "
        raise AnalysisContractError(f"{label} must be a finite {qualifier}number")
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisContractError("cannot compute a percentile of no observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _clustered_bootstrap(
    observations: Sequence[tuple[str, str, float, float]],
    *,
    seed: int,
    statistic: Callable[[Sequence[tuple[str, str, float, float]]], float],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    """Resample declared clusters, retaining every observation in each draw."""

    if replicates < 1:
        raise AnalysisContractError("bootstrap replicates must be positive")
    by_cluster: dict[str, list[tuple[str, str, float, float]]] = {}
    for row in observations:
        by_cluster.setdefault(row[1], []).append(row)
    cluster_ids = sorted(by_cluster)
    if not cluster_ids:
        raise AnalysisContractError("clustered analysis requires observations")
    generator = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(replicates):
        drawn: list[tuple[str, str, float, float]] = []
        for _ in cluster_ids:
            drawn.extend(by_cluster[generator.choice(cluster_ids)])
        bootstrapped.append(statistic(drawn))
    return bootstrapped


def _ratio_statistic(rows: Sequence[tuple[str, str, float, float]]) -> float:
    numerator = sum(row[2] for row in rows)
    denominator = sum(row[3] for row in rows)
    if denominator <= 0:
        raise AnalysisContractError("ratio denominator must be positive")
    return numerator / denominator


def _paired_ratio(
    observations: Sequence[tuple[str, str, float, float]],
    *,
    seed: int,
    direction: str,
) -> dict[str, Any]:
    bootstrapped = _clustered_bootstrap(
        observations,
        seed=seed,
        statistic=_ratio_statistic,
    )
    return {
        "direction": direction,
        "estimate": _ratio_statistic(observations),
        "ci95": {
            "lower": _percentile(bootstrapped, 0.025),
            "upper": _percentile(bootstrapped, 0.975),
        },
        "pair_count": len(observations),
        "cluster_count": len({row[1] for row in observations}),
        "confidence_method": "deterministic_clustered_bootstrap_percentile_95",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _descriptive(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "median": median(values) if values else None,
        "p95": _percentile(values, 0.95) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _prometheus(receipt: object, label: str) -> dict[str, float]:
    if not isinstance(receipt, dict) or receipt.get("status") != 200 or receipt.get("error") is not None:
        raise AnalysisContractError(f"{label} is not a successful metrics receipt")
    body = receipt.get("body")
    if not isinstance(body, str):
        raise AnalysisContractError(f"{label}.body must be text")
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line.startswith("llamacpp:"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        name = fields[0][len("llamacpp:") :]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value):
            values[name] = value
    return values


def _is_miss_heavy(sample: Mapping[str, Any]) -> bool:
    before = _prometheus(sample.get("metrics_before"), "sample.metrics_before")
    after = _prometheus(sample.get("metrics_after"), "sample.metrics_after")
    if any(name not in before or name not in after for name in _MISS_METRICS):
        return False
    deltas = {name: after[name] - before[name] for name in _MISS_METRICS}
    if any(value < 0 for value in deltas.values()):
        raise AnalysisContractError("PCSS metrics counters decreased within a request")
    source_present = {
        name for name in _CURRENT_SOURCE_MISS_METRICS if name in before or name in after
    }
    if source_present and source_present != set(_CURRENT_SOURCE_MISS_METRICS):
        raise AnalysisContractError("source metrics are incomplete within a request")
    if source_present:
        source_deltas = {
            name: after[name] - before[name] for name in _CURRENT_SOURCE_MISS_METRICS
        }
        if any(value < 0 for value in source_deltas.values()):
            raise AnalysisContractError("current-source metrics counters decreased within a request")
        if source_deltas["pcss_current_source_lookups_total"] > 0:
            # A current-source attempt belongs in the overhead population when
            # it produced no exact hit or its target verification rejected a
            # source draft. This is intentionally independent of whether a
            # later persistent/current-context fallback happened to help.
            return (
                source_deltas["pcss_current_source_hits_total"] == 0
                or source_deltas["pcss_current_source_rejections_total"] > 0
            )
    # The prospective four-arm protocol defines this population by absence of
    # a persistent hit, regardless of whether the optional current-context
    # tier supplied an accepted draft.
    return deltas["pcss_lookups_total"] > 0 and deltas["pcss_persistent_hits_total"] == 0


def _metric_delta(sample: Mapping[str, Any], metric: str) -> float:
    before = _prometheus(sample.get("metrics_before"), "sample.metrics_before")
    after = _prometheus(sample.get("metrics_after"), "sample.metrics_after")
    if metric not in before or metric not in after:
        raise AnalysisContractError(f"sample lacks required metric: {metric}")
    delta = after[metric] - before[metric]
    if delta < 0:
        raise AnalysisContractError(f"PCSS metric counter decreased within a request: {metric}")
    return delta


def _suffix_ladder_activation(
    samples: Sequence[Mapping[str, Any]],
    cluster_ids: Mapping[str, str],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that an observed effect came from the 32-token policy rung."""

    required_names = {
        "persistent_32_hit_requests_min",
        "persistent_32_hit_episodes_min",
        "persistent_32_accepted_tokens_min",
        "persistent_64_hit_requests_max",
        "persistent_32_suppressions_max",
        "fallback_disable_events_max",
    }
    if not required_names <= set(gates):
        return {"required": False, "passed": True}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        deltas = {
            "persistent_32_hits": _metric_delta(
                sample, "pcss_persistent_32_hits_total"
            ),
            "persistent_32_accepted_tokens": _metric_delta(
                sample, "pcss_persistent_32_accepted_tokens_total"
            ),
            "persistent_64_hits": _metric_delta(
                sample, "pcss_persistent_64_hits_total"
            ),
            "persistent_32_suppressions": _metric_delta(
                sample, "pcss_persistent_32_suppressions_total"
            ),
            "fallback_disable_events": _metric_delta(
                sample, "pcss_fallback_disable_events_total"
            ),
        }
        rows.append(
            {
                "request_id": sample["request_id"],
                "episode_id": cluster_ids[sample["request_id"]],
                **deltas,
            }
        )
    short_hit_rows = [row for row in rows if row["persistent_32_hits"] > 0]
    short_hit_episodes = {row["episode_id"] for row in short_hit_rows}
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "persistent_32_accepted_tokens",
            "persistent_64_hits",
            "persistent_32_suppressions",
            "fallback_disable_events",
        )
    }
    gates_passed = {
        "persistent_32_hit_requests": len(short_hit_rows)
        >= gates["persistent_32_hit_requests_min"],
        "persistent_32_hit_episodes": len(short_hit_episodes)
        >= gates["persistent_32_hit_episodes_min"],
        "persistent_32_accepted_tokens": totals["persistent_32_accepted_tokens"]
        >= gates["persistent_32_accepted_tokens_min"],
        "persistent_64_hit_requests": sum(
            row["persistent_64_hits"] > 0 for row in rows
        )
        <= gates["persistent_64_hit_requests_max"],
        "persistent_32_suppressions": totals["persistent_32_suppressions"]
        <= gates["persistent_32_suppressions_max"],
        "fallback_disable_events": totals["fallback_disable_events"]
        <= gates["fallback_disable_events_max"],
    }
    return {
        "required": True,
        "rows": rows,
        "short_hit_request_count": len(short_hit_rows),
        "short_hit_episode_count": len(short_hit_episodes),
        "totals": totals,
        "gates": {
            "persistent_32_hit_requests_min": gates[
                "persistent_32_hit_requests_min"
            ],
            "persistent_32_hit_episodes_min": gates[
                "persistent_32_hit_episodes_min"
            ],
            "persistent_32_accepted_tokens_min": gates[
                "persistent_32_accepted_tokens_min"
            ],
            "persistent_64_hit_requests_max": gates[
                "persistent_64_hit_requests_max"
            ],
            "persistent_32_suppressions_max": gates[
                "persistent_32_suppressions_max"
            ],
            "fallback_disable_events_max": gates["fallback_disable_events_max"],
        },
        "gate_verdicts": gates_passed,
        "passed": all(gates_passed.values()),
    }


def _current_source_activation(
    samples: Sequence[Mapping[str, Any]],
    workload: Sequence[Mapping[str, Any]],
    cluster_ids: Mapping[str, str],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove source attachment, target acceptance, index construction, and control abstention."""

    required_names = {
        "current_source_matching_requests_min",
        "current_source_lookup_requests_min",
        "current_source_hit_requests_min",
        "current_source_hit_episodes_min",
        "current_source_accepted_tokens_min",
        "current_source_peak_index_bytes_min",
        "current_source_rejections_max",
        "current_source_ambiguities_max",
        "current_source_abstentions_max",
        "current_source_mismatch_control_requests_min",
        "current_source_mismatch_abstentions_min",
    }
    if not required_names <= set(gates):
        raise AnalysisContractError("source-redacted workload lacks activation gates")
    modes = {
        row["request_id"]: row.get("pcss_current_source_commitment", {}).get(
            "attachment_mode"
        )
        for row in workload
    }
    rows: list[dict[str, Any]] = []
    metric_names = {
        "lookups": "pcss_current_source_lookups_total",
        "hits": "pcss_current_source_hits_total",
        "draft_tokens": "pcss_current_source_draft_tokens_total",
        "accepted_tokens": "pcss_current_source_accepted_tokens_total",
        "rejections": "pcss_current_source_rejections_total",
        "ambiguities": "pcss_current_source_ambiguities_total",
        "abstentions": "pcss_current_source_abstentions_total",
        "peak_index_bytes": "pcss_current_source_peak_index_bytes",
        "mismatch_abstentions": "pcss_current_source_mismatch_abstentions_total",
    }
    for sample in samples:
        request_id = sample["request_id"]
        mode = modes.get(request_id)
        deltas = {
            name: _metric_delta(sample, metric)
            for name, metric in metric_names.items()
        }
        rows.append(
            {
                "request_id": request_id,
                "episode_id": cluster_ids[request_id],
                "attachment_mode": mode,
                **deltas,
            }
        )
    matching_rows = [
        row
        for row in rows
        if row["attachment_mode"] == CURRENT_SOURCE_MATCHING_MODE
    ]
    mismatch_rows = [
        row
        for row in rows
        if row["attachment_mode"] == CURRENT_SOURCE_MISMATCH_CONTROL_MODE
    ]
    lookup_rows = [row for row in matching_rows if row["lookups"] > 0]
    hit_rows = [row for row in matching_rows if row["hits"] > 0]
    totals = {
        key: sum(row[key] for row in matching_rows)
        for key in (
            "draft_tokens",
            "accepted_tokens",
            "rejections",
            "ambiguities",
            "abstentions",
            "peak_index_bytes",
        )
    }
    mismatch_abstentions = sum(row["mismatch_abstentions"] for row in mismatch_rows)
    gate_verdicts = {
        "matching_requests": len(matching_rows)
        >= gates["current_source_matching_requests_min"],
        "lookup_requests": len(lookup_rows)
        >= gates["current_source_lookup_requests_min"],
        "hit_requests": len(hit_rows)
        >= gates["current_source_hit_requests_min"],
        "hit_episodes": len({row["episode_id"] for row in hit_rows})
        >= gates["current_source_hit_episodes_min"],
        "accepted_tokens": totals["accepted_tokens"]
        >= gates["current_source_accepted_tokens_min"],
        "peak_index_bytes": totals["peak_index_bytes"]
        >= gates["current_source_peak_index_bytes_min"],
        "rejections": totals["rejections"]
        <= gates["current_source_rejections_max"],
        "ambiguities": totals["ambiguities"]
        <= gates["current_source_ambiguities_max"],
        "abstentions": totals["abstentions"]
        <= gates["current_source_abstentions_max"],
        "mismatch_control_requests": len(mismatch_rows)
        >= gates["current_source_mismatch_control_requests_min"],
        "mismatch_abstentions": mismatch_abstentions
        >= gates["current_source_mismatch_abstentions_min"],
    }
    return {
        "required": True,
        "rows": rows,
        "matching_request_count": len(matching_rows),
        "lookup_request_count": len(lookup_rows),
        "hit_request_count": len(hit_rows),
        "hit_episode_count": len({row["episode_id"] for row in hit_rows}),
        "mismatch_control_request_count": len(mismatch_rows),
        "matching_totals": totals,
        "mismatch_abstentions": mismatch_abstentions,
        "gates": {key: gates[key] for key in sorted(required_names)},
        "gate_verdicts": gate_verdicts,
        "passed": all(gate_verdicts.values()),
    }


def _archived_workload(
    document: Mapping[str, Any], raw: bytes, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str, dict[str, Any], bool]:
    """Load either a legacy workload or a source-safe archival receipt.

    A redacted receipt cannot reproduce the original private input bytes. Its
    original digest is therefore an asserted manifest binding, not a newly
    calculated claim. Legacy studies continue to use their byte-for-byte
    workload copy and retain the historical validation path.
    """

    file_sha256 = _sha256_bytes(raw)
    schema_version = document.get("schema_version")
    if schema_version != REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA:
        workload = _validate_workload(document)
        if any(CURRENT_SOURCE_REQUEST_KEY in row["body"] for row in workload):
            raise AnalysisContractError(
                "source-bearing workload must be archived as a redacted receipt"
            )
        return workload, file_sha256, {"path": None, "sha256": file_sha256}, False
    workload, original_workload_sha256 = _validate_redacted_current_source_workload(
        document
    )
    if original_workload_sha256 != manifest.get("workload_sha256"):
        raise AnalysisContractError(
            "redacted workload original SHA-256 does not match the manifest"
        )
    source_request_count = sum(
        "pcss_current_source_commitment" in row for row in workload
    )
    return (
        workload,
        original_workload_sha256,
        {
            "path": None,
            "sha256": file_sha256,
            "original_workload_sha256": original_workload_sha256,
            "original_workload_sha256_asserted": True,
            "redacted_current_source": True,
            "source_request_count": source_request_count,
        },
        True,
    )


def _validate_redacted_sample(
    sample: Mapping[str, Any],
    workload_row: Mapping[str, Any],
    request_defaults: Mapping[str, Any],
    path: Path,
) -> None:
    expected_body = dict(request_defaults)
    expected_body.update(workload_row["body"])
    expected_body.pop("model", None)
    if sample.get("request_body") != expected_body:
        raise AnalysisContractError(
            f"source-redacted sample request body does not match archival evidence: {path}"
        )
    expected_commitment = workload_row.get("pcss_current_source_commitment")
    if sample.get("pcss_current_source_commitment") != expected_commitment:
        raise AnalysisContractError(
            f"source-redacted sample commitment does not match archival evidence: {path}"
        )
    response = sample.get("response")
    if not isinstance(response, dict):
        raise AnalysisContractError(f"source-redacted sample response is missing: {path}")
    if response.get("events") != [] or response.get("raw_body_utf8") is not None:
        raise AnalysisContractError(
            f"source-redacted sample retains raw response material: {path}"
        )


def _load_samples(
    raw_directory: Path,
    workload: Sequence[Mapping[str, Any]],
    *,
    source_redacted: bool = False,
    request_defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, str]]]]:
    expected = [(row["request_id"], row["chronological_index"]) for row in workload]
    results: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, list[dict[str, str]]] = {}
    for arm in ARM_NAMES:
        sample_directory = raw_directory / "arms" / arm / "samples"
        paths = sorted(sample_directory.glob("*.json")) if sample_directory.is_dir() else []
        if len(paths) != len(expected):
            raise AnalysisContractError(
                f"arm {arm} does not contain the complete frozen sample matrix"
            )
        arm_rows: list[dict[str, Any]] = []
        arm_hashes: list[dict[str, str]] = []
        for position, path in enumerate(paths):
            sample, raw = _read_json(path, f"{arm} sample {position}")
            request_id, chronological_index = expected[position]
            if sample.get("arm") != arm:
                raise AnalysisContractError(f"sample arm mismatch: {path}")
            if sample.get("request_id") != request_id:
                raise AnalysisContractError(f"sample request order mismatch: {path}")
            if sample.get("chronological_index") != chronological_index:
                raise AnalysisContractError(f"sample chronological index mismatch: {path}")
            if source_redacted:
                if request_defaults is None:  # pragma: no cover - caller contract
                    raise AnalysisContractError("source-redacted analysis lacks request defaults")
                _validate_redacted_sample(
                    sample, workload[position], request_defaults, path
                )
            _number(sample.get("accounted_latency_ns"), "sample.accounted_latency_ns", positive=True)
            response = sample.get("response")
            if not isinstance(response, dict):
                raise AnalysisContractError(f"sample response is missing: {path}")
            if response.get("error") is not None or response.get("status") != 200:
                raise AnalysisContractError(f"sample response is not successful: {path}")
            output_tokens = response.get("output_tokens")
            if output_tokens is not None and (
                isinstance(output_tokens, bool)
                or not isinstance(output_tokens, int)
                or output_tokens < 0
            ):
                raise AnalysisContractError(f"sample output_tokens is invalid: {path}")
            ttft = response.get("ttft_ns")
            if ttft is not None:
                _number(ttft, "sample.response.ttft_ns")
            arm_rows.append(sample)
            arm_hashes.append(
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256_bytes(raw),
                }
            )
        results[arm] = arm_rows
        hashes[arm] = arm_hashes
    return results, hashes


def _verify_policy_binding(
    raw_directory: Path,
    result: Mapping[str, Any],
    receipts_by_arm: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify policy identity from the frozen manifest through every launch.

    Legacy artifacts were produced before this binding existed.  They are
    intentionally analyzable as historical evidence, but cannot receive a
    policy-family claim.  A manifest that declares a policy is stricter: both
    the summary and the per-arm startup receipt must carry the exact pair.
    """

    expected = _policy_binding(manifest)
    if expected["policy_id"] is None:
        observed = result.get("policy_binding")
        if observed is not None and observed != expected:
            raise AnalysisContractError(
                "legacy raw result contains a policy binding that does not match its manifest"
            )
        return {
            **expected,
            "status": "legacy_unbound",
            "raw_result_binding_verified": observed == expected,
            "arm_startup_bindings_verified": False,
            "arm_receipts": [],
        }

    if result.get("policy_binding") != expected:
        raise AnalysisContractError(
            "raw result policy binding does not match the frozen manifest"
        )
    policy_spec_path = raw_directory / "policy-spec.json"
    policy_spec_document, policy_spec_raw = _read_json(
        policy_spec_path, "copied policy specification"
    )
    if _sha256_bytes(policy_spec_raw) != expected["policy_spec_sha256"]:
        raise AnalysisContractError(
            "copied policy specification SHA-256 does not match manifest"
        )
    if policy_spec_document.get("policy_id") != expected["policy_id"]:
        raise AnalysisContractError(
            "copied policy specification identity does not match manifest"
        )
    if result.get("policy_spec") != {
        "relative_path": "qualification/policies/"
        + expected["policy_id"].replace(".", "-").replace("/", "-")
        + ".json",
        "sha256": expected["policy_spec_sha256"],
    }:
        raise AnalysisContractError(
            "raw result copied policy specification evidence does not match manifest"
        )

    receipt_evidence: list[dict[str, Any]] = []
    for arm in ARM_NAMES:
        receipt = receipts_by_arm[arm]
        command = receipt.get("command")
        if not isinstance(command, list) or not all(
            isinstance(argument, str) and argument for argument in command
        ):
            raise AnalysisContractError(
                f"raw result {arm} arm receipt has no executable startup command"
            )
        startup = receipt.get("startup_evidence")
        if not isinstance(startup, dict):
            raise AnalysisContractError(
                f"raw result {arm} arm receipt lacks policy startup evidence"
            )
        if startup.get("policy_binding") != expected:
            raise AnalysisContractError(
                f"raw result {arm} startup policy binding does not match the frozen manifest"
            )
        if startup.get("command_sha256") != _canonical_json_sha256(command):
            raise AnalysisContractError(
                f"raw result {arm} startup command digest does not match the receipt command"
            )
        if arm == "pcss-cache":
            attestation = startup.get("server_policy_attestation")
            expected_attestation = _policy_metric_attestation(
                receipt.get("metrics_at_ready"), expected
            )
            if attestation != expected_attestation or not expected_attestation["verified"]:
                raise AnalysisContractError(
                    "PCSS server policy metric does not attest to the frozen policy"
                )
        arm_path = raw_directory / "arms" / arm / "arm.json"
        arm_document, arm_raw = _read_json(arm_path, f"{arm} arm receipt")
        if arm_document != receipt:
            raise AnalysisContractError(
                f"raw result {arm} arm receipt differs from its startup evidence file"
            )
        receipt_evidence.append(
            {"arm": arm, "path": str(arm_path.resolve()), "sha256": _sha256_bytes(arm_raw)}
        )
    return {
        **expected,
        "status": "bound",
        "raw_result_binding_verified": True,
        "arm_startup_bindings_verified": True,
        "pcss_server_metric_verified": True,
        "policy_spec": {
            "path": str(policy_spec_path.resolve()),
            "sha256": _sha256_bytes(policy_spec_raw),
        },
        "arm_receipts": receipt_evidence,
    }


def _verify_capture_binding(
    raw_directory: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    workload: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Recheck the public capture binding without reopening private source data."""

    workload_class = manifest["workload_class"]
    archived_source_ids = {
        row["request_id"]
        for row in workload
        if "pcss_current_source_commitment" in row
    }
    is_agent_assisted = workload_class == "agent_assisted_calibration"
    is_owner_holdout = workload_class == "owner_representative_holdout"
    requires_receipt = is_agent_assisted or is_owner_holdout
    if not requires_receipt:
        if result.get("capture_receipt_sha256") is not None:
            raise AnalysisContractError(
                "uncaptured raw run must not contain capture receipt evidence"
            )
        return None
    receipt_path = raw_directory / "capture-receipt.json"
    receipt, raw = _read_json(receipt_path, "copied capture receipt")
    provenance = manifest["workload_provenance"]
    digest = _sha256_bytes(raw)
    if (
        digest != provenance["capture_receipt_sha256"]
        or result.get("capture_receipt_sha256") != digest
    ):
        raise AnalysisContractError(
            "copied capture receipt SHA-256 does not match frozen evidence"
        )
    if receipt.get("origin") != provenance["origin"]:
        raise AnalysisContractError("copied capture receipt origin does not match manifest")
    if receipt.get("source_sha256") != provenance["source_event_sha256"]:
        raise AnalysisContractError(
            "copied capture receipt source SHA-256 does not match manifest"
        )
    if receipt.get("workload_sha256") != manifest["workload_sha256"]:
        raise AnalysisContractError(
            "copied capture receipt workload SHA-256 does not match manifest"
        )
    if receipt.get("contains_responses") is not False:
        raise AnalysisContractError("copied capture receipt must exclude responses")
    source_capture = bool(archived_source_ids)
    expected_keys = (
        _CAPTURE_RECEIPT_V2_KEYS if source_capture else _CAPTURE_RECEIPT_V1_KEYS
    )
    _exact_mapping(receipt, expected_keys, "copied capture receipt")
    expected_schema = (
        CAPTURE_RECEIPT_SCHEMA_V2 if source_capture else CAPTURE_RECEIPT_SCHEMA_V1
    )
    if receipt["schema_version"] != expected_schema:
        raise AnalysisContractError("copied capture receipt schema is unsupported")
    _positive_integer(receipt["source_event_count"], "copied capture receipt.source_event_count")
    if _positive_integer(
        receipt["selected_event_count"], "copied capture receipt.selected_event_count"
    ) != len(workload):
        raise AnalysisContractError(
            "copied capture receipt selected event count does not bind workload"
        )
    _positive_integer(receipt["per_episode"], "copied capture receipt.per_episode")
    _sha256_digest(receipt["source_sha256"], "copied capture receipt.source_sha256")
    _sha256_digest(receipt["workload_sha256"], "copied capture receipt.workload_sha256")
    _sha256_digest(receipt["final_event_sha256"], "copied capture receipt.final_event_sha256")
    request_episode_map = receipt.get("request_episode_map")
    if not isinstance(request_episode_map, dict):
        raise AnalysisContractError("copied capture receipt lacks request-to-episode map")
    if set(request_episode_map) != {row["request_id"] for row in workload}:
        raise AnalysisContractError(
            "copied capture receipt does not bind the frozen workload matrix"
        )
    if request_episode_map != manifest["clustering"]["request_cluster_ids"]:
        raise AnalysisContractError(
            "manifest clusters do not match the copied capture receipt"
        )
    evidence: dict[str, Any] = {
        "path": str(receipt_path.resolve()),
        "sha256": digest,
        "origin": receipt["origin"],
        "source_event_sha256": receipt["source_sha256"],
    }
    if source_capture:
        expected_origin = (
            "agent_assisted_local_llama_server"
            if is_agent_assisted
            else "ordinary_local_llama_server"
        )
        if receipt["capture_tool_version"] != "pcss-project-capture-proxy/2":
            raise AnalysisContractError("source capture receipt requires capture proxy v2")
        if receipt["origin"] != expected_origin:
            raise AnalysisContractError(
                "source capture receipt origin does not match workload class"
            )
        if receipt["source_sidecar_sha256"] != provenance["source_sidecar_sha256"]:
            raise AnalysisContractError(
                "copied capture receipt source sidecar SHA-256 does not match manifest"
            )
        _sha256_digest(
            receipt["source_sidecar_sha256"],
            "copied capture receipt.source_sidecar_sha256",
        )
        attachment_count = _positive_integer(
            receipt["source_attachment_count"],
            "copied capture receipt.source_attachment_count",
        )
        if attachment_count != provenance["source_attachment_count"]:
            raise AnalysisContractError(
                "copied capture receipt source attachment count does not match manifest"
            )
        sidecar_event_count = _positive_integer(
            receipt["source_sidecar_event_count"],
            "copied capture receipt.source_sidecar_event_count",
        )
        if sidecar_event_count < attachment_count:
            raise AnalysisContractError(
                "copied capture receipt sidecar cannot omit selected source attachments"
            )
        _sha256_digest(
            receipt["final_sidecar_event_sha256"],
            "copied capture receipt.final_sidecar_event_sha256",
        )
        commitments = _exact_mapping(
            receipt["source_attachment_commitments"],
            set(archived_source_ids),
            "copied capture receipt.source_attachment_commitments",
        )
        if attachment_count != len(commitments):
            raise AnalysisContractError(
                "copied capture receipt source attachment count does not bind commitments"
            )
        archived_commitments = {
            row["request_id"]: row["pcss_current_source_commitment"]
            for row in workload
            if "pcss_current_source_commitment" in row
        }
        for request_id, commitment in commitments.items():
            value = _exact_mapping(
                commitment,
                _CAPTURE_SOURCE_COMMITMENT_KEYS,
                f"copied capture receipt.source_attachment_commitments.{request_id}",
            )
            if value["schema_version"] != CAPTURE_SOURCE_COMMITMENT_SCHEMA:
                raise AnalysisContractError("copied capture receipt source commitment schema is unsupported")
            _sha256_digest(
                value["source_envelope_sha256"],
                f"copied capture receipt.source_attachment_commitments.{request_id}.source_envelope_sha256",
            )
            _positive_integer(
                value["token_count"],
                f"copied capture receipt.source_attachment_commitments.{request_id}.token_count",
            )
            _sha256_digest(
                value["tokens_sha256"],
                f"copied capture receipt.source_attachment_commitments.{request_id}.tokens_sha256",
            )
            archived_commitment = archived_commitments[request_id]
            if (
                value["token_count"] != archived_commitment["token_count"]
                or value["tokens_sha256"] != archived_commitment["tokens_sha256"]
            ):
                raise AnalysisContractError(
                    "copied capture receipt public source commitment does not match redacted workload"
                )
        evidence.update(
            {
                "source_sidecar_sha256": receipt["source_sidecar_sha256"],
                "source_attachment_count": attachment_count,
            }
        )
    elif receipt["capture_tool_version"] != "pcss-project-capture-proxy/1":
        raise AnalysisContractError("capture receipt requires capture proxy v1")
    return evidence


def _validate_holdout_marker(
    result: Mapping[str, Any], manifest: Mapping[str, Any], manifest_sha256: str
) -> dict[str, Any]:
    marker_value = result.get("holdout_opening_marker")
    if manifest["workload_class"] != "owner_representative_holdout":
        if marker_value is not None:
            raise AnalysisContractError("non-holdout raw run cannot contain a holdout marker")
        return {"required": False, "valid": False, "path": None, "sha256": None, "reason": "not_owner_holdout"}
    if not isinstance(marker_value, str) or not marker_value:
        return {"required": True, "valid": False, "path": None, "sha256": None, "reason": "marker_missing"}
    marker_path = Path(marker_value)
    try:
        payload, raw = _read_json(marker_path, "holdout opening marker")
    except AnalysisContractError:
        return {"required": True, "valid": False, "path": str(marker_path), "sha256": None, "reason": "marker_unreadable"}
    safe_study_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in manifest["study_id"]
    )
    expected_name = f".{safe_study_id}.{manifest_sha256}.owner-holdout-opened"
    expected_payload = {
        "study_id": manifest["study_id"],
        "manifest_sha256": manifest_sha256,
        "one_shot": True,
    }
    valid = marker_path.name == expected_name and payload == expected_payload
    return {
        "required": True,
        "valid": valid,
        "path": str(marker_path.resolve()),
        "sha256": _sha256_bytes(raw),
        "reason": None if valid else "marker_identity_or_payload_mismatch",
    }


def _miss_overhead(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    cluster_ids: Mapping[str, str],
    *,
    seed: int,
) -> dict[str, Any]:
    baseline = {row["request_id"]: row for row in results["baseline"]}
    observations: list[tuple[str, str, float, float]] = []
    for pcss in results["pcss-cache"]:
        request_id = pcss["request_id"]
        if _is_miss_heavy(pcss):
            baseline_latency = float(baseline[request_id]["accounted_latency_ns"])
            overhead = float(pcss["accounted_latency_ns"]) / baseline_latency - 1.0
            observations.append((request_id, cluster_ids[request_id], overhead, 1.0))
    if not observations:
        return {
            "definition": "PCSS request with positive lookups and zero persistent-hit delta; current-context drafts remain included",
            "sample_count": 0,
            "cluster_count": 0,
            "p95_overhead": None,
            "ucb95": None,
            "confidence_method": "deterministic_clustered_bootstrap_one_sided_95",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }

    def p95(rows: Sequence[tuple[str, str, float, float]]) -> float:
        return _percentile([row[2] for row in rows], 0.95)

    bootstrapped = _clustered_bootstrap(
        observations,
        seed=seed,
        statistic=p95,
    )
    return {
        "definition": "PCSS request with positive lookups and zero persistent-hit delta; current-context drafts remain included",
        "sample_count": len(observations),
        "cluster_count": len({row[1] for row in observations}),
        "p95_overhead": p95(observations),
        "ucb95": _percentile(bootstrapped, 0.95),
        "confidence_method": "deterministic_clustered_bootstrap_one_sided_95",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def _arm_descriptives(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["accounted_latency_ns"]) for row in rows]
    ttft = [float(row["response"]["ttft_ns"]) for row in rows if row["response"].get("ttft_ns") is not None]
    per_request_throughput = [
        float(row["response"]["output_tokens"]) * 1e9 / float(row["accounted_latency_ns"])
        for row in rows
        if row["response"].get("output_tokens") is not None
    ]
    total_tokens = sum(
        int(row["response"]["output_tokens"])
        for row in rows
        if row["response"].get("output_tokens") is not None
    )
    return {
        "accounted_latency_ns": _descriptive(latencies),
        "ttft_ns": _descriptive(ttft),
        "output_tokens_per_second": {
            **_descriptive(per_request_throughput),
            "aggregate": total_tokens * 1e9 / sum(latencies),
            "total_output_tokens": total_tokens,
        },
    }


def analyze(
    *,
    raw_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate and analyze one frozen raw directory, writing a fresh receipt."""

    raw_directory = raw_directory.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise AnalysisContractError(f"refusing to overwrite analysis output: {output_path}")
    result, result_raw = _read_json(raw_directory / "result.json", "raw result")
    manifest, manifest_raw = _read_json(raw_directory / "manifest.json", "frozen manifest")
    workload_document, workload_raw = _read_json(raw_directory / "workload.json", "frozen workload")
    result_sha256 = _sha256_bytes(result_raw)
    manifest_sha256 = _sha256_bytes(manifest_raw)
    workload_file_sha256 = _sha256_bytes(workload_raw)
    try:
        (
            workload,
            original_workload_sha256,
            workload_evidence,
            source_redacted,
        ) = _archived_workload(workload_document, workload_raw, manifest)
        manifest = _validate_manifest(
            manifest,
            workload_sha256=original_workload_sha256,
            workload_request_ids=[row["request_id"] for row in workload],
            requires_current_source_activation=source_redacted,
            current_source_request_count=sum(
                "pcss_current_source_commitment" in row for row in workload
            ),
        )
    except ValueError as error:
        raise AnalysisContractError(f"frozen input contract failed: {error}") from error
    workload_evidence["path"] = str((raw_directory / "workload.json").resolve())
    if result.get("schema_version") != RESULT_SCHEMA:
        raise AnalysisContractError("unsupported raw result schema_version")
    if result.get("study_id") != manifest["study_id"]:
        raise AnalysisContractError("raw result study_id does not match manifest")
    expected_hashes = {
        "manifest_sha256": manifest_sha256,
        "workload_sha256": original_workload_sha256,
        "model_sha256": manifest["model_sha256"],
        "server_sha256": manifest["server_sha256"],
    }
    for field, expected in expected_hashes.items():
        if result.get(field) != expected:
            raise AnalysisContractError(f"raw result {field} does not match frozen evidence")
    if source_redacted:
        expected_redaction = {
            "schema_version": REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
            "original_workload_sha256": original_workload_sha256,
            "redacted_workload_sha256": workload_file_sha256,
            "source_request_count": workload_evidence["source_request_count"],
            "source_metrics_complete": True,
        }
        if result.get("source_workload_redaction") != expected_redaction:
            raise AnalysisContractError(
                "raw result current-source redaction receipt does not match workload evidence"
            )
    elif result.get("source_workload_redaction") is not None:
        raise AnalysisContractError(
            "legacy workload must not claim a current-source redaction receipt"
        )
    if result.get("valid_raw_run") is not True:
        raise AnalysisContractError("raw result is not marked as a valid raw run")
    required_raw_gates = (
        "metrics_complete",
        "request_matrix_complete",
        "cleanup_complete",
        "resource_evidence_complete",
        "resource_ceilings_passed",
    )
    if source_redacted and result.get("source_workload_redaction", {}).get(
        "source_metrics_complete"
    ) is not True:
        raise AnalysisContractError("raw source run lacks complete activation metric evidence")
    failed_raw_gates = [field for field in required_raw_gates if result.get(field) is not True]
    if failed_raw_gates:
        raise AnalysisContractError(f"raw result validity gates are not complete: {failed_raw_gates}")
    budget_status = result.get("budget_status")
    if not isinstance(budget_status, dict) or any(
        budget_status.get(field) is not True
        for field in (
            "requests_within_budget",
            "wall_clock_within_budget",
            "artifact_within_budget",
        )
    ):
        raise AnalysisContractError("raw result budget validity is incomplete")
    expected_request_count = len(workload) * len(ARM_NAMES)
    if result.get("measured_request_count") != expected_request_count:
        raise AnalysisContractError("raw result measured request count is not the full matrix")
    arm_receipts = result.get("arm_receipts")
    if not isinstance(arm_receipts, list) or len(arm_receipts) != len(ARM_NAMES):
        raise AnalysisContractError("raw result arm receipts are incomplete")
    receipts_by_arm = {
        receipt.get("arm"): receipt for receipt in arm_receipts if isinstance(receipt, dict)
    }
    if set(receipts_by_arm) != set(ARM_NAMES) or any(
        receipts_by_arm[arm].get("error") is not None
        or receipts_by_arm[arm].get("request_results") != len(workload)
        for arm in ARM_NAMES
    ):
        raise AnalysisContractError("raw result arm receipt matrix is invalid")
    if source_redacted:
        source_startup = receipts_by_arm["pcss-cache"].get("startup_evidence")
        expected_source_metrics = _current_source_metric_attestation(
            receipts_by_arm["pcss-cache"].get("metrics_at_ready"), required=True
        )
        if (
            not isinstance(source_startup, dict)
            or source_startup.get("current_source_metric_attestation")
            != expected_source_metrics
            or not expected_source_metrics["verified"]
        ):
            raise AnalysisContractError(
                "PCSS startup lacks the required current-source metric attestation"
            )
    policy_evidence = _verify_policy_binding(
        raw_directory,
        result,
        receipts_by_arm,
        manifest,
    )
    capture_evidence = _verify_capture_binding(
        raw_directory,
        result,
        manifest,
        workload,
    )
    if result.get("exact_output_gate", {}).get("passed") is not True:
        raise AnalysisContractError("raw result exact-output gate did not pass")
    if manifest["effect_gates"]["ratio_direction"] != _RATIO_DIRECTION:
        raise AnalysisContractError("unsupported preregistered ratio direction")
    if manifest["effect_gates"]["confidence_method"] != "clustered_bootstrap_lcb95":
        raise AnalysisContractError("unsupported preregistered confidence method")

    results, sample_hashes = _load_samples(
        raw_directory,
        workload,
        source_redacted=source_redacted,
        request_defaults=manifest["request_defaults"],
    )
    recomputed_exact = _exact_output_gate(
        results,
        required_arms=manifest["effect_gates"].get("exact_output_arms", list(ARM_NAMES[1:])),
    )
    if not recomputed_exact["passed"] or result["exact_output_gate"] != recomputed_exact:
        raise AnalysisContractError("raw sample matrix does not reproduce the exact-output gate")
    cluster_ids = manifest["clustering"]["request_cluster_ids"]
    baseline = {row["request_id"]: row for row in results["baseline"]}
    comparisons: dict[str, Any] = {}
    for arm in ARM_NAMES[1:]:
        observations = [
            (
                row["request_id"],
                cluster_ids[row["request_id"]],
                float(baseline[row["request_id"]]["accounted_latency_ns"]),
                float(row["accounted_latency_ns"]),
            )
            for row in results[arm]
        ]
        comparisons[f"baseline_over_{arm}"] = _paired_ratio(
            observations,
            seed=_seed(result_sha256, manifest_sha256, arm),
            direction=_RATIO_DIRECTION,
        )
    ngram = {row["request_id"]: row for row in results["upstream-ngram-cache"]}
    pcss_over_ngram_observations = [
        (
            row["request_id"],
            cluster_ids[row["request_id"]],
            float(ngram[row["request_id"]]["accounted_latency_ns"]),
            float(row["accounted_latency_ns"]),
        )
        for row in results["pcss-cache"]
    ]
    comparisons["ngram_over_pcss"] = _paired_ratio(
        pcss_over_ngram_observations,
        seed=_seed(result_sha256, manifest_sha256, "ngram-over-pcss"),
        direction="upstream_ngram_over_pcss_gt_1_means_pcss_is_faster",
    )
    miss = _miss_overhead(
        results,
        cluster_ids,
        seed=_seed(result_sha256, manifest_sha256, "miss-heavy"),
    )
    ladder_activation = _suffix_ladder_activation(
        results["pcss-cache"],
        cluster_ids,
        manifest["effect_gates"],
    )
    source_activation = (
        _current_source_activation(
            results["pcss-cache"],
            workload,
            cluster_ids,
            manifest["effect_gates"],
        )
        if source_redacted
        else None
    )
    primary_threshold = float(manifest["effect_gates"]["baseline_over_pcss_lcb95_min"])
    miss_threshold = float(manifest["effect_gates"]["miss_heavy_p95_overhead_ucb95_max"])
    primary_lcb = comparisons["baseline_over_pcss-cache"]["ci95"]["lower"]
    primary_passed = primary_lcb >= primary_threshold
    miss_passed = miss["ucb95"] is not None and miss["ucb95"] <= miss_threshold
    marker = _validate_holdout_marker(result, manifest, manifest_sha256)
    representative_protocol = bool(
        manifest["workload_class"] == "owner_representative_holdout"
        and manifest["holdout"] is True
        and manifest["holdout_split"]["one_shot"] is True
        and marker["valid"]
        and policy_evidence["status"] == "bound"
    )
    all_effect_gates_passed = bool(
        primary_passed
        and miss_passed
        and ladder_activation["passed"]
        and (source_activation is None or source_activation["passed"])
    )
    if policy_evidence["status"] == "legacy_unbound":
        enforced_claim_boundary = (
            "legacy unbound evidence only; no policy-family, representative, or general speedup claim"
        )
        claim_class = "legacy_unbound"
    elif manifest["workload_class"] == "synthetic_control":
        enforced_claim_boundary = "synthetic capability only; no representative or general speedup claim"
        claim_class = "capability_only"
    elif manifest["workload_class"] == "project_naturalistic_development":
        enforced_claim_boundary = "project-local development evidence only; no representative or general speedup claim"
        claim_class = "project_local_only"
    elif manifest["workload_class"] == "agent_assisted_calibration":
        enforced_claim_boundary = "agent-assisted calibration only; no representative or general speedup claim"
        claim_class = "agent_assisted_calibration_only"
    else:
        enforced_claim_boundary = (
            "owner representative one-shot holdout"
            if representative_protocol
            else "representative/general claim refused: one-shot owner holdout marker is invalid"
        )
        claim_class = "owner_representative_holdout" if representative_protocol else "claim_refused"

    analysis = {
        "schema_version": ANALYSIS_SCHEMA,
        "study_id": manifest["study_id"],
        "analysis_policy": {
            "frozen": True,
            "post_result_tuning": False,
            "cluster_source": "manifest.clustering.request_cluster_ids",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "raw_evidence": {
            "directory": str(raw_directory),
            "result": {"path": str((raw_directory / "result.json").resolve()), "sha256": result_sha256},
            "manifest": {"path": str((raw_directory / "manifest.json").resolve()), "sha256": manifest_sha256},
            "workload": workload_evidence,
            "samples": sample_hashes,
            "capture_receipt": capture_evidence,
            "policy_spec": policy_evidence.get("policy_spec"),
            "policy_binding_arm_receipts": policy_evidence["arm_receipts"],
            "holdout_opening_marker": marker,
        },
        "policy_binding": policy_evidence,
        "validation": {
            "valid_raw_run": True,
            "hashes_match": True,
            "full_paired_matrix": True,
            "exact_output_gate_recomputed": True,
            "pair_count": len(workload),
            "cluster_count": len(set(cluster_ids.values())),
        },
        "paired_end_to_end_latency_ratios": comparisons,
        "miss_heavy_pcss_over_baseline": miss,
        "suffix_ladder_activation": ladder_activation,
        "descriptive_only": {
            arm: _arm_descriptives(results[arm]) for arm in ARM_NAMES
        },
        "effect_gate_verdicts": {
            "baseline_over_pcss_lcb95": {
                "value": primary_lcb,
                "threshold_min": primary_threshold,
                "passed": primary_passed,
            },
            "miss_heavy_p95_overhead_ucb95": {
                "value": miss["ucb95"],
                "threshold_max": miss_threshold,
                "passed": miss_passed,
            },
            "suffix_ladder_activation": ladder_activation,
            "exact_output": {"passed": True},
            "all_preregistered_effect_gates_passed": all_effect_gates_passed,
        },
        "ngram_comparison": {
            "reporting_only_no_preregistered_superiority_gate": True,
            "ngram_over_pcss": comparisons["ngram_over_pcss"],
        },
        "claims": {
            "workload_class": manifest["workload_class"],
            "manifest_claim_boundary": manifest["claim_boundary"],
            "enforced_claim_boundary": enforced_claim_boundary,
            "claim_class": claim_class,
            "representative_effect_evaluation_permitted": representative_protocol,
            "policy_family_binding_verified": policy_evidence["status"] == "bound",
            "general_speedup_claim_supported": bool(
                representative_protocol and all_effect_gates_passed
            ),
        },
    }
    if source_redacted:
        analysis["validation"]["current_source_redaction_verified"] = True
        analysis["validation"]["current_source_activation_verified"] = True
        analysis["current_source_activation"] = source_activation
        analysis["effect_gate_verdicts"]["current_source_activation"] = source_activation
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(analysis, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise AnalysisContractError(f"refusing to overwrite analysis output: {output_path}") from error
    except OSError as error:
        raise AnalysisContractError(f"cannot write analysis output: {output_path}") from error
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = analyze(raw_directory=arguments.raw_directory, output_path=arguments.output)
    except AnalysisContractError as error:
        print(f"analysis contract error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
