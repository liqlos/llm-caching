import hashlib
import json
from pathlib import Path

import pytest

from pcss import live_server_analysis as analysis
from pcss import live_server_benchmark as benchmark


def _write(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _canonical_policy_spec() -> tuple[bytes, str]:
    path = (
        Path(__file__).parents[1]
        / "qualification"
        / "policies"
        / "pcss-exact-suffix-ladder-v1.json"
    )
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _metrics(
    *,
    lookups: int,
    hits: int,
    misses: int,
    current_hits: int = 0,
    accepted: int = 0,
    persistent_32_hits: int = 0,
    persistent_32_accepted_tokens: int = 0,
    persistent_64_hits: int = 0,
    persistent_32_suppressions: int = 0,
    fallback_disable_events: int = 0,
    current_source_lookups: int = 0,
    current_source_hits: int = 0,
    current_source_ambiguities: int = 0,
    current_source_abstentions: int = 0,
    current_source_draft_tokens: int = 0,
    current_source_accepted_tokens: int = 0,
    current_source_rejections: int = 0,
    current_source_peak_index_bytes: int = 0,
    current_source_mismatch_abstentions: int = 0,
    policy_id: str | None = None,
    policy_spec_sha256: str | None = None,
) -> dict[str, object]:
    lines = [
        f"llamacpp:pcss_lookups_total {lookups}",
        f"llamacpp:pcss_persistent_hits_total {hits}",
        f"llamacpp:pcss_current_context_hits_total {current_hits}",
        f"llamacpp:pcss_accepted_tokens_total {accepted}",
        f"llamacpp:pcss_persistent_misses_total {misses}",
        f"llamacpp:pcss_abstentions_total {misses}",
        f"llamacpp:pcss_persistent_32_hits_total {persistent_32_hits}",
        "llamacpp:pcss_persistent_32_accepted_tokens_total "
        f"{persistent_32_accepted_tokens}",
        f"llamacpp:pcss_persistent_64_hits_total {persistent_64_hits}",
        "llamacpp:pcss_persistent_32_suppressions_total "
        f"{persistent_32_suppressions}",
        f"llamacpp:pcss_fallback_disable_events_total {fallback_disable_events}",
        f"llamacpp:pcss_current_source_lookups_total {current_source_lookups}",
        f"llamacpp:pcss_current_source_hits_total {current_source_hits}",
        "llamacpp:pcss_current_source_ambiguities_total "
        f"{current_source_ambiguities}",
        "llamacpp:pcss_current_source_abstentions_total "
        f"{current_source_abstentions}",
        "llamacpp:pcss_current_source_draft_tokens_total "
        f"{current_source_draft_tokens}",
        "llamacpp:pcss_current_source_accepted_tokens_total "
        f"{current_source_accepted_tokens}",
        "llamacpp:pcss_current_source_rejections_total "
        f"{current_source_rejections}",
        "llamacpp:pcss_current_source_peak_index_bytes "
        f"{current_source_peak_index_bytes}",
        "llamacpp:pcss_current_source_mismatch_abstentions_total "
        f"{current_source_mismatch_abstentions}",
    ]
    if policy_id is not None and policy_spec_sha256 is not None:
        lines.append(
            "llamacpp:pcss_policy_info{"
            f'policy_id="{policy_id}",'
            f'policy_spec_sha256="{policy_spec_sha256}"'
            "} 1"
        )
    body = "\n".join(lines)
    return {"status": 200, "latency_ns": 1, "body": body + "\n", "error": None}


def _raw_fixture(
    tmp_path: Path,
    *,
    workload_class: str = "synthetic_control",
    baseline_latencies: tuple[int, ...] = (200, 200, 200, 200),
    pcss_latencies: tuple[int, ...] = (100, 100, 100, 100),
) -> tuple[Path, dict[str, object]]:
    raw = tmp_path / "raw"
    request_ids = [f"request-{index}" for index in range(4)]
    workload = {
        "schema_version": benchmark.WORKLOAD_SCHEMA,
        "requests": [
            {
                "request_id": request_id,
                "chronological_index": (index + 1) * 10,
                "endpoint": "/v1/completions",
                "body": {"prompt": f"visible prompt {index}"},
            }
            for index, request_id in enumerate(request_ids)
        ],
    }
    workload_raw = _write(raw / "workload.json", workload)
    holdout = workload_class == "owner_representative_holdout"
    policy_spec_raw, policy_spec_sha256 = _canonical_policy_spec()
    manifest = {
        "schema_version": benchmark.MANIFEST_SCHEMA,
        "study_id": "analysis-unit-study",
        "frozen": True,
        "git_revision": "1" * 40,
        "patch_sha256": "2" * 64,
        "llama_cpp_revision": "3" * 40,
        "runtime_fingerprint": "4" * 64,
        "policy_id": "pcss.exact-suffix-ladder/v1",
        "policy_spec_sha256": policy_spec_sha256,
        "hardware_os": {"hardware": "unit machine", "os": "unit os"},
        "workload_sha256": hashlib.sha256(workload_raw).hexdigest(),
        "workload_class": workload_class,
        "holdout": holdout,
        "workload_provenance": {
            "source": "unit fixture",
            "capture_protocol": "fixture/v1",
            "captured_before_arm_results": True,
            "immutable": True,
        },
        "eligibility_rules": ["all fixture requests"],
        "calibration_split": {
            "request_ids": [] if holdout else request_ids,
            "tuning_permitted": not holdout,
        },
        "holdout_split": {
            "request_ids": request_ids if holdout else [],
            "one_shot": holdout,
        },
        "clustering": {
            "method": "opaque session cluster",
            "request_cluster_ids": {
                request_id: "cluster-a" if index < 2 else "cluster-b"
                for index, request_id in enumerate(request_ids)
            },
        },
        "model_sha256": "5" * 64,
        "server_sha256": "6" * 64,
        "server_args": ["--ctx-size", "128", "--seed", "17"],
        "request_defaults": {"max_tokens": 4, "temperature": 0, "stream": False},
        "arms": [
            {"name": "baseline"},
            {"name": "upstream-ngram-cache"},
            {
                "name": "pcss-cache",
                "pcss": {
                    "fingerprint": "sha256:test",
                    "namespace": "unit",
                    "max_disk_gib": 1,
                    "ram_mib": 1,
                    "read_only": False,
                    "disable_learning": False,
                },
            },
        ],
        "arm_order": list(benchmark.ARM_NAMES),
        "counterbalance": {"method": "latin_square", "sequence_id": "row-1"},
        "state_definitions": {"warm": "immutable snapshot", "cold": "fresh process"},
        "arm_store_initialization": {
            arm: {
                "state": "cold",
                "initialization": f"fresh isolated {arm}",
                "snapshot_sha256": None,
            }
            for arm in benchmark.ARM_NAMES
        },
        "readiness": {
            "path": "/health",
            "timeout_seconds": 2,
            "max_attempts": 4,
            "interval_seconds": 0.01,
        },
        "request_timeout_seconds": 3,
        "request_count": 4,
        "effect_gates": {
            "primary_metric": "paired_end_to_end_latency",
            "ratio_direction": "baseline_over_arm_gt_1_is_faster",
            "baseline_over_pcss_lcb95_min": 1.05,
            "miss_heavy_p95_overhead_ucb95_max": 0.03,
            "confidence_method": "clustered_bootstrap_lcb95",
            "exact_output_required": True,
        },
        "resource_ceilings": {
            "ram_mib": 1024,
            "disk_gib": 2,
            "startup_seconds": 30,
            "compaction_write_amplification_max": 2,
        },
        "allowed_policy_families": ["conservative_exact"],
        "finite_parameter_grids": {"draft_tokens": [2, 4]},
        "budgets": {
            "max_requests": 12,
            "max_wall_clock_seconds": 60,
            "max_artifact_bytes": 10_000_000,
        },
        "stopping_rule": "one frozen pass",
        "kill_gates": ["server crash"],
        "exclusion_rules": ["exclude nothing"],
        "claim_boundary": "maliciously broad wording is not authoritative",
    }
    capture_receipt_sha256: str | None = None
    if holdout:
        capture_receipt = {
            "schema_version": benchmark.CAPTURE_RECEIPT_SCHEMA_V1,
            "source_event_count": len(request_ids),
            "selected_event_count": len(request_ids),
            "episodes": ["cluster-a", "cluster-b"],
            "per_episode": 2,
            "source_sha256": "8" * 64,
            "workload_sha256": manifest["workload_sha256"],
            "capture_tool_version": "pcss-project-capture-proxy/1",
            "origin": "ordinary_local_llama_server",
            "request_episode_map": manifest["clustering"]["request_cluster_ids"],
            "final_event_sha256": "9" * 64,
            "selection_rule": "first visible pre-forward ordinary local llama-server requests",
            "contains_responses": False,
        }
        capture_receipt_raw = _write(raw / "capture-receipt.json", capture_receipt)
        capture_receipt_sha256 = hashlib.sha256(capture_receipt_raw).hexdigest()
        manifest["workload_provenance"] = {
            **manifest["workload_provenance"],
            "origin": "ordinary_local_llama_server",
            "capture_receipt_sha256": capture_receipt_sha256,
            "source_event_sha256": "8" * 64,
        }
    manifest_raw = _write(raw / "manifest.json", manifest)
    (raw / "policy-spec.json").write_bytes(policy_spec_raw)
    latencies = {
        "baseline": baseline_latencies,
        "upstream-ngram-cache": (80, 80, 80, 80),
        "pcss-cache": pcss_latencies,
    }
    samples: dict[str, list[dict[str, object]]] = {}
    for arm in benchmark.ARM_NAMES:
        rows = []
        for index, request_id in enumerate(request_ids):
            is_pcss = arm == "pcss-cache"
            row = {
                "arm": arm,
                "request_id": request_id,
                "chronological_index": (index + 1) * 10,
                "endpoint": "/v1/completions",
                "request_body": {"prompt": f"visible prompt {index}"},
                "process_resource_before": {},
                "metrics_before": _metrics(lookups=0, hits=0, misses=0),
                "response": {
                    "status": 200,
                    "latency_ns": latencies[arm][index] - 1,
                    "ttft_ns": 25,
                    "streaming": False,
                    "output_text": "same exact output",
                    "output_tokens": 4,
                    "finish_reason": "stop",
                    "server_timings": {"predicted_ms": 1},
                    "events": [],
                    "raw_body_utf8": "{}",
                    "error": None,
                },
                "metrics_after": _metrics(
                    lookups=1 if is_pcss else 0,
                    hits=0,
                    misses=1 if is_pcss else 0,
                ),
                "accounted_latency_ns": latencies[arm][index],
                "process_resource_after": {},
            }
            rows.append(row)
            _write(raw / "arms" / arm / "samples" / f"{index:06d}.json", row)
        samples[arm] = rows
    exact = benchmark._exact_output_gate(samples)
    policy_binding = benchmark._policy_binding(manifest)
    arm_receipts = []
    for arm in benchmark.ARM_NAMES:
        command = ["/fixture/llama-server", "--fixture-arm", arm]
        metrics_at_ready = _metrics(
            lookups=0,
            hits=0,
            misses=0,
            policy_id=policy_binding["policy_id"] if arm == "pcss-cache" else None,
            policy_spec_sha256=(
                policy_binding["policy_spec_sha256"] if arm == "pcss-cache" else None
            ),
        )
        attestation = (
            benchmark._policy_metric_attestation(metrics_at_ready, policy_binding)
            if arm == "pcss-cache"
            else {
                "required": False,
                "verified": False,
                "observed": None,
                "matching_lines": 0,
            }
        )
        receipt = {
            "arm": arm,
            "command": command,
            "startup_evidence": {
                "policy_binding": policy_binding,
                "command_sha256": benchmark._canonical_json_sha256(command),
                "server_policy_attestation": attestation,
            },
            "metrics_at_ready": metrics_at_ready,
            "request_results": 4,
            "error": None,
        }
        arm_receipts.append(receipt)
        _write(raw / "arms" / arm / "arm.json", receipt)
    result = {
        "schema_version": benchmark.RESULT_SCHEMA,
        "study_id": manifest["study_id"],
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "workload_sha256": hashlib.sha256(workload_raw).hexdigest(),
        "model_sha256": manifest["model_sha256"],
        "server_sha256": manifest["server_sha256"],
        "policy_binding": policy_binding,
        "policy_spec": {
            "relative_path": "qualification/policies/pcss-exact-suffix-ladder-v1.json",
            "sha256": policy_spec_sha256,
        },
        "capture_receipt_sha256": capture_receipt_sha256,
        "holdout_opening_marker": None,
        "arm_receipts": arm_receipts,
        "exact_output_gate": exact,
        "metrics_complete": True,
        "request_matrix_complete": True,
        "cleanup_complete": True,
        "resource_evidence_complete": True,
        "resource_ceilings_passed": True,
        "measured_request_count": 12,
        "budget_status": {
            "requests_within_budget": True,
            "wall_clock_within_budget": True,
            "artifact_within_budget": True,
        },
        "valid_raw_run": True,
    }
    _write(raw / "result.json", result)
    return raw, result


def _source_redacted_raw_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    raw, result = _raw_fixture(tmp_path)
    manifest = json.loads((raw / "manifest.json").read_text())
    private_workload = json.loads((raw / "workload.json").read_text())
    matching_source = {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "sha256:test",
        "source_id": "sha256:" + "a" * 64,
        "visible_token_ids": [17, 23, 42],
    }
    mismatch_source = {
        **matching_source,
        "compatibility_fingerprint": "mismatch-abstention-control",
        "source_id": "sha256:" + "b" * 64,
    }
    private_workload["requests"][0]["body"]["pcss_current_source"] = matching_source
    private_workload["requests"][1]["body"]["pcss_current_source"] = mismatch_source
    private_workload_raw = (
        json.dumps(private_workload, sort_keys=True) + "\n"
    ).encode()
    original_workload_sha256 = hashlib.sha256(private_workload_raw).hexdigest()
    matching_commitment = {
        "schema_version": benchmark.CURRENT_SOURCE_COMMITMENT_SCHEMA,
        "token_count": 3,
        "tokens_sha256": benchmark._canonical_json_sha256(
            matching_source["visible_token_ids"]
        ),
        "attachment_mode": benchmark.CURRENT_SOURCE_MATCHING_MODE,
    }
    mismatch_commitment = {
        **matching_commitment,
        "attachment_mode": benchmark.CURRENT_SOURCE_MISMATCH_CONTROL_MODE,
    }
    archived_workload = {
        "schema_version": benchmark.REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
        "original_workload_sha256": original_workload_sha256,
        "requests": [
            {
                **request,
                **(
                    {"pcss_current_source_commitment": matching_commitment}
                    if index == 0
                    else {"pcss_current_source_commitment": mismatch_commitment}
                    if index == 1
                    else {}
                ),
            }
            for index, request in enumerate(private_workload["requests"])
        ],
    }
    for request in archived_workload["requests"]:
        request["body"].pop("pcss_current_source", None)
    archived_workload_raw = _write(raw / "workload.json", archived_workload)
    manifest["workload_sha256"] = original_workload_sha256
    manifest["arms"][2]["pcss"]["enable_current_source"] = True
    manifest["effect_gates"].update(
        {
            "current_source_matching_requests_min": 1,
            "current_source_lookup_requests_min": 1,
            "current_source_hit_requests_min": 1,
            "current_source_hit_episodes_min": 1,
            "current_source_accepted_tokens_min": 2,
            "current_source_peak_index_bytes_min": 1,
            "current_source_rejections_max": 0,
            "current_source_ambiguities_max": 0,
            "current_source_abstentions_max": 0,
            "current_source_mismatch_control_requests_min": 1,
            "current_source_mismatch_abstentions_min": 1,
        }
    )
    manifest_raw = _write(raw / "manifest.json", manifest)
    source_stats = [
        {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "current_source_lookups": 0,
            "current_source_hits": 0,
            "current_source_draft_tokens": 0,
            "current_source_accepted_tokens": 0,
            "current_source_rejections": 0,
            "current_source_peak_index_bytes": 0,
            "current_source_mismatch_abstentions": 0,
        },
        {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "current_source_lookups": 1,
            "current_source_hits": 1,
            "current_source_draft_tokens": 2,
            "current_source_accepted_tokens": 2,
            "current_source_rejections": 0,
            "current_source_peak_index_bytes": 96,
            "current_source_mismatch_abstentions": 0,
        },
        {
            "lookups": 1,
            "hits": 0,
            "misses": 1,
            "current_source_lookups": 1,
            "current_source_hits": 1,
            "current_source_draft_tokens": 2,
            "current_source_accepted_tokens": 2,
            "current_source_rejections": 0,
            "current_source_peak_index_bytes": 96,
            "current_source_mismatch_abstentions": 1,
        },
        {
            "lookups": 2,
            "hits": 0,
            "misses": 2,
            "current_source_lookups": 1,
            "current_source_hits": 1,
            "current_source_draft_tokens": 2,
            "current_source_accepted_tokens": 2,
            "current_source_rejections": 0,
            "current_source_peak_index_bytes": 96,
            "current_source_mismatch_abstentions": 1,
        },
        {
            "lookups": 3,
            "hits": 0,
            "misses": 3,
            "current_source_lookups": 1,
            "current_source_hits": 1,
            "current_source_draft_tokens": 2,
            "current_source_accepted_tokens": 2,
            "current_source_rejections": 0,
            "current_source_peak_index_bytes": 96,
            "current_source_mismatch_abstentions": 1,
        },
    ]
    samples: dict[str, list[dict[str, object]]] = {}
    for arm in benchmark.ARM_NAMES:
        rows = []
        for index, request in enumerate(archived_workload["requests"]):
            path = raw / "arms" / arm / "samples" / f"{index:06d}.json"
            row = json.loads(path.read_text())
            row["request_body"] = {
                **manifest["request_defaults"],
                **request["body"],
            }
            row["response"]["events"] = []
            row["response"]["raw_body_utf8"] = None
            if "pcss_current_source_commitment" in request:
                row["pcss_current_source_commitment"] = request[
                    "pcss_current_source_commitment"
                ]
            if arm == "pcss-cache":
                row["metrics_before"] = _metrics(**source_stats[index])
                row["metrics_after"] = _metrics(**source_stats[index + 1])
            rows.append(row)
            _write(path, row)
        samples[arm] = rows
    policy_binding = benchmark._policy_binding(manifest)
    for receipt in result["arm_receipts"]:
        if receipt["arm"] != "pcss-cache":
            continue
        metrics_at_ready = _metrics(
            lookups=0,
            hits=0,
            misses=0,
            policy_id=policy_binding["policy_id"],
            policy_spec_sha256=policy_binding["policy_spec_sha256"],
        )
        receipt["metrics_at_ready"] = metrics_at_ready
        receipt["startup_evidence"]["current_source_metric_attestation"] = (
            benchmark._current_source_metric_attestation(metrics_at_ready, required=True)
        )
        _write(raw / "arms" / "pcss-cache" / "arm.json", receipt)
    result["arm_receipts"] = result["arm_receipts"]
    result["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    result["workload_sha256"] = original_workload_sha256
    result["exact_output_gate"] = benchmark._exact_output_gate(samples)
    result["source_workload_redaction"] = {
        "schema_version": benchmark.REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
        "original_workload_sha256": original_workload_sha256,
        "redacted_workload_sha256": hashlib.sha256(archived_workload_raw).hexdigest(),
        "source_request_count": 2,
        "source_metrics_complete": True,
    }
    _write(raw / "result.json", result)
    return raw, result


def test_analysis_preserves_preregistered_ratio_direction(tmp_path: Path) -> None:
    raw, _ = _raw_fixture(tmp_path)
    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")

    ratio = result["paired_end_to_end_latency_ratios"]["baseline_over_pcss-cache"]
    assert ratio["direction"] == "baseline_over_arm_gt_1_is_faster"
    assert ratio["estimate"] == 2.0
    assert ratio["ci95"] == {"lower": 2.0, "upper": 2.0}
    assert result["effect_gate_verdicts"]["baseline_over_pcss_lcb95"]["passed"] is True
    assert result["paired_end_to_end_latency_ratios"]["ngram_over_pcss"]["estimate"] == 0.8


def test_cluster_bootstrap_resamples_whole_declared_clusters() -> None:
    rows = [
        ("a-1", "a", 0.0, 1.0),
        ("a-2", "a", 0.0, 1.0),
        ("b-1", "b", 1.0, 1.0),
        ("b-2", "b", 1.0, 1.0),
    ]
    values = analysis._clustered_bootstrap(
        rows,
        seed=7,
        statistic=lambda selected: sum(row[2] for row in selected),
        replicates=100,
    )

    assert set(values) == {0.0, 2.0, 4.0}


def test_miss_heavy_ucb95_uses_pcss_over_baseline_overhead(tmp_path: Path) -> None:
    raw, _ = _raw_fixture(
        tmp_path,
        baseline_latencies=(100, 100, 100, 100),
        pcss_latencies=(101, 101, 101, 101),
    )
    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")

    miss = result["miss_heavy_pcss_over_baseline"]
    assert miss["sample_count"] == 4
    assert miss["p95_overhead"] == pytest.approx(0.01)
    assert miss["ucb95"] == pytest.approx(0.01)
    assert result["effect_gate_verdicts"]["miss_heavy_p95_overhead_ucb95"]["passed"] is True


def test_no_persistent_hit_population_includes_current_context_candidates() -> None:
    sample = {
        "metrics_before": _metrics(lookups=0, hits=0, misses=0),
        "metrics_after": _metrics(lookups=3, hits=0, misses=3, current_hits=3, accepted=3),
    }
    assert analysis._is_miss_heavy(sample)


def test_source_miss_heavy_includes_rejected_source_drafts() -> None:
    sample = {
        "metrics_before": _metrics(lookups=0, hits=0, misses=0),
        "metrics_after": _metrics(
            lookups=0,
            hits=0,
            misses=0,
            current_source_lookups=1,
            current_source_hits=1,
            current_source_rejections=1,
        ),
    }

    assert analysis._is_miss_heavy(sample)


def test_analysis_validates_source_redaction_activation_and_mismatch_control(
    tmp_path: Path,
) -> None:
    raw, _ = _source_redacted_raw_fixture(tmp_path)
    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")

    activation = result["current_source_activation"]
    assert result["validation"]["current_source_redaction_verified"] is True
    assert result["validation"]["current_source_activation_verified"] is True
    assert activation["passed"] is True
    assert activation["matching_request_count"] == 1
    assert activation["mismatch_control_request_count"] == 1
    assert activation["matching_totals"]["accepted_tokens"] == 2
    assert activation["mismatch_abstentions"] == 1
    workload_evidence = result["raw_evidence"]["workload"]
    assert workload_evidence["redacted_current_source"] is True
    assert workload_evidence["original_workload_sha256_asserted"] is True
    raw_text = "\n".join(
        path.read_text() for path in raw.rglob("*") if path.is_file()
    )
    assert "sha256:" + "a" * 64 not in raw_text
    assert "visible_token_ids" not in raw_text


def test_analysis_rechecks_owner_source_holdout_capture_without_private_sidecar(tmp_path: Path) -> None:
    raw, result = _source_redacted_raw_fixture(tmp_path)
    manifest = json.loads((raw / "manifest.json").read_text())
    archived_workload = json.loads((raw / "workload.json").read_text())
    request_episode_map = manifest["clustering"]["request_cluster_ids"]
    matching_source = {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "sha256:test",
        "source_id": "sha256:" + "a" * 64,
        "visible_token_ids": [17, 23, 42],
    }
    mismatch_source = {
        **matching_source,
        "compatibility_fingerprint": "mismatch-abstention-control",
        "source_id": "sha256:" + "b" * 64,
    }
    receipt = {
        "schema_version": benchmark.CAPTURE_RECEIPT_SCHEMA_V2,
        "source_event_count": 4,
        "selected_event_count": 4,
        "episodes": ["cluster-a", "cluster-b"],
        "per_episode": 2,
        "source_sha256": "8" * 64,
        "source_sidecar_sha256": "a" * 64,
        "source_sidecar_event_count": 2,
        "source_attachment_count": 2,
        "source_attachment_commitments": {
            "request-0": benchmark._capture_source_commitment(matching_source),
            "request-1": benchmark._capture_source_commitment(mismatch_source),
        },
        "workload_sha256": manifest["workload_sha256"],
        "capture_tool_version": "pcss-project-capture-proxy/2",
        "origin": "ordinary_local_llama_server",
        "request_episode_map": request_episode_map,
        "final_event_sha256": "9" * 64,
        "final_sidecar_event_sha256": "b" * 64,
        "selection_rule": "first visible pre-forward ordinary local llama-server requests",
        "contains_responses": False,
    }
    receipt_raw = _write(raw / "capture-receipt.json", receipt)
    manifest["workload_class"] = "owner_representative_holdout"
    manifest["holdout"] = True
    manifest["calibration_split"] = {"request_ids": [], "tuning_permitted": False}
    manifest["holdout_split"] = {
        "request_ids": [row["request_id"] for row in archived_workload["requests"]],
        "one_shot": True,
    }
    manifest["workload_provenance"] = {
        **manifest["workload_provenance"],
        "origin": "ordinary_local_llama_server",
        "capture_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "source_event_sha256": "8" * 64,
        "source_sidecar_sha256": "a" * 64,
        "source_attachment_count": 2,
    }
    manifest_raw = _write(raw / "manifest.json", manifest)
    result["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    result["capture_receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    _write(raw / "result.json", result)

    analysis_result = analysis.analyze(
        raw_directory=raw,
        output_path=tmp_path / "analysis-owner-source.json",
    )

    capture = analysis_result["raw_evidence"]["capture_receipt"]
    assert capture["origin"] == "ordinary_local_llama_server"
    assert capture["source_sidecar_sha256"] == "a" * 64
    assert capture["source_attachment_count"] == 2
    receipt_text = (raw / "capture-receipt.json").read_text()
    assert "visible_token_ids" not in receipt_text
    assert matching_source["source_id"] not in receipt_text

    receipt["source_attachment_commitments"]["request-0"]["tokens_sha256"] = "0" * 64
    receipt_raw = _write(raw / "capture-receipt.json", receipt)
    manifest["workload_provenance"]["capture_receipt_sha256"] = hashlib.sha256(
        receipt_raw
    ).hexdigest()
    manifest_raw = _write(raw / "manifest.json", manifest)
    result["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    result["capture_receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    _write(raw / "result.json", result)

    with pytest.raises(analysis.AnalysisContractError, match="public source commitment"):
        analysis.analyze(
            raw_directory=raw,
            output_path=tmp_path / "analysis-owner-source-tampered.json",
        )


def test_analysis_refuses_source_redaction_without_mismatch_metric_evidence(
    tmp_path: Path,
) -> None:
    raw, _ = _source_redacted_raw_fixture(tmp_path)
    sample_path = raw / "arms" / "pcss-cache" / "samples" / "000001.json"
    sample = json.loads(sample_path.read_text())
    sample["metrics_after"]["body"] = sample["metrics_after"]["body"].replace(
        "llamacpp:pcss_current_source_mismatch_abstentions_total 1",
        "llamacpp:pcss_current_source_mismatch_abstentions_total 0",
    )
    _write(sample_path, sample)

    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")
    assert result["current_source_activation"]["passed"] is False
    assert result["current_source_activation"]["gate_verdicts"][
        "mismatch_abstentions"
    ] is False


def test_suffix_ladder_activation_requires_short_tier_without_longer_match() -> None:
    sample = {
        "request_id": "request-1",
        "metrics_before": _metrics(lookups=0, hits=0, misses=0),
        "metrics_after": _metrics(
            lookups=1,
            hits=1,
            misses=0,
            persistent_32_hits=1,
            persistent_32_accepted_tokens=2,
        ),
    }
    gates = {
        "persistent_32_hit_requests_min": 1,
        "persistent_32_hit_episodes_min": 1,
        "persistent_32_accepted_tokens_min": 2,
        "persistent_64_hit_requests_max": 0,
        "persistent_32_suppressions_max": 0,
        "fallback_disable_events_max": 0,
    }

    result = analysis._suffix_ladder_activation(
        [sample], {"request-1": "episode-a"}, gates
    )

    assert result["required"] is True
    assert result["short_hit_request_count"] == 1
    assert result["short_hit_episode_count"] == 1
    assert result["passed"] is True


def test_analysis_refuses_raw_exact_gate_not_reproduced(tmp_path: Path) -> None:
    raw, _ = _raw_fixture(tmp_path)
    sample_path = raw / "arms" / "pcss-cache" / "samples" / "000000.json"
    sample = json.loads(sample_path.read_text())
    sample["response"]["output_text"] = "different"
    _write(sample_path, sample)

    with pytest.raises(analysis.AnalysisContractError, match="exact-output"):
        analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")


def test_synthetic_claim_boundary_is_capability_only_and_output_is_immutable(
    tmp_path: Path,
) -> None:
    raw, _ = _raw_fixture(tmp_path)
    output = tmp_path / "analysis.json"
    result = analysis.analyze(raw_directory=raw, output_path=output)

    assert result["claims"]["claim_class"] == "capability_only"
    assert result["claims"]["general_speedup_claim_supported"] is False
    assert "no representative or general speedup claim" in result["claims"]["enforced_claim_boundary"]
    with pytest.raises(analysis.AnalysisContractError, match="overwrite"):
        analysis.analyze(raw_directory=raw, output_path=output)


def test_analysis_verifies_policy_binding_in_every_arm_startup_receipt(
    tmp_path: Path,
) -> None:
    raw, raw_result = _raw_fixture(tmp_path)
    receipt = raw_result["arm_receipts"][0]
    receipt["startup_evidence"] = {
        **receipt["startup_evidence"],
        "policy_binding": {
            **receipt["startup_evidence"]["policy_binding"],
            "policy_id": "pcss.other/v1",
        },
    }
    _write(raw / "arms" / "baseline" / "arm.json", receipt)
    _write(raw / "result.json", raw_result)

    with pytest.raises(analysis.AnalysisContractError, match="startup policy binding"):
        analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")


def test_legacy_raw_evidence_is_readable_but_cannot_support_policy_claims(
    tmp_path: Path,
) -> None:
    raw, raw_result = _raw_fixture(
        tmp_path,
        workload_class="owner_representative_holdout",
    )
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["policy_id"]
    del manifest["policy_spec_sha256"]
    manifest_raw = _write(manifest_path, manifest)
    raw_result["manifest_sha256"] = hashlib.sha256(manifest_raw).hexdigest()
    raw_result.pop("policy_binding")
    for receipt in raw_result["arm_receipts"]:
        receipt.pop("command")
        receipt.pop("startup_evidence")
    _write(raw / "result.json", raw_result)

    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")

    assert result["policy_binding"]["status"] == "legacy_unbound"
    assert result["claims"]["claim_class"] == "legacy_unbound"
    assert result["claims"]["policy_family_binding_verified"] is False
    assert result["claims"]["general_speedup_claim_supported"] is False


def test_owner_holdout_without_valid_one_shot_marker_refuses_general_claim(
    tmp_path: Path,
) -> None:
    raw, _ = _raw_fixture(tmp_path, workload_class="owner_representative_holdout")
    result = analysis.analyze(raw_directory=raw, output_path=tmp_path / "analysis.json")

    assert result["claims"]["claim_class"] == "claim_refused"
    assert result["claims"]["representative_effect_evaluation_permitted"] is False
    assert result["claims"]["general_speedup_claim_supported"] is False


def test_owner_holdout_with_valid_one_shot_marker_can_support_passing_claim(
    tmp_path: Path,
) -> None:
    raw, raw_result = _raw_fixture(
        tmp_path, workload_class="owner_representative_holdout"
    )
    manifest_sha256 = raw_result["manifest_sha256"]
    marker = tmp_path / (
        f".analysis-unit-study.{manifest_sha256}.owner-holdout-opened"
    )
    _write(
        marker,
        {
            "study_id": "analysis-unit-study",
            "manifest_sha256": manifest_sha256,
            "one_shot": True,
        },
    )
    raw_result["holdout_opening_marker"] = str(marker)
    _write(raw / "result.json", raw_result)

    result = analysis.analyze(
        raw_directory=raw, output_path=tmp_path / "analysis.json"
    )

    assert result["claims"]["claim_class"] == "owner_representative_holdout"
    assert result["claims"]["representative_effect_evaluation_permitted"] is True
    assert result["claims"]["general_speedup_claim_supported"] is True
