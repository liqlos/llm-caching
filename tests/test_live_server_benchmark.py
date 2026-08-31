import hashlib
import json
import subprocess
from pathlib import Path
from urllib.error import URLError

import pytest

from pcss import live_server_benchmark as benchmark


def write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def canonical_policy_spec() -> tuple[bytes, str]:
    path = (
        Path(__file__).parents[1]
        / "qualification"
        / "policies"
        / "pcss-exact-suffix-ladder-v1.json"
    )
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def current_source_payload() -> dict[str, object]:
    return {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "sha256:test",
        "source_id": "sha256:" + "a" * 64,
        "visible_token_ids": [17, 23, 42],
    }


def test_exclusive_compute_ignores_runner_wrapper_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        (
            "100 1 /bin/zsh -c .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
            "200 100 lean-ctx -c .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
            "300 200 .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
        )
    )
    monkeypatch.setattr(benchmark.os, "getpid", lambda: 300)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )

    benchmark._assert_exclusive_local_compute()


def test_exclusive_compute_keeps_independent_server_as_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = "\n".join(
        (
            "100 1 /bin/zsh -c .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
            "200 100 lean-ctx -c .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
            "300 200 .venv/bin/python -m pcss.live_server_benchmark --server build/llama.cpp/bin/llama-server",
            "400 1 /Users/example/build/llama-server --model model.gguf",
        )
    )
    monkeypatch.setattr(benchmark.os, "getpid", lambda: 300)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )

    with pytest.raises(benchmark.BenchmarkContractError, match=r"pids=\[400\]"):
        benchmark._assert_exclusive_local_compute()


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict, dict]:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"fake-model-not-a-real-gguf")
    server = tmp_path / "llama-server"
    server.write_bytes(b"#!/bin/sh\nexit 0\n")
    server.chmod(0o755)
    workload = {
        "schema_version": benchmark.WORKLOAD_SCHEMA,
        "requests": [
            {
                "request_id": "request-1",
                "chronological_index": 10,
                "endpoint": "/v1/completions",
                "body": {"prompt": "Visible prompt only"},
            },
            {
                "request_id": "request-2",
                "chronological_index": 20,
                "endpoint": "/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": "Next"}], "stream": True},
            },
        ],
    }
    workload_path = tmp_path / "workload.json"
    workload_raw = write_json(workload_path, workload)
    _, policy_spec_sha256 = canonical_policy_spec()
    manifest = {
        "schema_version": benchmark.MANIFEST_SCHEMA,
        "study_id": "unit-study",
        "frozen": True,
        "git_revision": "1" * 40,
        "patch_sha256": "2" * 64,
        "llama_cpp_revision": "3" * 40,
        "runtime_fingerprint": "4" * 64,
        "policy_id": "pcss.exact-suffix-ladder/v1",
        "policy_spec_sha256": policy_spec_sha256,
        "hardware_os": {"hardware": "unit machine", "os": "unit os"},
        "workload_sha256": hashlib.sha256(workload_raw).hexdigest(),
        "workload_class": "synthetic_control",
        "holdout": False,
        "workload_provenance": {
            "source": "unit fixture",
            "capture_protocol": "fixture/v1",
            "captured_before_arm_results": True,
            "immutable": True,
        },
        "eligibility_rules": ["all fixture requests are eligible"],
        "calibration_split": {
            "request_ids": ["request-1", "request-2"],
            "tuning_permitted": True,
        },
        "holdout_split": {"request_ids": [], "one_shot": False},
        "clustering": {
            "method": "opaque session cluster",
            "request_cluster_ids": {"request-1": "cluster-a", "request-2": "cluster-b"},
        },
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "server_args": ["--ctx-size", "128", "--threads", "1", "--seed", "17"],
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
        "arm_order": ["pcss-cache", "baseline", "upstream-ngram-cache"],
        "counterbalance": {"method": "latin_square", "sequence_id": "row-2"},
        "state_definitions": {
            "warm": "declared immutable store snapshot loaded before readiness",
            "cold": "fresh empty per-arm state before process launch",
        },
        "arm_store_initialization": {
            "baseline": {
                "state": "cold",
                "initialization": "fresh process with no speculative store",
                "snapshot_sha256": None,
            },
            "upstream-ngram-cache": {
                "state": "cold",
                "initialization": "fresh process with empty in-memory ngram cache",
                "snapshot_sha256": None,
            },
            "pcss-cache": {
                "state": "cold",
                "initialization": "fresh empty isolated PCSS store",
                "snapshot_sha256": None,
            },
        },
        "readiness": {
            "path": "/health",
            "timeout_seconds": 2,
            "max_attempts": 4,
            "interval_seconds": 0.01,
        },
        "request_timeout_seconds": 3,
        "request_count": 2,
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
            "max_requests": 6,
            "max_wall_clock_seconds": 60,
            "max_artifact_bytes": 10_000_000,
        },
        "stopping_rule": "stop after the frozen chronological workload is consumed once",
        "kill_gates": ["server crash", "request timeout", "resource ceiling breach"],
        "exclusion_rules": ["exclude nothing after arm results are visible"],
        "claim_boundary": "synthetic capability only; no general speedup claim",
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, workload_path, model, server, manifest, workload


class FakeProcess:
    def __init__(self, command: list[str], **kwargs: object) -> None:
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.signal = None
        self.pid = 4242

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, value: int) -> None:
        self.signal = value
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def test_persistent_only_pcss_arm_passes_the_explicit_fallback_flag(tmp_path: Path) -> None:
    _, _, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["arms"][2]["pcss"]["disable_current_context"] = True
    manifest["arms"][2]["pcss"]["enable_current_source"] = True
    manifest["arms"][2]["pcss"]["max_draft_tokens"] = 2
    command = benchmark._server_command(
        server_path=server,
        model_path=model,
        manifest=manifest,
        arm_name="pcss-cache",
        port=19333,
        store_path=tmp_path / "store",
    )
    assert "--pcss-disable-current-context" in command
    assert "--pcss-enable-current-source" in command
    assert command[command.index("--pcss-max-draft-tokens") + 1] == "2"


def test_ngram_arm_uses_the_proven_safe_draft_cap(tmp_path: Path) -> None:
    _, _, model, server, manifest, _ = make_inputs(tmp_path)
    command = benchmark._server_command(
        server_path=server,
        model_path=model,
        manifest=manifest,
        arm_name="upstream-ngram-cache",
        port=19333,
        store_path=tmp_path / "store",
    )
    assert command[command.index("--spec-draft-n-max") + 1] == "2"


def test_benchmark_refuses_unproven_pcss_draft_cap(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["arms"][2]["pcss"]["max_draft_tokens"] = 3
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="proven-safe"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_benchmark_binds_explicit_pcss_draft_cap_to_frozen_grid(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["arms"][2]["pcss"]["max_draft_tokens"] = 2
    manifest["finite_parameter_grids"]["draft_tokens"] = [1]
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="finite_parameter_grids"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_benchmark_binds_default_pcss_draft_cap_to_frozen_grid(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["finite_parameter_grids"]["draft_tokens"] = [1]
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="effective pcss"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_policy_study_requires_a_complete_canonical_binding(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    policy_spec_sha256 = manifest["policy_spec_sha256"]
    del manifest["policy_spec_sha256"]
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="declared together"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    manifest["policy_spec_sha256"] = policy_spec_sha256
    manifest["policy_id"] = "PCSS.Exact-Suffix-Ladder/v1"
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="canonical non-legacy"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_legacy_manifest_without_policy_binding_remains_loadable(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    del manifest["policy_id"]
    del manifest["policy_spec_sha256"]
    write_json(manifest_path, manifest)

    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)

    assert benchmark._policy_binding(inputs.manifest) == {
        "policy_id": None,
        "policy_spec_sha256": None,
    }


def test_agent_assisted_capture_receipt_binds_origin_and_clusters(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    request_episode_map = {
        "request-1": "episode-a",
        "request-2": "episode-b",
    }
    receipt = {
        "schema_version": "pcss.project-workload-capture-receipt/v1",
        "source_event_count": 2,
        "selected_event_count": 2,
        "episodes": ["episode-a", "episode-b"],
        "per_episode": 1,
        "source_sha256": "8" * 64,
        "workload_sha256": manifest["workload_sha256"],
        "capture_tool_version": "pcss-project-capture-proxy/1",
        "origin": "agent_assisted_local_llama_server",
        "request_episode_map": request_episode_map,
        "final_event_sha256": "9" * 64,
        "selection_rule": "first visible pre-forward agent-assisted local llama-server requests",
        "contains_responses": False,
    }
    receipt_path = tmp_path / "capture-receipt.json"
    receipt_raw = write_json(receipt_path, receipt)
    manifest["workload_class"] = "agent_assisted_calibration"
    manifest["workload_provenance"] = {
        **manifest["workload_provenance"],
        "origin": "agent_assisted_local_llama_server",
        "capture_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "source_event_sha256": "8" * 64,
    }
    manifest["calibration_split"]["tuning_permitted"] = False
    manifest["clustering"] = {
        "method": "trusted proxy episode UUID",
        "request_cluster_ids": request_episode_map,
    }
    manifest["counterbalance"] = {
        "method": "single_preregistered_order",
        "sequence_id": "row-1",
    }
    manifest["effect_gates"].update(
        {
            "persistent_32_hit_requests_min": 1,
            "persistent_32_hit_episodes_min": 1,
            "persistent_32_accepted_tokens_min": 1,
            "persistent_64_hit_requests_max": 0,
            "persistent_32_suppressions_max": 0,
            "fallback_disable_events_max": 0,
        }
    )
    write_json(manifest_path, manifest)
    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)

    evidence = benchmark._bound_capture_receipt(
        receipt_path, inputs.manifest, inputs.workload
    )

    assert evidence is not None
    assert evidence["origin"] == "agent_assisted_local_llama_server"
    assert evidence["source_event_sha256"] == "8" * 64
    receipt["request_episode_map"] = {"request-1": "episode-a"}
    receipt_raw = write_json(receipt_path, receipt)
    manifest["workload_provenance"]["capture_receipt_sha256"] = hashlib.sha256(
        receipt_raw
    ).hexdigest()
    write_json(manifest_path, manifest)
    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)
    with pytest.raises(benchmark.BenchmarkContractError, match="bind every frozen"):
        benchmark._bound_capture_receipt(receipt_path, inputs.manifest, inputs.workload)


def test_owner_source_holdout_receipt_binds_private_sidecar_commitments(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    matching_source = current_source_payload()
    matching_source["source_id"] = "sha256:" + hashlib.sha256(
        (json.dumps(matching_source["visible_token_ids"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    mismatch_source = {
        **matching_source,
        "compatibility_fingerprint": "sha256:mismatch",
        "visible_token_ids": [19, 29, 31],
    }
    mismatch_source["source_id"] = "sha256:" + hashlib.sha256(
        (json.dumps(mismatch_source["visible_token_ids"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    workload["requests"][0]["body"]["pcss_current_source"] = matching_source
    workload["requests"][1]["body"]["pcss_current_source"] = mismatch_source
    workload_raw = write_json(workload_path, workload)
    request_episode_map = {"request-1": "episode-a", "request-2": "episode-b"}
    receipt = {
        "schema_version": benchmark.CAPTURE_RECEIPT_SCHEMA_V2,
        "source_event_count": 2,
        "selected_event_count": 2,
        "episodes": ["episode-a", "episode-b"],
        "per_episode": 1,
        "source_sha256": "8" * 64,
        "source_sidecar_sha256": "a" * 64,
        "source_sidecar_event_count": 2,
        "source_attachment_count": 2,
        "source_attachment_commitments": {
            "request-1": benchmark._capture_source_commitment(matching_source),
            "request-2": benchmark._capture_source_commitment(mismatch_source),
        },
        "workload_sha256": hashlib.sha256(workload_raw).hexdigest(),
        "capture_tool_version": "pcss-project-capture-proxy/2",
        "origin": "ordinary_local_llama_server",
        "request_episode_map": request_episode_map,
        "final_event_sha256": "9" * 64,
        "final_sidecar_event_sha256": "b" * 64,
        "selection_rule": "first visible pre-forward ordinary local llama-server requests",
        "contains_responses": False,
    }
    receipt_path = tmp_path / "capture-receipt.json"
    receipt_raw = write_json(receipt_path, receipt)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    manifest["workload_class"] = "owner_representative_holdout"
    manifest["holdout"] = True
    manifest["calibration_split"] = {"request_ids": [], "tuning_permitted": False}
    manifest["holdout_split"] = {
        "request_ids": ["request-1", "request-2"],
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
    manifest["clustering"] = {
        "method": "trusted proxy episode UUID",
        "request_cluster_ids": request_episode_map,
    }
    manifest["arms"][2]["pcss"].update(
        {"enable_current_source": True, "max_draft_tokens": 2}
    )
    manifest["effect_gates"].update(
        {
            "current_source_matching_requests_min": 1,
            "current_source_lookup_requests_min": 1,
            "current_source_hit_requests_min": 1,
            "current_source_hit_episodes_min": 1,
            "current_source_accepted_tokens_min": 1,
            "current_source_peak_index_bytes_min": 1,
            "current_source_rejections_max": 0,
            "current_source_ambiguities_max": 0,
            "current_source_abstentions_max": 0,
            "current_source_mismatch_control_requests_min": 1,
            "current_source_mismatch_abstentions_min": 1,
        }
    )
    write_json(manifest_path, manifest)
    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)

    evidence = benchmark._bound_capture_receipt(
        receipt_path, inputs.manifest, inputs.workload
    )

    assert evidence is not None
    assert evidence["origin"] == "ordinary_local_llama_server"
    assert evidence["source_sidecar_sha256"] == "a" * 64
    receipt["source_attachment_commitments"]["request-1"]["tokens_sha256"] = "0" * 64
    receipt_raw = write_json(receipt_path, receipt)
    manifest["workload_provenance"]["capture_receipt_sha256"] = hashlib.sha256(
        receipt_raw
    ).hexdigest()
    write_json(manifest_path, manifest)
    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)
    with pytest.raises(benchmark.BenchmarkContractError, match="does not bind frozen workload"):
        benchmark._bound_capture_receipt(receipt_path, inputs.manifest, inputs.workload)


def test_run_preserves_raw_samples_uses_frozen_order_and_never_claims_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    processes: list[FakeProcess] = []
    dispatched_bodies: list[dict] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "_assert_exclusive_local_compute", lambda: None)
    monkeypatch.setattr(benchmark, "_reserve_loopback_port", lambda: 19333)
    monkeypatch.setattr(
        benchmark,
        "_process_resource_snapshot",
        lambda process: {"pid": process.pid, "rss_bytes": 4096, "cpu_time": "0:00.01", "error": None},
    )
    monkeypatch.setattr(
        benchmark,
        "_wait_ready",
        lambda process, base_url, readiness: {
            "ready": True,
            "latency_ns": 1,
            "attempts": [{"attempt": 1, "status": 200, "error": None}],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_fetch_metrics",
        lambda base_url, timeout: {
            "status": 200,
            "latency_ns": 1,
            "content_type": "text/plain",
            "body": (
                "llamacpp:pcss_lookups_total 1\n"
                "llamacpp:pcss_store_bytes_written_total 0\n"
                "llamacpp:pcss_policy_info{"
                f'policy_id="{manifest["policy_id"]}",'
                f'policy_spec_sha256="{manifest["policy_spec_sha256"]}"'
                "} 1\n"
            ),
            "error": None,
        },
    )

    def dispatch(base_url: str, endpoint: str, body: dict, timeout: float) -> dict:
        del base_url, endpoint, timeout
        dispatched_bodies.append(body)
        return {
            "status": 200,
            "latency_ns": 100,
            "ttft_ns": 40 if body.get("stream") else None,
            "streaming": body.get("stream") is True,
            "output_text": "same deterministic answer",
            "output_tokens": 3,
            "finish_reason": "stop",
            "server_timings": {"predicted_ms": 1.0},
            "events": [],
            "raw_body_utf8": "{}",
            "error": None,
        }

    monkeypatch.setattr(benchmark, "_dispatch_request", dispatch)
    output = tmp_path / "fresh-output"
    result = benchmark.run_benchmark(
        manifest_path=manifest_path,
        workload_path=workload_path,
        model_path=model,
        server_path=server,
        output_path=output,
    )

    assert [process.command for process in processes]
    assert all(Path(process.command[0]).is_absolute() for process in processes)
    assert all(Path(process.command[process.command.index("--model") + 1]).is_absolute() for process in processes)
    assert result["arm_order"] == ["pcss-cache", "baseline", "upstream-ngram-cache"]
    assert len(processes) == 3
    assert "--spec-type" not in processes[1].command
    assert processes[2].command[-4:] == ["--spec-type", "ngram-cache", "--spec-draft-n-max", "2"]
    assert "--pcss-store" in processes[0].command
    assert all(process.returncode == 0 for process in processes)
    assert result["valid_raw_run"] is True
    assert result["exact_output_gate"]["passed"] is True
    assert result["general_speedup_claim"] is False
    assert result["representative_effect_evaluation_permitted"] is False
    assert result["resource_evidence_complete"] is True
    assert result["measured_request_count"] == 6
    expected_policy_binding = {
        "policy_id": manifest["policy_id"],
        "policy_spec_sha256": manifest["policy_spec_sha256"],
    }
    assert result["policy_binding"] == expected_policy_binding
    for receipt in result["arm_receipts"]:
        assert receipt["startup_evidence"]["policy_binding"] == expected_policy_binding
        assert receipt["startup_evidence"]["command_sha256"] == benchmark._canonical_json_sha256(
            receipt["command"]
        )
    assert processes[0].command[processes[0].command.index("--pcss-policy-id") + 1] == manifest["policy_id"]
    assert processes[0].command[processes[0].command.index("--pcss-policy-spec-sha256") + 1] == manifest["policy_spec_sha256"]
    assert all(receipt["cold_start_readiness_latency_ns"] == 1 for receipt in result["arm_receipts"])
    assert all(receipt["store_growth_bytes"] == 0 for receipt in result["arm_receipts"])
    assert len(list((output / "arms").glob("*/samples/*.json"))) == 6
    assert (output / "result.json").is_file()
    assert (output / "workload.json").read_bytes() == workload_path.read_bytes()
    expected_spec, _ = canonical_policy_spec()
    assert (output / "policy-spec.json").read_bytes() == expected_spec
    assert (output / "result.json").stat().st_mode & 0o777 == 0o444
    assert (output / "stores").stat().st_mode & 0o077 == 0
    assert (output / "stores" / "pcss-cache").stat().st_mode & 0o077 == 0
    assert all("expected_output" not in body for body in dispatched_bodies)
    assert dispatched_bodies[1]["stream"] is True
    sample_path = output / "arms" / "baseline" / "samples" / "000000.json"
    sample = json.loads(sample_path.read_text())
    assert sample["accounted_latency_ns"] == 101

    with pytest.raises(benchmark.BenchmarkContractError, match="refusing to overwrite"):
        benchmark.run_benchmark(
            manifest_path=manifest_path,
            workload_path=workload_path,
            model_path=model,
            server_path=server,
            output_path=output,
        )


def test_source_workload_dispatches_privately_and_archives_only_commitment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    source = current_source_payload()
    mismatch_source = {
        **source,
        "compatibility_fingerprint": "mismatch-abstention-control",
        "source_id": "sha256:" + "b" * 64,
    }
    workload["requests"][0]["body"]["pcss_current_source"] = source
    workload["requests"][1]["body"]["pcss_current_source"] = mismatch_source
    workload["requests"].append(
        {
            "request_id": "request-3",
            "chronological_index": 30,
            "endpoint": "/v1/completions",
            "body": {"prompt": "Ordinary control without a source attachment"},
        }
    )
    workload_raw = write_json(workload_path, workload)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    manifest["request_count"] = 3
    manifest["budgets"]["max_requests"] = 9
    manifest["calibration_split"]["request_ids"].append("request-3")
    manifest["clustering"]["request_cluster_ids"]["request-3"] = "cluster-c"
    manifest["arms"][2]["pcss"]["enable_current_source"] = True
    manifest["effect_gates"].update(
        {
            "current_source_matching_requests_min": 1,
            "current_source_lookup_requests_min": 1,
            "current_source_hit_requests_min": 1,
            "current_source_hit_episodes_min": 1,
            "current_source_accepted_tokens_min": 1,
            "current_source_peak_index_bytes_min": 1,
            "current_source_rejections_max": 0,
            "current_source_ambiguities_max": 0,
            "current_source_abstentions_max": 0,
            "current_source_mismatch_control_requests_min": 1,
            "current_source_mismatch_abstentions_min": 1,
        }
    )
    write_json(manifest_path, manifest)

    dispatched_bodies: list[dict] = []
    processes: list[FakeProcess] = []

    def popen(command: list[str], **kwargs: object) -> FakeProcess:
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.setattr(benchmark, "_assert_exclusive_local_compute", lambda: None)
    monkeypatch.setattr(benchmark, "_reserve_loopback_port", lambda: 19334)
    monkeypatch.setattr(
        benchmark,
        "_process_resource_snapshot",
        lambda process: {
            "pid": process.pid,
            "rss_bytes": 4096,
            "cpu_time": "0:00.01",
            "error": None,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_wait_ready",
        lambda process, base_url, readiness: {
            "ready": True,
            "latency_ns": 1,
            "attempts": [{"attempt": 1, "status": 200, "error": None}],
        },
    )
    monkeypatch.setattr(
        benchmark,
        "_fetch_metrics",
        lambda base_url, timeout: {
            "status": 200,
            "latency_ns": 1,
            "content_type": "text/plain",
            "body": (
                "llamacpp:pcss_lookups_total 1\n"
                "llamacpp:pcss_store_bytes_written_total 0\n"
                "llamacpp:pcss_current_source_lookups_total 0\n"
                "llamacpp:pcss_current_source_hits_total 0\n"
                "llamacpp:pcss_current_source_ambiguities_total 0\n"
                "llamacpp:pcss_current_source_abstentions_total 0\n"
                "llamacpp:pcss_current_source_draft_tokens_total 0\n"
                "llamacpp:pcss_current_source_accepted_tokens_total 0\n"
                "llamacpp:pcss_current_source_rejections_total 0\n"
                "llamacpp:pcss_current_source_peak_index_bytes 0\n"
                "llamacpp:pcss_current_source_mismatch_abstentions_total 0\n"
                "llamacpp:pcss_policy_info{"
                f'policy_id="{manifest["policy_id"]}",'
                f'policy_spec_sha256="{manifest["policy_spec_sha256"]}"'
                "} 1\n"
            ),
            "error": None,
        },
    )

    def dispatch(base_url: str, endpoint: str, body: dict, timeout: float) -> dict:
        del base_url, endpoint, timeout
        dispatched_bodies.append(body)
        return {
            "status": 200,
            "latency_ns": 100,
            "ttft_ns": None,
            "streaming": False,
            "output_text": source["source_id"],
            "output_tokens": 3,
            "finish_reason": "stop",
            "server_timings": {"compatibility_fingerprint": source["compatibility_fingerprint"]},
            "events": [{"pcss_current_source": source}],
            "raw_body_utf8": json.dumps({"echo": source}),
            "error": None,
        }

    monkeypatch.setattr(benchmark, "_dispatch_request", dispatch)
    output = tmp_path / "source-output"
    result = benchmark.run_benchmark(
        manifest_path=manifest_path,
        workload_path=workload_path,
        model_path=model,
        server_path=server,
        output_path=output,
    )

    assert len(processes) == 3
    assert sum("pcss_current_source" in body for body in dispatched_bodies) == 6
    assert all(
        json.dumps(body["pcss_current_source"], sort_keys=True)
        in {json.dumps(source, sort_keys=True), json.dumps(mismatch_source, sort_keys=True)}
        for body in dispatched_bodies
        if "pcss_current_source" in body
    )
    archived = json.loads((output / "workload.json").read_text())
    assert archived["schema_version"] == benchmark.REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA
    assert archived["original_workload_sha256"] == manifest["workload_sha256"]
    archived_request = archived["requests"][0]
    assert archived_request["body"] == {"prompt": "Visible prompt only"}
    assert archived_request["pcss_current_source_commitment"] == {
        "schema_version": benchmark.CURRENT_SOURCE_COMMITMENT_SCHEMA,
        "token_count": 3,
        "tokens_sha256": benchmark._canonical_json_sha256(source["visible_token_ids"]),
        "attachment_mode": benchmark.CURRENT_SOURCE_MATCHING_MODE,
    }
    assert archived["requests"][1]["pcss_current_source_commitment"][
        "attachment_mode"
    ] == benchmark.CURRENT_SOURCE_MISMATCH_CONTROL_MODE
    sample = json.loads(
        (output / "arms" / "baseline" / "samples" / "000000.json").read_text()
    )
    assert sample["request_body"] == {"max_tokens": 4, "temperature": 0, "stream": False, "prompt": "Visible prompt only"}
    assert sample["pcss_current_source_commitment"] == archived_request[
        "pcss_current_source_commitment"
    ]
    assert sample["response"]["events"] == []
    assert sample["response"]["raw_body_utf8"] is None
    assert sample["response"]["output_text"] == "[redacted-current-source]"
    ordinary_sample = json.loads(
        (output / "arms" / "baseline" / "samples" / "000002.json").read_text()
    )
    assert "pcss_current_source_commitment" not in ordinary_sample
    assert ordinary_sample["response"]["events"] == []
    assert ordinary_sample["response"]["raw_body_utf8"] is None
    assert source["source_id"] not in "\n".join(
        path.read_text() for path in output.rglob("*") if path.is_file()
    )
    assert mismatch_source["source_id"] not in "\n".join(
        path.read_text() for path in output.rglob("*") if path.is_file()
    )
    assert "visible_token_ids" not in "\n".join(
        path.read_text() for path in output.rglob("*") if path.is_file()
    )
    assert result["workload_sha256"] == manifest["workload_sha256"]
    assert result["source_workload_redaction"]["redacted_workload_sha256"] == hashlib.sha256(
        (output / "workload.json").read_bytes()
    ).hexdigest()


def test_source_workload_requires_enabled_matching_pcss_configuration(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    source = current_source_payload()
    workload["requests"][0]["body"]["pcss_current_source"] = source
    workload_raw = write_json(workload_path, workload)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    manifest["effect_gates"].update(
        {
            "current_source_matching_requests_min": 1,
            "current_source_lookup_requests_min": 1,
            "current_source_hit_requests_min": 1,
            "current_source_hit_episodes_min": 1,
            "current_source_accepted_tokens_min": 1,
            "current_source_peak_index_bytes_min": 1,
            "current_source_rejections_max": 0,
            "current_source_ambiguities_max": 0,
            "current_source_abstentions_max": 0,
            "current_source_mismatch_control_requests_min": 1,
            "current_source_mismatch_abstentions_min": 1,
        }
    )
    write_json(manifest_path, manifest)

    with pytest.raises(benchmark.BenchmarkContractError, match="enable_current_source"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    manifest["arms"][2]["pcss"]["enable_current_source"] = True
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="mismatch abstention control"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    workload["requests"][1]["body"]["pcss_current_source"] = {
        **source,
        "compatibility_fingerprint": "mismatch-abstention-control",
        "source_id": "sha256:" + "b" * 64,
    }
    workload_raw = write_json(workload_path, workload)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    write_json(manifest_path, manifest)
    assert benchmark.load_inputs(manifest_path, workload_path, model, server).workload


def test_manifest_and_workload_reject_unfrozen_mismatch_and_future_material(
    tmp_path: Path,
) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    manifest["frozen"] = False
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="frozen=true"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    workload["requests"][0]["body"]["expected_output"] = "leaked suffix"
    workload_raw = write_json(workload_path, workload)
    manifest["frozen"] = True
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="future/output material"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


@pytest.mark.parametrize(
    "field",
    (
        "git_revision",
        "patch_sha256",
        "llama_cpp_revision",
        "runtime_fingerprint",
        "hardware_os",
        "workload_provenance",
        "eligibility_rules",
        "calibration_split",
        "holdout_split",
        "clustering",
        "state_definitions",
        "arm_store_initialization",
        "resource_ceilings",
        "allowed_policy_families",
        "finite_parameter_grids",
        "budgets",
        "stopping_rule",
        "kill_gates",
        "exclusion_rules",
    ),
)
def test_manifest_requires_every_preregistration_binding(tmp_path: Path, field: str) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    del manifest[field]
    write_json(manifest_path, manifest)

    with pytest.raises(benchmark.BenchmarkContractError, match="keys differ from schema"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_effect_resource_grid_and_native_disk_contracts_are_bounded(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["effect_gates"]["miss_heavy_p95_overhead_ucb95_max"] = 0.031
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="cannot exceed 3%"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    manifest["effect_gates"]["miss_heavy_p95_overhead_ucb95_max"] = 0.03
    manifest["arms"][2]["pcss"]["max_disk_gib"] = 0.01
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="integer >= 1"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    manifest["arms"][2]["pcss"]["max_disk_gib"] = 1
    manifest["finite_parameter_grids"] = {"draft_tokens": []}
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="1..1000"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


def test_owner_representative_holdout_is_full_untuned_and_one_shot(tmp_path: Path) -> None:
    manifest_path, workload_path, model, server, manifest, _ = make_inputs(tmp_path)
    manifest["workload_class"] = "owner_representative_holdout"
    manifest["holdout"] = True
    manifest["calibration_split"] = {"request_ids": [], "tuning_permitted": False}
    manifest["holdout_split"] = {
        "request_ids": ["request-1", "request-2"],
        "one_shot": True,
    }
    receipt = {
        "schema_version": benchmark.CAPTURE_RECEIPT_SCHEMA_V1,
        "source_event_count": 2,
        "selected_event_count": 2,
        "episodes": ["cluster-a", "cluster-b"],
        "per_episode": 1,
        "source_sha256": "8" * 64,
        "workload_sha256": manifest["workload_sha256"],
        "capture_tool_version": "pcss-project-capture-proxy/1",
        "origin": "ordinary_local_llama_server",
        "request_episode_map": manifest["clustering"]["request_cluster_ids"],
        "final_event_sha256": "9" * 64,
        "selection_rule": "first visible pre-forward ordinary local llama-server requests",
        "contains_responses": False,
    }
    receipt_path = tmp_path / "capture-receipt.json"
    receipt_raw = write_json(receipt_path, receipt)
    manifest["workload_provenance"] = {
        **manifest["workload_provenance"],
        "origin": "ordinary_local_llama_server",
        "capture_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "source_event_sha256": "8" * 64,
    }
    write_json(manifest_path, manifest)
    inputs = benchmark.load_inputs(manifest_path, workload_path, model, server)
    with pytest.raises(benchmark.BenchmarkContractError, match="requires a sealed capture receipt"):
        benchmark._bound_capture_receipt(None, inputs.manifest, inputs.workload)
    assert benchmark._bound_capture_receipt(
        receipt_path, inputs.manifest, inputs.workload
    ) is not None

    marker = benchmark._claim_holdout_once(manifest_path, inputs)
    assert marker is not None and marker.is_file()
    with pytest.raises(benchmark.BenchmarkContractError, match="already opened"):
        benchmark._claim_holdout_once(manifest_path, inputs)

    marker.chmod(0o644)
    marker.unlink()
    manifest["holdout_split"]["one_shot"] = False
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="one-shot"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

def test_workload_must_remain_chronological_and_manifest_pins_request_count(
    tmp_path: Path,
) -> None:
    manifest_path, workload_path, model, server, manifest, workload = make_inputs(tmp_path)
    workload["requests"][1]["chronological_index"] = 9
    workload_raw = write_json(workload_path, workload)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="chronological order"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)

    workload["requests"][1]["chronological_index"] = 20
    workload_raw = write_json(workload_path, workload)
    manifest["workload_sha256"] = hashlib.sha256(workload_raw).hexdigest()
    manifest["request_count"] = 3
    write_json(manifest_path, manifest)
    with pytest.raises(benchmark.BenchmarkContractError, match="request_count"):
        benchmark.load_inputs(manifest_path, workload_path, model, server)


class FakeStreamingResponse:
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self) -> None:
        self.lines = iter(
            (
                b'data: {"choices":[{"text":"hello","finish_reason":null}]}\n',
                b'data: {"choices":[{"text":" world","finish_reason":"stop"}],'
                b'"usage":{"completion_tokens":2},"timings":{"predicted_ms":3}}\n',
                b"data: [DONE]\n",
            )
        )

    def __enter__(self) -> "FakeStreamingResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def readline(self) -> bytes:
        return next(self.lines, b"")


def test_streaming_dispatch_captures_ttft_tokens_timings_and_raw_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "urlopen", lambda request, timeout: FakeStreamingResponse())
    result = benchmark._dispatch_request(
        "http://127.0.0.1:8080",
        "/v1/completions",
        {"prompt": "p", "stream": True},
        1,
    )

    assert result["error"] is None
    assert result["output_text"] == "hello world"
    assert result["output_tokens"] == 2
    assert result["finish_reason"] == "stop"
    assert result["ttft_ns"] is not None
    assert result["server_timings"] == {"predicted_ms": 3}
    assert len(result["events"]) == 2
    assert "data: [DONE]" in result["raw_body_utf8"]


def test_readiness_stops_at_preregistered_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def unavailable(url: str, timeout: float) -> None:
        nonlocal attempts
        del url, timeout
        attempts += 1
        raise URLError("not ready")

    monkeypatch.setattr(benchmark, "urlopen", unavailable)
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    process = FakeProcess(["server"])
    with pytest.raises(benchmark.BenchmarkContractError, match="bounds exhausted"):
        benchmark._wait_ready(
            process,
            "http://127.0.0.1:9999",
            {"path": "/health", "timeout_seconds": 10, "max_attempts": 3, "interval_seconds": 0},
        )
    assert attempts == 3


def test_exact_output_gate_reports_text_mismatch() -> None:
    def row(arm: str, text: str) -> dict:
        return {
            "arm": arm,
            "request_id": "request-1",
            "response": {
                "error": None,
                "status": 200,
                "output_text": text,
                "output_tokens": 1,
                "finish_reason": "stop",
            },
        }

    gate = benchmark._exact_output_gate(
        {
            "baseline": [row("baseline", "a")],
            "upstream-ngram-cache": [row("upstream-ngram-cache", "a")],
            "pcss-cache": [row("pcss-cache", "b")],
        }
    )

    assert gate["passed"] is False
    assert gate["comparisons"][-1]["reasons"] == ["output_text_mismatch"]


def test_exact_output_gate_can_keep_an_unsafe_reporting_comparator_nonrequired() -> None:
    def row(arm: str, text: str) -> dict:
        return {
            "arm": arm,
            "request_id": "request-1",
            "response": {"error": None, "status": 200, "output_text": text, "output_tokens": 1, "finish_reason": "stop"},
        }

    gate = benchmark._exact_output_gate(
        {
            "baseline": [row("baseline", "a")],
            "upstream-ngram-cache": [row("upstream-ngram-cache", "different")],
            "pcss-cache": [row("pcss-cache", "a")],
        },
        required_arms=("pcss-cache",),
    )
    assert gate["passed"] is True
    assert gate["comparisons"][0]["required"] is False
    assert gate["comparisons"][0]["passed"] is False


def test_benchmark_script_requires_all_explicit_inputs() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark.sh"
    completed = subprocess.run(("bash", str(script)), capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert "MANIFEST.json WORKLOAD.json MODEL.gguf OUTPUT_DIRECTORY" in completed.stderr
