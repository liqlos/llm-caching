"""Controlled sealed-archive lookup plus isolated bridge-decode harness.

The archive is invoked outside the bridge before each measured synthetic request;
its measured cost is added to the bridge candidate clock. This is deliberately
not a live llama.cpp provider integration or a production decode path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import monotonic_ns, perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json
from .exact_prompt_archive import ExactPromptArchive, open_exact_prompt_archive
from .exact_prompt_provider import ExactPromptArchiveShadowProvider, ExactPromptDescriptor
from .llamacpp_fixture_builder import BridgeConfig, load_bridge_config
from .llamacpp_live_benchmark import _pair, _request, _run_bridge, _validate_response
from .manifest import validate_manifest
from .metrics import clustered_bootstrap_ci
from .offline_runner import load_trace_fixture
from .provider import Feedback, Fingerprints, Proposal, RequestContext, TrustedScope
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


def _fingerprints(record: Mapping[str, Any]) -> Fingerprints:
    return Fingerprints(
        model=record["model_fingerprint"], tokenizer=record["tokenizer_fingerprint"],
        chat_template=record["chat_template_fingerprint"],
        generation_protocol=record["generation_protocol_fingerprint"],
    )


def _scope(record: Mapping[str, Any], *, widening_level: int) -> TrustedScope:
    trusted = record["trusted_scope"]
    lineage = record["artifact_lineage"]
    return TrustedScope(
        principal_namespace=trusted["principal_namespace"], application=trusted["application_id"] or "unknown",
        repository=trusted["repository_id"], revision_family=lineage["revision_family"],
        operation_class=trusted["workflow_stage"] or "unknown", workflow_block=trusted["semantic_block_id"],
        schema_or_tool_version=record["generation_protocol"]["schema_or_tool_version"], widening_level=widening_level,
    )


def _candidate_from_archive(
    archive: ExactPromptArchive,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    output_tokens: int,
    require_hit: bool = True,
) -> tuple[tuple[int, ...], int, ExactPromptArchiveShadowProvider, RequestContext, Proposal | None]:
    prompt = tuple(target["content"]["prompt_token_ids"])
    fingerprints = _fingerprints(target)
    scope = _scope(target, widening_level=1)
    descriptor = ExactPromptDescriptor(
        tail_tokens=prompt, source_record_id=source["record_id"], source_class="same_repository",
        scope=scope, fingerprints=fingerprints, provenance_family=source["source"]["provenance_family"],
        matched_suffix_tokens=len(prompt),
    )
    provider = ExactPromptArchiveShadowProvider(archive, (descriptor,))
    request = RequestContext(
        request_id=target["request_id"], fingerprints=fingerprints, scope=scope,
        source_tiers=("same_repository",), deadline_ns=monotonic_ns() + 60_000_000_000,
        probe_budget=1,
    )
    started = perf_counter_ns()
    provider.begin_request(request)
    proposal = provider.propose(request.request_id, tail_tokens=prompt, max_tokens=output_tokens, deadline_ns=request.deadline_ns)
    elapsed = perf_counter_ns() - started
    if not isinstance(proposal, Proposal):
        if require_hit:
            raise ContractError("exact-prompt archive provider abstained during a required exact hit")
        return (), elapsed, provider, request, None
    candidate = proposal.tokens
    if len(candidate) != output_tokens:
        raise ContractError("exact-prompt archive cursor is shorter than the requested output")
    return candidate, elapsed, provider, request, proposal


def run_exact_prompt_live_harness(
    *,
    config_path: Path,
    manifest_path: Path,
    trace_path: Path,
    archive_root: Path,
    pairs: Sequence[str],
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    miss_pairs: Sequence[str] = (),
    rejection_pairs: Sequence[str] = (),
    k: int = 2,
    output_tokens: int = 48,
    warmup_repetitions: int = 2,
    measurement_repetitions: int = 8,
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    bridge_runner: Callable[[BridgeConfig, Mapping[str, Any]], Mapping[str, Any]] = _run_bridge,
) -> ResultReceipt:
    all_pairs = tuple(pairs) + tuple(miss_pairs) + tuple(rejection_pairs)
    if not pairs or len(set(all_pairs)) != len(all_pairs):
        raise ContractError("exact-prompt live harness requires unique hit, miss, and rejection pairs")
    if warmup_repetitions < 0 or measurement_repetitions < 2:
        raise ContractError("exact-prompt live harness requires non-negative warmups and two measurements")
    config = load_bridge_config(config_path)
    records = load_trace_fixture(trace_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    host = target_observer()
    if manifest["evidence_class"] != "LOCAL_MEASURED" or manifest["verifier_gate"]["host_identity"] != host.as_dict():
        raise ContractError("exact-prompt live harness requires matching LOCAL_MEASURED target manifest")
    model_hash = file_sha256(config.model_artifact)
    binding = manifest["verifier_gate"]["adapter_binding"]
    if manifest["model_artifact"]["sha256"] != model_hash or binding["model_artifact_sha256"] != model_hash:
        raise ContractError("exact-prompt live harness model differs from observed manifest")
    bridge_hash = file_sha256(config.bridge_executable)
    if binding["backend_binary_sha256"] != bridge_hash or binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint:
        raise ContractError("exact-prompt live harness bridge or runtime differs from observed manifest")
    archive = open_exact_prompt_archive(archive_root)
    cases: list[dict[str, Any]] = []
    ratios_by_request: dict[str, list[float]] = {}
    request_fingerprints: dict[str, str] = {}
    work_items = (
        tuple((pair, "exact_hit") for pair in pairs)
        + tuple((pair, "edited_miss") for pair in miss_pairs)
        + tuple((pair, "controlled_rejection") for pair in rejection_pairs)
    )
    for pair_spec, cluster in work_items:
        require_hit = cluster != "edited_miss"
        source, target = _pair(records, pair_spec, pair_mode="exact_repeat" if require_hit else "source_reuse")
        for _ in range(warmup_repetitions):
            _, _, provider, provider_request, proposal = _candidate_from_archive(
                archive, source, target, output_tokens=output_tokens, require_hit=require_hit
            )
            if proposal is not None:
                provider.feedback(provider_request.request_id, Feedback(proposal.proposal_id, "not_verified", 0, (), None, None))
            provider.end_request(provider_request.request_id)
        candidates_and_times = [
            _candidate_from_archive(archive, source, target, output_tokens=output_tokens, require_hit=require_hit)
            for _ in range(measurement_repetitions)
        ]
        candidate = candidates_and_times[0][0]
        if any(item[0] != candidate for item in candidates_and_times):
            raise ContractError("exact-prompt archive lookup is not deterministic across measurements")
        request = _request(
            config, source, target, k=k, output_tokens=output_tokens,
            warmup_repetitions=warmup_repetitions, measurement_repetitions=measurement_repetitions,
            candidate_tokens=candidate,
            operation="live_pending",
            forced_rejection_position=(output_tokens // 2 if cluster == "controlled_rejection" else None),
        )
        response = _validate_response(
            bridge_runner(config, request), request, config,
            expected_authoritative_tokens=target["content"]["committed_token_ids"],
        )
        lookup_times = [item[1] for item in candidates_and_times]
        ratios: list[float] = []
        combined_rows: list[dict[str, Any]] = []
        for row, lookup_ns in zip(response["measurements"], lookup_times, strict=True):
            paired_serial = (row["serial_before_ns"] + row["serial_after_ns"]) / 2.0
            combined_candidate_ns = row["candidate_decode_ns"] + lookup_ns
            ratios.append(paired_serial / combined_candidate_ns)
            combined_rows.append({
                "iteration": row["iteration"], "archive_lookup_ns": lookup_ns,
                "bridge_candidate_decode_ns": row["candidate_decode_ns"],
                "lookup_inclusive_candidate_ns": combined_candidate_ns,
            })
        provider_telemetry: list[dict[str, Any]] = []
        for _, _, provider, provider_request, proposal in candidates_and_times:
            if proposal is not None:
                provider.feedback(provider_request.request_id, Feedback(proposal.proposal_id, "not_verified", 0, (), None, None))
            provider.end_request(provider_request.request_id)
            provider_telemetry.extend(
                {"event": event.event, "request_id": event.request_id, "at_ns": event.at_ns, **event.payload}
                for event in provider.telemetry
            )
        ratios_by_request[target["request_id"]] = ratios
        request_fingerprints[response["case_id"]] = hashlib.sha256(canonical_json(request).encode()).hexdigest()
        cases.append({
            "source_record_id": source["record_id"], "target_record_id": target["record_id"],
            "request_id": target["request_id"], "response": response,
            "workload_cluster": cluster,
            "archive_measurements": combined_rows, "provider_telemetry": provider_telemetry,
            "paired_lookup_inclusive_ratio_mean": fmean(ratios),
        })
    confidence = clustered_bootstrap_ci(ratios_by_request)
    analysis = {
        "hypothesis": (
            "On the fixed mixed workload, the paired serial-over-candidate end-to-end decode ratio "
            "has a clustered 95% lower bound greater than 1.0 while visible outputs, final state, "
            "and continuation remain exact."
        ),
        "kill_gate": (
            "Declare NO_GO_LIVE_PENDING_MIXED if any native output/state check fails, or if the "
            "clustered 95% lower bound of the paired ratio is less than or equal to 1.0."
        ),
        "case_count": len(cases),
        "workload_clusters": {cluster: sum(case["workload_cluster"] == cluster for case in cases)
                              for cluster in ("exact_hit", "edited_miss", "controlled_rejection")},
        "all_exact_output_and_state": all(case["response"]["exact_output_and_state"] for case in cases),
        "paired_serial_over_lookup_inclusive_candidate_ratio": asdict(confidence),
        "decision": "LIVE_PENDING_MIXED_POSITIVE" if confidence.lower > 1.0 else "NO_GO_LIVE_PENDING_MIXED",
    }
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.exact_prompt_live_harness", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "manifest": file_sha256(manifest_path), "trace": file_sha256(trace_path),
            "archive_manifest": file_sha256(archive_root / "archive-manifest.json"),
            "bridge_binary": bridge_hash, "config": file_sha256(config.config_path),
            "model_artifact": model_hash, "llama_cpp_library": file_sha256(config.llama_cpp_library),
            **request_fingerprints,
        },
        status=analysis["decision"],
        payload={
            "kind": "sealed_exact_prompt_archive_to_isolated_bridge_harness",
            "host_identity": host.as_dict(), "analysis": analysis, "cases": cases,
            "archive_byte_accounting": archive.archive.byte_accounting(),
            "runtime_claim": "external sealed archive provider plus single-context live bridge decode; hit and abstention lookup costs are inside the candidate clock, but this remains an isolated bridge harness rather than a llama.cpp provider hook",
            "not_claimed": ["end_to_end_request_speedup", "representative_workload_speedup", "thermal_sustainability", "online_storage"],
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS sealed exact-prompt archive bridge harness")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--miss-pair", action="append", default=[])
    parser.add_argument("--rejection-pair", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=48)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--measurement-repetitions", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parser().parse_args(arguments)
    run_exact_prompt_live_harness(
        config_path=parsed.config, manifest_path=parsed.manifest, trace_path=parsed.trace,
        archive_root=parsed.archive_root, pairs=parsed.pair, miss_pairs=parsed.miss_pair,
        rejection_pairs=parsed.rejection_pair, output_path=parsed.output,
        report_path=parsed.report, repository_root=parsed.repository_root, command_arguments=arguments,
        k=parsed.k, output_tokens=parsed.output_tokens, warmup_repetitions=parsed.warmup_repetitions,
        measurement_repetitions=parsed.measurement_repetitions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
