from __future__ import annotations

from pathlib import Path

import pytest

from pcss.adaptive_datastore import PolicyName, SequenceMode, SourceCondition
from pcss.adaptive_datastore_study import (
    configurations,
    dataset_metadata,
    load_trace_fixture,
    run_configuration,
    run_study,
    _verifier_compatibility,
)
from pcss.canonical import ContractError
from pcss.replay import Fingerprints, ReplayRecord, ScopeIdentity, record_from_mapping, replay_chronologically


ROOT = Path(__file__).parents[1]
FINGERPRINTS = Fingerprints("model", "tokenizer", "template", "protocol")


def record(record_id: str, ordinal: int, committed: tuple[int, ...], prompt: tuple[int, ...] = ()) -> ReplayRecord:
    return ReplayRecord(
        record_id=record_id,
        timestamp_ns=ordinal,
        chronological_index=ordinal,
        request_id=record_id,
        prompt_tokens=prompt,
        committed_tokens=committed,
        fingerprints=FINGERPRINTS,
        scope=ScopeIdentity(
            principal_namespace="principal",
            application_id="application",
            repository_id="repository",
            workflow_stage="edit",
            semantic_block_id="block",
            session_id="session",
            revision_family="git",
            path_hash="path",
            before_blob="before",
        ),
        source_class="visible_code",
        provenance_family="repository",
        eligible_for_persistent_store=True,
    )


def test_fixed_matrix_has_one_legacy_and_forty_redesigned_configurations() -> None:
    values = configurations()
    assert len(values) == 41
    assert values[0] == (PolicyName.LEGACY_SPARSE, SequenceMode.PROMPT_PLUS_COMPLETION, 2)
    assert len({(policy, mode, k) for policy, mode, k in values}) == 41


def test_dataset_metadata_keeps_trace_identity_and_clusters() -> None:
    path = ROOT / "fixtures" / "replay" / "chronological-oracle-v1.json"
    import json

    records = tuple(record_from_mapping(item) for item in json.loads(path.read_text(encoding="utf-8")))
    metadata = dataset_metadata(path, records)
    assert metadata["sha256"]
    assert metadata["record_count"] == 2
    assert metadata["independent_cluster_count"] == 2
    assert metadata["chronological"] is True


def test_run_configuration_separates_all_positions_from_non_overlapping_online_steps() -> None:
    key = tuple(range(16))
    source = record("source", 1, key + (7, 8, 9))
    target = record("target", 2, (7, 8, 9), prompt=key)
    result = replay_chronologically((source, target), min_suffix=1, max_continuation=32)
    opportunities = {(item.record_id, item.position): item for item in result.opportunities}
    output = run_configuration((source, target), PolicyName.DENSE_MODAL, SequenceMode.PROMPT_PLUS_COMPLETION, 4, opportunities)
    persistent = output["source_conditions"][SourceCondition.PERSISTENT_ONLY.value]
    assert persistent["all_position_metrics"]["proposals"] >= persistent["online_step_metrics"]["proposals"]
    assert persistent["online_step_metrics"]["steps"] <= persistent["output_tokens"]
    assert persistent["persistent_unique_accepted_tokens"] == persistent["online_persistent_unique_accepted_tokens"]
    assert 0.0 <= persistent["persistent_unique_coverage"] <= 1.0
    assert persistent["all_position_persistent_unique_accepted_tokens"] >= persistent["persistent_unique_accepted_tokens"]


def test_run_study_records_fixed_source_priority_and_no_speed_without_profile(tmp_path: Path) -> None:
    trace = ROOT / "fixtures" / "replay" / "chronological-oracle-v1.json"
    output = run_study((trace,), seed=7, bootstrap_resamples=0, phase="GRID_PASS", repository_root=ROOT)
    assert output["source_tier_priority"][:3] == ["current_context", "current_source", "same_session"]
    assert output["configuration_count"] == 41
    assert len(output["results"]) == 41
    assert output["selection"]["economic_measurement_status"] == "NOT_COMPUTED"


def test_study_output_refuses_existing_paths(tmp_path: Path) -> None:
    trace = ROOT / "fixtures" / "replay" / "chronological-oracle-v1.json"
    output = tmp_path / "result.json"
    report = tmp_path / "report.md"
    from pcss.adaptive_datastore_study import main

    assert main(["--trace", str(trace), "--output", str(output), "--report", str(report)]) == 0
    with pytest.raises(ContractError, match="output path already exists"):
        main(["--trace", str(trace), "--output", str(output), "--report", str(tmp_path / "report-2.md")])


def test_trace_chronological_fixture_is_unit_sanity() -> None:
    path = ROOT / "fixtures" / "traces" / "chronological-v1.jsonl"
    records = tuple(record_from_mapping(item) for item in load_trace_fixture(path))
    assert dataset_metadata(path, records)["evidence_class"] == "UNIT_SANITY"


def test_verifier_profile_requires_matching_trace_model() -> None:
    records = (record("source", 1, (1, 2, 3)),)
    with pytest.raises(ContractError, match="model fingerprint"):
        _verifier_compatibility(records, {"model_artifact_sha256": "other-model"})
    compatible = _verifier_compatibility(records, {"model_artifact_sha256": "model", "backend": "llama.cpp"})
    assert compatible["status"] == "COMPATIBLE_MODEL_HASH"
