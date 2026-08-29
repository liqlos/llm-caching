from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.canonical import sha256
from pcss.manifest import validate_manifest


FIXTURE = Path(__file__).parents[1] / "fixtures" / "manifests" / "baseline-unfilled.json"


def load_manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def test_baseline_manifest_keeps_target_unknown() -> None:
    manifest = load_manifest()
    validate_manifest(manifest)
    assert manifest["target"]["hardware"]["value"] is None
    assert manifest["evidence_class"] == "NOT_RUN"
    assert manifest["verifier_gate"] == {
        "observation_status": "not_observed",
        "host_identity": None,
        "adapter_binding": None,
    }


def test_unknown_m4_value_is_rejected_before_observation() -> None:
    manifest = load_manifest()
    manifest["target"]["hardware"]["value"] = {"chip": "invented"}
    with pytest.raises(ContractError, match="must be null"):
        validate_manifest(manifest)


def test_manifest_rejects_untracked_fields() -> None:
    manifest = load_manifest()
    manifest["speedup_claim"] = 2.0
    with pytest.raises(ContractError, match="unknown"):
        validate_manifest(manifest)


def observed_verifier_gate() -> dict:
    host_identity = {
        "schema_version": "pcss.host-identity/v1",
        "target_profile": "contract-test-host",
        "operating_system": "contract-test-os",
        "architecture": "contract-test-arch",
        "hardware_model": "contract-test-model",
        "chip": "contract-test-chip",
        "memory_bytes": 1,
    }
    host_identity["fingerprint"] = sha256(host_identity)
    return {
        "observation_status": "observed",
        "host_identity": host_identity,
        "adapter_binding": {
            "schema_version": "pcss.verifier-adapter-binding/v1",
            "adapter_source_sha256": "a" * 64,
            "backend_binary_sha256": "b" * 64,
            "model_artifact_sha256": "c" * 64,
            "llama_cpp_build_fingerprint": "d" * 64,
        },
    }


def test_observed_verifier_gate_requires_complete_bound_identity() -> None:
    manifest = load_manifest()
    manifest["verifier_gate"] = observed_verifier_gate()
    validate_manifest(manifest)

    manifest = load_manifest()
    manifest["verifier_gate"] = observed_verifier_gate()
    manifest["verifier_gate"]["host_identity"] = None
    with pytest.raises(ContractError, match="requires host_identity and adapter_binding"):
        validate_manifest(manifest)

    manifest = load_manifest()
    manifest["verifier_gate"] = observed_verifier_gate()
    manifest["verifier_gate"]["host_identity"]["fingerprint"] = "0" * 64
    with pytest.raises(ContractError, match="does not match"):
        validate_manifest(manifest)


def test_unobserved_verifier_gate_cannot_be_prefilled() -> None:
    manifest = load_manifest()
    manifest["verifier_gate"]["host_identity"] = observed_verifier_gate()["host_identity"]
    with pytest.raises(ContractError, match="must be null"):
        validate_manifest(manifest)
