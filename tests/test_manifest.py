from __future__ import annotations

import json
from pathlib import Path

import pytest

from pcss.canonical import ContractError
from pcss.manifest import validate_manifest


FIXTURE = Path(__file__).parents[1] / "fixtures" / "manifests" / "baseline-unfilled.json"


def load_manifest() -> dict:
    return json.loads(FIXTURE.read_text())


def test_baseline_manifest_keeps_target_unknown() -> None:
    manifest = load_manifest()
    validate_manifest(manifest)
    assert manifest["target"]["hardware"]["value"] is None
    assert manifest["evidence_class"] == "NOT_RUN"


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
