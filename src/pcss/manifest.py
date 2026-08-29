"""Baseline evidence manifests; values unknown on a target stay explicit."""

from __future__ import annotations

from typing import Any

from .canonical import ContractError, require_exact_keys, require_mapping, require_string

MANIFEST_SCHEMA_VERSION = "pcss.manifest/v1"
EVIDENCE_CLASSES = frozenset({"SIMULATOR", "LOCAL_MEASURED", "NOT_RUN"})
OBSERVATION_STATUSES = frozenset({"not_observed", "observed"})

_TOP_LEVEL = {
    "schema_version",
    "run_id",
    "evidence_class",
    "target",
    "model_artifact",
    "quantization",
    "tokenizer",
    "chat_template",
    "generation_protocol",
    "llama_cpp",
    "context_kv",
    "benchmark",
}
_TARGET_KEYS = {"hardware", "operating_system"}
_OBSERVED_VALUE_KEYS = {"observation_status", "value"}
_MODEL_KEYS = {"observation_status", "artifact_uri", "sha256"}
_PROTOCOL_KEYS = {"observation_status", "fingerprint", "parameters"}


def _validate_observed_value(value: Any, field: str) -> None:
    obj = require_mapping(value, field)
    require_exact_keys(obj, _OBSERVED_VALUE_KEYS, field)
    status = require_string(obj["observation_status"], f"{field}.observation_status")
    if status not in OBSERVATION_STATUSES:
        raise ContractError(f"{field}.observation_status is invalid")
    if status == "not_observed" and obj["value"] is not None:
        raise ContractError(f"{field}.value must be null until observed")
    if status == "observed" and obj["value"] is None:
        raise ContractError(f"{field}.value is required when observed")


def _validate_model(value: Any, field: str) -> None:
    obj = require_mapping(value, field)
    require_exact_keys(obj, _MODEL_KEYS, field)
    status = require_string(obj["observation_status"], f"{field}.observation_status")
    if status not in OBSERVATION_STATUSES:
        raise ContractError(f"{field}.observation_status is invalid")
    for key in ("artifact_uri", "sha256"):
        item = obj[key]
        if status == "not_observed" and item is not None:
            raise ContractError(f"{field}.{key} must be null until observed")
        if status == "observed":
            require_string(item, f"{field}.{key}")


def _validate_protocol(value: Any, field: str) -> None:
    obj = require_mapping(value, field)
    require_exact_keys(obj, _PROTOCOL_KEYS, field)
    status = require_string(obj["observation_status"], f"{field}.observation_status")
    if status not in OBSERVATION_STATUSES:
        raise ContractError(f"{field}.observation_status is invalid")
    if status == "not_observed":
        if obj["fingerprint"] is not None or obj["parameters"] is not None:
            raise ContractError(f"{field} details must be null until observed")
    else:
        require_string(obj["fingerprint"], f"{field}.fingerprint")
        require_mapping(obj["parameters"], f"{field}.parameters")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate a complete baseline manifest without filling unknown facts."""
    obj = require_mapping(manifest, "manifest")
    require_exact_keys(obj, _TOP_LEVEL, "manifest")
    if obj["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ContractError("unsupported manifest.schema_version")
    require_string(obj["run_id"], "manifest.run_id")
    evidence_class = require_string(obj["evidence_class"], "manifest.evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ContractError("manifest.evidence_class is invalid")

    target = require_mapping(obj["target"], "manifest.target")
    require_exact_keys(target, _TARGET_KEYS, "manifest.target")
    _validate_observed_value(target["hardware"], "manifest.target.hardware")
    _validate_observed_value(target["operating_system"], "manifest.target.operating_system")
    _validate_model(obj["model_artifact"], "manifest.model_artifact")
    for field in (
        "quantization",
        "tokenizer",
        "chat_template",
        "generation_protocol",
        "llama_cpp",
        "context_kv",
        "benchmark",
    ):
        _validate_protocol(obj[field], f"manifest.{field}")

