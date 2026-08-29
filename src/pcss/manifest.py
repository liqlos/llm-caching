"""Baseline evidence manifests; values unknown on a target stay explicit."""

from __future__ import annotations

from typing import Any

from .canonical import ContractError, require_exact_keys, require_int, require_mapping, require_string, sha256

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
    "verifier_gate",
}
_TARGET_KEYS = {"hardware", "operating_system"}
_OBSERVED_VALUE_KEYS = {"observation_status", "value"}
_MODEL_KEYS = {"observation_status", "artifact_uri", "sha256"}
_PROTOCOL_KEYS = {"observation_status", "fingerprint", "parameters"}
_VERIFIER_GATE_KEYS = {"observation_status", "host_identity", "adapter_binding"}
_HOST_IDENTITY_KEYS = {
    "schema_version",
    "target_profile",
    "operating_system",
    "architecture",
    "hardware_model",
    "chip",
    "memory_bytes",
    "fingerprint",
}
_ADAPTER_BINDING_KEYS = {
    "schema_version",
    "adapter_source_sha256",
    "backend_binary_sha256",
    "model_artifact_sha256",
    "llama_cpp_build_fingerprint",
    "runtime_configuration_fingerprint",
}


def _require_sha256(value: Any, field: str) -> str:
    digest = require_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return digest


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


def _validate_host_identity(value: Any) -> None:
    identity = require_mapping(value, "manifest.verifier_gate.host_identity")
    require_exact_keys(identity, _HOST_IDENTITY_KEYS, "manifest.verifier_gate.host_identity")
    if identity["schema_version"] != "pcss.host-identity/v1":
        raise ContractError("unsupported manifest.verifier_gate.host_identity.schema_version")
    for key in ("target_profile", "operating_system", "architecture", "hardware_model", "chip"):
        require_string(identity[key], f"manifest.verifier_gate.host_identity.{key}")
    require_int(identity["memory_bytes"], "manifest.verifier_gate.host_identity.memory_bytes", minimum=1)
    fingerprint = _require_sha256(identity["fingerprint"], "manifest.verifier_gate.host_identity.fingerprint")
    identity_without_fingerprint = {key: identity[key] for key in _HOST_IDENTITY_KEYS - {"fingerprint"}}
    if fingerprint != sha256(identity_without_fingerprint):
        raise ContractError("manifest.verifier_gate.host_identity.fingerprint does not match host identity")


def _validate_adapter_binding(value: Any) -> None:
    binding = require_mapping(value, "manifest.verifier_gate.adapter_binding")
    require_exact_keys(binding, _ADAPTER_BINDING_KEYS, "manifest.verifier_gate.adapter_binding")
    if binding["schema_version"] != "pcss.verifier-adapter-binding/v2":
        raise ContractError("unsupported manifest.verifier_gate.adapter_binding.schema_version")
    for key in _ADAPTER_BINDING_KEYS - {"schema_version"}:
        _require_sha256(binding[key], f"manifest.verifier_gate.adapter_binding.{key}")


def _validate_verifier_gate(value: Any) -> None:
    gate = require_mapping(value, "manifest.verifier_gate")
    require_exact_keys(gate, _VERIFIER_GATE_KEYS, "manifest.verifier_gate")
    status = require_string(gate["observation_status"], "manifest.verifier_gate.observation_status")
    if status not in OBSERVATION_STATUSES:
        raise ContractError("manifest.verifier_gate.observation_status is invalid")
    if status == "not_observed":
        if gate["host_identity"] is not None or gate["adapter_binding"] is not None:
            raise ContractError("manifest.verifier_gate details must be null until observed")
        return
    if gate["host_identity"] is None or gate["adapter_binding"] is None:
        raise ContractError("observed manifest.verifier_gate requires host_identity and adapter_binding")
    _validate_host_identity(gate["host_identity"])
    _validate_adapter_binding(gate["adapter_binding"])


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
    _validate_verifier_gate(obj["verifier_gate"])
