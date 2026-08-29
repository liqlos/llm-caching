"""Fail-closed boundary for a future local ``pcss_llama_bridge`` executable.

The bridge is deliberately an external executable rather than a Python binding
to any particular llama.cpp package.  This keeps the verifier contract usable
with a locally built, pinned runtime while keeping this module dependency-free.
Until an explicit configuration and a strict full-state capability response are
available, this adapter reports ``NOT_RUN``.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256 as hashlib_sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .canonical import ContractError, canonical_json, require_int, require_string, sha256
from .verifier import AdapterAvailability, AdapterOutcome, TargetAdapterBinding, VerifierCase


CONFIG_ENVIRONMENT_VARIABLE = "PCSS_LLAMACPP_CONFIG"
CONFIG_SCHEMA_VERSION = "pcss.llamacpp-adapter-config/v1"
BRIDGE_PROTOCOL_VERSION = "pcss.llama-bridge/v1"
CAPABILITIES_SCHEMA_VERSION = "pcss.llama-bridge-capabilities/v1"
REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
RESPONSE_SCHEMA_VERSION = "pcss.llama-bridge-response/v1"

_CONFIG_FIELDS = {
    "schema_version",
    "bridge_protocol",
    "bridge_executable",
    "model_artifact",
    "llama_cpp_library",
    "llama_cpp_revision",
    "runtime",
    "request_timeout_seconds",
}
_RUNTIME_FIELDS = {
    "n_ctx",
    "n_batch",
    "n_gpu_layers",
    "flash_attn",
    "seed",
    "n_threads",
    "n_threads_batch",
    "chat_template",
    "thinking_mode",
}
_CAPABILITY_FIELDS = {
    "schema_version",
    "bridge_protocol",
    "operation",
    "backend",
    "full_state",
    "runtime_configuration_fingerprint",
    "probe",
    "generation_mode",
}
_CAPABILITY_BACKEND_FIELDS = {"runtime", "llama_cpp_revision", "library_sha256", "model_artifact_sha256"}
_CAPABILITY_FULL_STATE_FIELDS = {"snapshot_api", "restore_api", "comparison", "covered_components"}
_CAPABILITY_PROBE_FIELDS = {
    "model_loaded",
    "state_round_trip",
    "comparison",
    "covered_components",
    "before_digest",
    "after_digest",
}
_RESPONSE_FIELDS = {
    "schema_version",
    "bridge_protocol",
    "operation",
    "outcome",
    "state_comparison",
    "runtime_configuration_fingerprint",
    "generation_mode",
}
_OUTCOME_FIELDS = {
    "accepted_draft_tokens",
    "correction_token",
    "bonus_token",
    "committed_tokens",
    "mutable_state_digest",
}
_STATE_COMPARISON_FIELDS = {"comparison", "reference_digest", "candidate_digest", "equal"}
_FULL_STATE_COMPONENTS = ("llama_state", "logits", "token_history", "positions")
_GENERATION_MODE = "greedy_argmax"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ContractError(f"{field} has invalid keys: {', '.join(details)}")


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be an object")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    return value


def _require_sha256(value: object, field: str) -> str:
    value = require_string(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _require_tokens(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be an array of token ids")
    tokens: list[int] = []
    for index, token in enumerate(value):
        tokens.append(require_int(token, f"{field}[{index}]", minimum=0))
    return tuple(tokens)


def _strict_json_object(text: str, field: str) -> dict[str, Any]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"{field} must be one strict JSON object") from error
    return _require_mapping(value, field)


def _absolute_regular_path(value: object, field: str, *, executable: bool = False) -> Path:
    text = require_string(value, field)
    path = Path(text)
    if not path.is_absolute():
        raise ContractError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        file_mode = resolved.stat().st_mode
    except OSError as error:
        raise ContractError(f"{field} must resolve to a readable regular file") from error
    if not stat.S_ISREG(file_mode):
        raise ContractError(f"{field} must resolve to a regular file")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{field} must resolve to an executable regular file")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib_sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"unable to hash configured file: {path}") from error
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _BridgeConfig:
    config_path: Path
    bridge_executable: Path
    model_artifact: Path
    llama_cpp_library: Path
    llama_cpp_revision: str
    runtime: Mapping[str, Any]
    request_timeout_seconds: int
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedBridge:
    config: _BridgeConfig
    capabilities: dict[str, Any]
    bridge_sha256: str
    model_sha256: str
    library_sha256: str

    @property
    def build_fingerprint(self) -> str:
        return sha256(
            {
                "schema_version": "pcss.llamacpp-build-binding/v1",
                "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
                "llama_cpp_revision": self.config.llama_cpp_revision,
                "bridge_binary_sha256": self.bridge_sha256,
                "llama_cpp_library_sha256": self.library_sha256,
                "model_artifact_sha256": self.model_sha256,
                "capability_fingerprint": sha256(self.capabilities),
            }
        )


def _load_config(environment: Mapping[str, str]) -> _BridgeConfig:
    configured_path = environment.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not configured_path:
        raise ContractError(f"{CONFIG_ENVIRONMENT_VARIABLE} must name an explicit config file")
    config_path = _absolute_regular_path(configured_path, CONFIG_ENVIRONMENT_VARIABLE)
    try:
        raw = _strict_json_object(config_path.read_text(encoding="utf-8"), "llama.cpp adapter config")
    except OSError as error:
        raise ContractError(f"unable to read {CONFIG_ENVIRONMENT_VARIABLE}") from error
    _require_exact_keys(raw, _CONFIG_FIELDS, "llama.cpp adapter config")
    if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ContractError("unsupported llama.cpp adapter config schema")
    if raw["bridge_protocol"] != BRIDGE_PROTOCOL_VERSION:
        raise ContractError("unsupported llama.cpp bridge protocol")

    runtime = _require_mapping(raw["runtime"], "llama.cpp adapter config.runtime")
    _require_exact_keys(runtime, _RUNTIME_FIELDS, "llama.cpp adapter config.runtime")
    require_int(runtime["n_ctx"], "llama.cpp adapter config.runtime.n_ctx", minimum=1)
    require_int(runtime["n_batch"], "llama.cpp adapter config.runtime.n_batch", minimum=1)
    require_int(runtime["n_gpu_layers"], "llama.cpp adapter config.runtime.n_gpu_layers", minimum=0)
    if runtime["flash_attn"] not in {"auto", "enabled", "disabled"}:
        raise ContractError(
            "llama.cpp adapter config.runtime.flash_attn must be one of auto, enabled, disabled"
        )
    require_int(runtime["seed"], "llama.cpp adapter config.runtime.seed", minimum=0)
    require_int(runtime["n_threads"], "llama.cpp adapter config.runtime.n_threads", minimum=1)
    require_int(runtime["n_threads_batch"], "llama.cpp adapter config.runtime.n_threads_batch", minimum=1)
    require_string(runtime["chat_template"], "llama.cpp adapter config.runtime.chat_template")
    if runtime["thinking_mode"] != "disabled":
        raise ContractError("llama.cpp adapter config.runtime.thinking_mode must be disabled")

    frozen_runtime = MappingProxyType(dict(runtime))
    return _BridgeConfig(
        config_path=config_path,
        bridge_executable=_absolute_regular_path(raw["bridge_executable"], "bridge_executable", executable=True),
        model_artifact=_absolute_regular_path(raw["model_artifact"], "model_artifact"),
        llama_cpp_library=_absolute_regular_path(raw["llama_cpp_library"], "llama_cpp_library"),
        llama_cpp_revision=require_string(raw["llama_cpp_revision"], "llama_cpp_revision"),
        runtime=frozen_runtime,
        request_timeout_seconds=require_int(raw["request_timeout_seconds"], "request_timeout_seconds", minimum=1),
        config_fingerprint=sha256(
            {
                "schema_version": "pcss.llamacpp-runtime-config/v1",
                "bridge_protocol": raw["bridge_protocol"],
                "model_artifact": str(_absolute_regular_path(raw["model_artifact"], "model_artifact")),
                "llama_cpp_library": str(_absolute_regular_path(raw["llama_cpp_library"], "llama_cpp_library")),
                "llama_cpp_revision": raw["llama_cpp_revision"],
                "runtime": dict(frozen_runtime),
            }
        ),
    )


def _run_bridge(config: _BridgeConfig, arguments: tuple[str, ...], request: dict[str, Any] | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), *arguments),
            input=None if request is None else canonical_json(request),
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("pcss_llama_bridge process did not complete") from error
    if completed.returncode != 0:
        raise ContractError("pcss_llama_bridge returned a non-zero exit status")
    return _strict_json_object(completed.stdout, "pcss_llama_bridge response")


def _validate_capabilities(capabilities: dict[str, Any], config: _BridgeConfig, *, library_sha256: str, model_sha256: str) -> None:
    _require_exact_keys(capabilities, _CAPABILITY_FIELDS, "pcss_llama_bridge capabilities")
    if capabilities["schema_version"] != CAPABILITIES_SCHEMA_VERSION:
        raise ContractError("pcss_llama_bridge capabilities schema is unsupported")
    if capabilities["bridge_protocol"] != BRIDGE_PROTOCOL_VERSION:
        raise ContractError("pcss_llama_bridge capabilities protocol does not match config")
    if capabilities["operation"] != "verify_case":
        raise ContractError("pcss_llama_bridge lacks verify_case capability")
    if capabilities["generation_mode"] != _GENERATION_MODE:
        raise ContractError("pcss_llama_bridge capability does not use deterministic greedy argmax")
    if _require_sha256(
        capabilities["runtime_configuration_fingerprint"],
        "pcss_llama_bridge capabilities.runtime_configuration_fingerprint",
    ) != config.config_fingerprint:
        raise ContractError("pcss_llama_bridge capability is not bound to the configured runtime")

    backend = _require_mapping(capabilities["backend"], "pcss_llama_bridge capabilities.backend")
    _require_exact_keys(backend, _CAPABILITY_BACKEND_FIELDS, "pcss_llama_bridge capabilities.backend")
    if backend["runtime"] != "llama.cpp":
        raise ContractError("pcss_llama_bridge capability is not llama.cpp")
    if backend["llama_cpp_revision"] != config.llama_cpp_revision:
        raise ContractError("pcss_llama_bridge llama.cpp revision differs from config")
    if _require_sha256(backend["library_sha256"], "pcss_llama_bridge capabilities.backend.library_sha256") != library_sha256:
        raise ContractError("pcss_llama_bridge library hash differs from the configured library")
    if _require_sha256(backend["model_artifact_sha256"], "pcss_llama_bridge capabilities.backend.model_artifact_sha256") != model_sha256:
        raise ContractError("pcss_llama_bridge model hash differs from the configured model")

    full_state = _require_mapping(capabilities["full_state"], "pcss_llama_bridge capabilities.full_state")
    _require_exact_keys(full_state, _CAPABILITY_FULL_STATE_FIELDS, "pcss_llama_bridge capabilities.full_state")
    if full_state["snapshot_api"] != "llama_state_get_data":
        raise ContractError("pcss_llama_bridge does not declare full-state snapshot support")
    if full_state["restore_api"] != "llama_state_set_data":
        raise ContractError("pcss_llama_bridge does not declare full-state restore support")
    if full_state["comparison"] != "byte_exact":
        raise ContractError("pcss_llama_bridge full-state comparison is not byte exact")
    if not isinstance(full_state["covered_components"], list) or tuple(full_state["covered_components"]) != _FULL_STATE_COMPONENTS:
        raise ContractError("pcss_llama_bridge full-state capability has incomplete coverage")

    probe = _require_mapping(capabilities["probe"], "pcss_llama_bridge capabilities.probe")
    _require_exact_keys(probe, _CAPABILITY_PROBE_FIELDS, "pcss_llama_bridge capabilities.probe")
    if not _require_bool(probe["model_loaded"], "pcss_llama_bridge capabilities.probe.model_loaded"):
        raise ContractError("pcss_llama_bridge capability probe did not load the configured model")
    if not _require_bool(probe["state_round_trip"], "pcss_llama_bridge capabilities.probe.state_round_trip"):
        raise ContractError("pcss_llama_bridge capability probe did not complete a full-state round trip")
    if probe["comparison"] != "byte_exact":
        raise ContractError("pcss_llama_bridge capability probe is not byte exact")
    if not isinstance(probe["covered_components"], list) or tuple(probe["covered_components"]) != _FULL_STATE_COMPONENTS:
        raise ContractError("pcss_llama_bridge capability probe has incomplete full-state coverage")
    before_digest = _require_sha256(probe["before_digest"], "pcss_llama_bridge capabilities.probe.before_digest")
    after_digest = _require_sha256(probe["after_digest"], "pcss_llama_bridge capabilities.probe.after_digest")
    if before_digest != after_digest:
        raise ContractError("pcss_llama_bridge capability probe full-state round trip changed state bytes")


def _tokens_from_case(case: VerifierCase) -> tuple[int, ...]:
    value = getattr(case, "initial_context_tokens", ())
    if not isinstance(value, tuple):
        raise ContractError("case.initial_context_tokens must be a token tuple")
    for index, token in enumerate(value):
        require_int(token, f"case.initial_context_tokens[{index}]", minimum=0)
    return value


def _request_for_case(config: _BridgeConfig, case: VerifierCase) -> dict[str, Any]:
    provenance = getattr(case, "reference_provenance", None)
    reference_provenance: dict[str, Any] | None = None
    if provenance is not None:
        reference_provenance = {
            "schema_version": provenance.schema_version,
            "reference_method": provenance.reference_method,
            "model_artifact_sha256": provenance.model_artifact_sha256,
            "runtime_configuration_fingerprint": provenance.runtime_configuration_fingerprint,
        }
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "operation": "verify_case",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "runtime_configuration_fingerprint": config.config_fingerprint,
        "generation_mode": _GENERATION_MODE,
        "reference": {
            "generation_mode": _GENERATION_MODE,
            "comparison": "byte_exact",
            "full_state_components": list(_FULL_STATE_COMPONENTS),
            "provenance": reference_provenance,
        },
        "case": {
            "case_id": case.case_id,
            "initial_context_tokens": list(_tokens_from_case(case)),
            "draft_tokens": list(case.draft_tokens),
            "k": case.k,
            "include_bonus_on_full_accept": case.include_bonus_on_full_accept,
        },
    }


def _capability_request(config: _BridgeConfig) -> dict[str, Any]:
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "bridge_protocol": BRIDGE_PROTOCOL_VERSION,
        "operation": "capabilities",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
        "runtime_configuration_fingerprint": config.config_fingerprint,
        "generation_mode": _GENERATION_MODE,
    }


def _response_to_outcome(response: dict[str, Any], case: VerifierCase) -> AdapterOutcome:
    _require_exact_keys(response, _RESPONSE_FIELDS, "pcss_llama_bridge verify response")
    if response["schema_version"] != RESPONSE_SCHEMA_VERSION:
        raise ContractError("pcss_llama_bridge verify response schema is unsupported")
    if response["bridge_protocol"] != BRIDGE_PROTOCOL_VERSION or response["operation"] != "verify_case":
        raise ContractError("pcss_llama_bridge verify response has an unsupported operation")
    if response["generation_mode"] != _GENERATION_MODE:
        raise ContractError("pcss_llama_bridge verify response does not use deterministic greedy argmax")
    # The bridge must echo the runtime fingerprint that it was asked to use;
    # otherwise a response from a stale/default configuration is ambiguous.
    # Validation occurs in ``verify`` where the configured value is available.

    outcome = _require_mapping(response["outcome"], "pcss_llama_bridge verify response.outcome")
    _require_exact_keys(outcome, _OUTCOME_FIELDS, "pcss_llama_bridge verify response.outcome")
    accepted = require_int(outcome["accepted_draft_tokens"], "bridge outcome.accepted_draft_tokens", minimum=0)
    if accepted > case.k:
        raise ContractError("bridge outcome accepts more draft tokens than k")
    correction = outcome["correction_token"]
    if correction is not None:
        correction = require_int(correction, "bridge outcome.correction_token", minimum=0)
    bonus = outcome["bonus_token"]
    if bonus is not None:
        bonus = require_int(bonus, "bridge outcome.bonus_token", minimum=0)
    committed = _require_tokens(outcome["committed_tokens"], "bridge outcome.committed_tokens")
    mutable_state_digest = _require_sha256(outcome["mutable_state_digest"], "bridge outcome.mutable_state_digest")

    if accepted < case.k:
        if correction is None or bonus is not None:
            raise ContractError("partial bridge acceptance requires one correction and no bonus")
        expected_committed = case.draft_tokens[:accepted] + (correction,)
    else:
        if correction is not None or (bonus is not None and not case.include_bonus_on_full_accept):
            raise ContractError("full bridge acceptance has an invalid correction or bonus")
        expected_committed = case.draft_tokens + (() if bonus is None else (bonus,))
    if committed != expected_committed:
        raise ContractError("bridge committed tokens do not match its acceptance outcome")

    comparison = _require_mapping(response["state_comparison"], "pcss_llama_bridge verify response.state_comparison")
    _require_exact_keys(comparison, _STATE_COMPARISON_FIELDS, "pcss_llama_bridge verify response.state_comparison")
    if comparison["comparison"] != "byte_exact":
        raise ContractError("bridge verify response does not use byte-exact state comparison")
    reference_digest = _require_sha256(comparison["reference_digest"], "bridge state comparison.reference_digest")
    candidate_digest = _require_sha256(comparison["candidate_digest"], "bridge state comparison.candidate_digest")
    equal = _require_bool(comparison["equal"], "bridge state comparison.equal")
    if mutable_state_digest != candidate_digest:
        raise ContractError("bridge mutable-state digest differs from candidate state comparison digest")
    if equal and reference_digest != candidate_digest:
        raise ContractError("bridge claims equal full state with differing byte-exact digests")
    if not equal and reference_digest == candidate_digest:
        raise ContractError("bridge claims unequal full state with identical byte-exact digests")

    return AdapterOutcome(
        accepted_draft_tokens=accepted,
        correction_token=correction,
        bonus_token=bonus,
        committed_tokens=committed,
        mutable_state_digest=mutable_state_digest,
    )


class LlamaCppBridgeAdapter:
    """Future target adapter that refuses a backend claim without an explicit bridge proof."""

    name = "pcss_llama_bridge"

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        self._environment = dict(os.environ if environment is None else environment)
        self._availability: AdapterAvailability | None = None
        self._prepared: _PreparedBridge | None = None

    def _prepare(self) -> _PreparedBridge:
        config = _load_config(self._environment)
        library_sha256 = _file_sha256(config.llama_cpp_library)
        model_sha256 = _file_sha256(config.model_artifact)
        bridge_sha256 = _file_sha256(config.bridge_executable)
        capabilities = _run_bridge(config, ("--pcss-capabilities",), _capability_request(config))
        _validate_capabilities(capabilities, config, library_sha256=library_sha256, model_sha256=model_sha256)
        return _PreparedBridge(
            config=config,
            capabilities=capabilities,
            bridge_sha256=bridge_sha256,
            model_sha256=model_sha256,
            library_sha256=library_sha256,
        )

    def availability(self) -> AdapterAvailability:
        if self._availability is not None:
            return self._availability
        try:
            self._prepared = self._prepare()
        except ContractError as error:
            self._availability = AdapterAvailability("NOT_RUN", str(error))
        else:
            self._availability = AdapterAvailability("TARGET_BACKEND", "pcss_llama_bridge full-state capability verified")
        return self._availability

    def _frozen_prepared(self) -> _PreparedBridge:
        if self._availability is None:
            raise ContractError("pcss_llama_bridge availability() must run before binding or verification")
        if self._availability.execution_class != "TARGET_BACKEND" or self._prepared is None:
            raise ContractError(f"pcss_llama_bridge is not available: {self._availability.reason}")
        return self._prepared

    def target_binding(self) -> TargetAdapterBinding:
        prepared = self._frozen_prepared()
        values: dict[str, Any] = {
            "schema_version": "pcss.verifier-adapter-binding/v1",
            "adapter_source_path": str(Path(__file__).resolve()),
            "backend_binary_path": str(prepared.config.bridge_executable),
            "model_artifact_path": str(prepared.config.model_artifact),
            "llama_cpp_build_fingerprint": prepared.build_fingerprint,
        }
        # The verifier contract is being extended to bind the runtime config separately.
        # Use it when present while retaining compatibility with the current v1 contract.
        if "runtime_configuration_fingerprint" in getattr(TargetAdapterBinding, "__dataclass_fields__", {}):
            values["schema_version"] = "pcss.verifier-adapter-binding/v2"
            values["runtime_configuration_fingerprint"] = prepared.config.config_fingerprint
        return TargetAdapterBinding(**values)

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        prepared = self._frozen_prepared()
        response = _run_bridge(
            prepared.config,
            ("--pcss-verify-case",),
            _request_for_case(prepared.config, case),
        )
        if _require_sha256(
            response.get("runtime_configuration_fingerprint"),
            "pcss_llama_bridge verify response.runtime_configuration_fingerprint",
        ) != prepared.config.config_fingerprint:
            raise ContractError("pcss_llama_bridge verify response is not bound to the configured runtime")
        return _response_to_outcome(response, case)


def create_adapter() -> LlamaCppBridgeAdapter:
    """Factory for ``pcss.verifier_runner --adapter pcss.llamacpp_adapter:create_adapter``."""

    return LlamaCppBridgeAdapter()
