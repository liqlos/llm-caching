"""Fail-closed Python boundary for the local ``pcss_llama_bridge`` executable.

The native bridge owns llama.cpp model loading and model-memory state handling.
This module owns explicit configuration, independently checks the bridge's
observed library identity, and rejects every incomplete or ambiguous response.
It is a verifier adapter only; it makes no performance or deployment claim.
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
REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
CAPABILITIES_SCHEMA_VERSION = "pcss.llama-bridge-capabilities/v2"
VERIFY_SCHEMA_VERSION = "pcss.llama-bridge-verify/v2"
PENDING_VERIFY_SCHEMA_VERSION = "pcss.llama-bridge-pending-verify/v1"

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
_COMMON_REQUEST_FIELDS = {
    "schema_version",
    "operation",
    "model_artifact",
    "llama_cpp_library",
    "llama_cpp_revision",
    "runtime",
}
_VERIFY_REQUEST_FIELDS = _COMMON_REQUEST_FIELDS | {"seed", "case"}
_VERIFY_SEED_FIELDS = {"seed_id", "initial_context_tokens"}
_VERIFY_CASE_FIELDS = {"case_id", "draft_tokens", "include_bonus_on_full_accept"}
_CONFIGURATION_FIELDS = {"model_artifact", "llama_cpp_library", "llama_cpp_revision", "runtime"}
_OBSERVED_CONTEXT_FIELDS = {
    "actual_n_ctx",
    "actual_n_batch",
    "actual_n_threads",
    "actual_n_threads_batch",
}
_CAPABILITIES_RESPONSE_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "state_export",
    "generation_mode",
    "greedy_tie_break",
    "raw_token_ids",
    "llama_c_api_version",
    "supports_gpu_offload",
    "loaded_llama_cpp_library_path",
    "supports_mmap",
    "supports_mlock",
    "vocabulary_size",
    "configuration",
    "observed_context",
    "state_components",
    "state_api",
    "sampler_rng_state",
    "probe_token",
    "probe_state_sha256",
    "probe_state_bytes",
    "restored_state_sha256",
    "restored_state_bytes",
    "state_round_trip_byte_exact",
    "operations",
    "declared_identity_fields",
    "chat_template_applied",
    "thinking_mode_applied",
}
_VERIFY_RESPONSE_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "generation_mode",
    "greedy_tie_break",
    "state_export",
    "reconstruction",
    "seed_id",
    "case_id",
    "target_tokens",
    "verification_target_tokens",
    "verification_targets_match",
    "vocabulary_size",
    "accepted_draft_tokens",
    "reference_accepted_draft_tokens",
    "correction_token",
    "bonus_token",
    "committed_tokens",
    "reference_committed_tokens",
    "prompt_state_sha256",
    "prompt_state_bytes",
    "reference_state_sha256",
    "reference_state_bytes",
    "committed_state_sha256",
    "committed_state_bytes",
    "verifier_draft_state_sha256",
    "verifier_draft_state_bytes",
    "reference_mutable_state_digest",
    "committed_mutable_state_digest",
    "mutable_state_digest",
    "state_matches",
    "reference_next_token",
    "committed_next_token",
    "continuation_matches",
    "configuration",
    "observed_context",
    "tokenization_boundary",
    "state_api",
    "sampler_rng_state",
    "declared_identity_fields",
    "chat_template_applied",
    "thinking_mode_applied",
}
_PENDING_VERIFY_RESPONSE_FIELDS = _VERIFY_RESPONSE_FIELDS | {
    "pending_state_sha256",
    "pending_state_bytes",
    "reference_pending_state_sha256",
    "reference_pending_state_bytes",
    "pending_state_matches",
    "tail_trimmed",
    "pending_token",
}
_FULL_STATE_COMPONENTS = (
    "model_architecture_identity",
    "llama_memory",
)
_STATE_EXPORT = "model-architecture+memory-state/v2"
_STATE_API = "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2"
_SAMPLER_RNG_STATE = "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1"
_DECLARED_IDENTITY_FIELDS = "caller-declared-not-verified-by-bridge/v1"
_GENERATION_MODE = "greedy_argmax"
_GREEDY_TIE_BREAK = "lowest-token-id/v1"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
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


def _require_tokens(value: object, field: str, *, nonempty: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = " non-empty" if nonempty else ""
        raise ContractError(f"{field} must be a{qualifier} token-id array")
    return tuple(require_int(token, f"{field}[{index}]", minimum=0) for index, token in enumerate(value))


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
        payload = json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"{field} must be one strict JSON object") from error
    return _require_mapping(payload, field)


def _absolute_regular_path(value: object, field: str, *, executable: bool = False) -> Path:
    text = require_string(value, field)
    path = Path(text)
    if not path.is_absolute():
        raise ContractError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise ContractError(f"{field} must resolve to a readable regular file") from error
    if not stat.S_ISREG(mode):
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
    runtime_configuration_fingerprint: str


@dataclass(frozen=True, slots=True)
class _PreparedBridge:
    config: _BridgeConfig
    capabilities: dict[str, Any]
    bridge_sha256: str
    model_sha256: str
    loaded_library_path: Path
    loaded_library_sha256: str

    @property
    def build_fingerprint(self) -> str:
        return sha256(
            {
                "schema_version": "pcss.llamacpp-build-binding/v2",
                "bridge_binary_sha256": self.bridge_sha256,
                "model_artifact_sha256": self.model_sha256,
                "configured_llama_cpp_library": str(self.config.llama_cpp_library),
                "loaded_llama_cpp_library": str(self.loaded_library_path),
                "loaded_llama_cpp_library_sha256": self.loaded_library_sha256,
                "llama_cpp_revision": self.config.llama_cpp_revision,
                "capability_response_fingerprint": sha256(self.capabilities),
            }
        )


def _load_config(environment: Mapping[str, str]) -> _BridgeConfig:
    configured_path = environment.get(CONFIG_ENVIRONMENT_VARIABLE)
    if not configured_path:
        raise ContractError(f"{CONFIG_ENVIRONMENT_VARIABLE} must name an explicit config file")
    config_path = _absolute_regular_path(configured_path, CONFIG_ENVIRONMENT_VARIABLE)
    try:
        payload = _strict_json_object(config_path.read_text(encoding="utf-8"), "llama.cpp adapter config")
    except OSError as error:
        raise ContractError(f"unable to read {CONFIG_ENVIRONMENT_VARIABLE}") from error
    _require_exact_keys(payload, _CONFIG_FIELDS, "llama.cpp adapter config")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ContractError("unsupported llama.cpp adapter config schema")
    if payload["bridge_protocol"] != BRIDGE_PROTOCOL_VERSION:
        raise ContractError("unsupported llama.cpp bridge protocol")

    runtime = _require_mapping(payload["runtime"], "llama.cpp adapter config.runtime")
    _require_exact_keys(runtime, _RUNTIME_FIELDS, "llama.cpp adapter config.runtime")
    normalized_runtime = dict(runtime)
    n_ctx = require_int(normalized_runtime["n_ctx"], "llama.cpp adapter config.runtime.n_ctx", minimum=1)
    n_batch = require_int(normalized_runtime["n_batch"], "llama.cpp adapter config.runtime.n_batch", minimum=1)
    if n_batch > n_ctx:
        raise ContractError("llama.cpp adapter config.runtime.n_batch must not exceed n_ctx")
    require_int(normalized_runtime["n_gpu_layers"], "llama.cpp adapter config.runtime.n_gpu_layers", minimum=0)
    flash_attn = require_string(normalized_runtime["flash_attn"], "llama.cpp adapter config.runtime.flash_attn")
    if flash_attn not in {"auto", "enabled", "disabled"}:
        raise ContractError("llama.cpp adapter config.runtime.flash_attn must be auto, enabled, or disabled")
    require_int(normalized_runtime["seed"], "llama.cpp adapter config.runtime.seed", minimum=0)
    require_int(normalized_runtime["n_threads"], "llama.cpp adapter config.runtime.n_threads", minimum=1)
    require_int(normalized_runtime["n_threads_batch"], "llama.cpp adapter config.runtime.n_threads_batch", minimum=1)
    require_string(normalized_runtime["chat_template"], "llama.cpp adapter config.runtime.chat_template")
    if normalized_runtime["thinking_mode"] != "disabled":
        raise ContractError("llama.cpp adapter config.runtime.thinking_mode must be disabled")

    bridge_executable = _absolute_regular_path(payload["bridge_executable"], "bridge_executable", executable=True)
    model_artifact = _absolute_regular_path(payload["model_artifact"], "model_artifact")
    llama_cpp_library = _absolute_regular_path(payload["llama_cpp_library"], "llama_cpp_library")
    llama_cpp_revision = require_string(payload["llama_cpp_revision"], "llama_cpp_revision")
    return _BridgeConfig(
        config_path=config_path,
        bridge_executable=bridge_executable,
        model_artifact=model_artifact,
        llama_cpp_library=llama_cpp_library,
        llama_cpp_revision=llama_cpp_revision,
        runtime=MappingProxyType(normalized_runtime),
        request_timeout_seconds=require_int(payload["request_timeout_seconds"], "request_timeout_seconds", minimum=1),
        runtime_configuration_fingerprint=sha256(
            {
                "schema_version": "pcss.llamacpp-runtime-config/v1",
                "bridge_protocol": payload["bridge_protocol"],
                "model_artifact": str(model_artifact),
                "llama_cpp_library": str(llama_cpp_library),
                "llama_cpp_revision": llama_cpp_revision,
                "runtime": normalized_runtime,
            }
        ),
    )


def _configuration(config: _BridgeConfig) -> dict[str, Any]:
    return {
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
    }


def _common_request(config: _BridgeConfig, operation: str) -> dict[str, Any]:
    request = {"schema_version": REQUEST_SCHEMA_VERSION, "operation": operation, **_configuration(config)}
    _require_exact_keys(request, _COMMON_REQUEST_FIELDS, "pcss_llama_bridge common request")
    return request


def _capability_request(config: _BridgeConfig) -> dict[str, Any]:
    return _common_request(config, "capabilities")


def _case_seed(case: VerifierCase) -> tuple[str, tuple[int, ...]]:
    seed_id = getattr(case, "seed_id", None)
    if not isinstance(seed_id, str) or not seed_id:
        raise ContractError("TARGET_BACKEND verifier cases require a non-empty case.seed_id")
    initial_context_tokens = getattr(case, "initial_context_tokens", None)
    if not isinstance(initial_context_tokens, tuple) or not initial_context_tokens:
        raise ContractError("TARGET_BACKEND verifier cases require non-empty case.initial_context_tokens")
    for index, token in enumerate(initial_context_tokens):
        require_int(token, f"case.initial_context_tokens[{index}]", minimum=0)
    return seed_id, initial_context_tokens


def _verify_request(
    config: _BridgeConfig,
    case: VerifierCase,
    *,
    operation: str = "verify",
) -> tuple[dict[str, Any], str]:
    if operation not in {"verify", "verify_pending"}:
        raise ContractError("unsupported llama.cpp verifier operation")
    seed_id, initial_context_tokens = _case_seed(case)
    if not isinstance(case.include_bonus_on_full_accept, bool):
        raise ContractError("case.include_bonus_on_full_accept must be a boolean")
    request = {
        **_common_request(config, operation),
        "seed": {"seed_id": seed_id, "initial_context_tokens": list(initial_context_tokens)},
        "case": {
            "case_id": require_string(case.case_id, "case.case_id"),
            "draft_tokens": list(case.draft_tokens),
            "include_bonus_on_full_accept": case.include_bonus_on_full_accept,
        },
    }
    _require_exact_keys(request, _VERIFY_REQUEST_FIELDS, "pcss_llama_bridge verify request")
    _require_exact_keys(request["seed"], _VERIFY_SEED_FIELDS, "pcss_llama_bridge verify request.seed")
    _require_exact_keys(request["case"], _VERIFY_CASE_FIELDS, "pcss_llama_bridge verify request.case")
    return request, seed_id


def _run_bridge(config: _BridgeConfig, arguments: tuple[str, ...], request: dict[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), *arguments),
            input=canonical_json(request) + "\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
        raise ContractError("pcss_llama_bridge process did not complete") from error
    if completed.returncode != 0:
        response = _strict_json_object(completed.stdout, "pcss_llama_bridge response")
        if (
            completed.returncode == 1
            and request["operation"] in {"verify", "verify_pending"}
            and response.get("schema_version") in {VERIFY_SCHEMA_VERSION, PENDING_VERIFY_SCHEMA_VERSION}
            and response.get("operation") == request["operation"]
            and response.get("status") == "mismatch"
        ):
            return response
        raise ContractError("pcss_llama_bridge process returned a non-zero exit status")
    return _strict_json_object(completed.stdout, "pcss_llama_bridge response")


def _validate_configuration_echo(value: object, config: _BridgeConfig, field: str) -> None:
    configuration = _require_mapping(value, field)
    _require_exact_keys(configuration, _CONFIGURATION_FIELDS, field)
    if configuration != _configuration(config):
        raise ContractError(f"{field} differs from explicit adapter configuration")


def _validate_observed_context(value: object, config: _BridgeConfig, field: str) -> None:
    observed = _require_mapping(value, field)
    _require_exact_keys(observed, _OBSERVED_CONTEXT_FIELDS, field)
    expected = {
        "actual_n_ctx": config.runtime["n_ctx"],
        "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }
    for name, expected_value in expected.items():
        actual = require_int(observed[name], f"{field}.{name}", minimum=1)
        if actual != expected_value:
            raise ContractError(f"{field} differs from explicit adapter configuration")


def _validate_shared_response_facts(response: Mapping[str, Any], config: _BridgeConfig, field: str) -> None:
    if response["backend"] != "llama.cpp":
        raise ContractError(f"{field} is not a llama.cpp result")
    if response["generation_mode"] != _GENERATION_MODE or response["greedy_tie_break"] != _GREEDY_TIE_BREAK:
        raise ContractError(f"{field} is not deterministic greedy argmax")
    if response["state_export"] != _STATE_EXPORT:
        raise ContractError(f"{field} does not declare the bound model-memory state export")
    if response["state_api"] != _STATE_API:
        raise ContractError(f"{field} does not declare full llama state get/set")
    if response["sampler_rng_state"] != _SAMPLER_RNG_STATE:
        raise ContractError(f"{field} unexpectedly uses sampler or RNG state")
    if response["declared_identity_fields"] != _DECLARED_IDENTITY_FIELDS:
        raise ContractError(f"{field} has an unsupported identity declaration")
    if _require_bool(response["chat_template_applied"], f"{field}.chat_template_applied"):
        raise ContractError(f"{field} unexpectedly applied a chat template")
    if _require_bool(response["thinking_mode_applied"], f"{field}.thinking_mode_applied"):
        raise ContractError(f"{field} unexpectedly applied a thinking-mode policy")
    _validate_configuration_echo(response["configuration"], config, f"{field}.configuration")
    _validate_observed_context(response["observed_context"], config, f"{field}.observed_context")


def _validate_capabilities(response: dict[str, Any], config: _BridgeConfig) -> tuple[Path, str]:
    field = "pcss_llama_bridge capabilities response"
    _require_exact_keys(response, _CAPABILITIES_RESPONSE_FIELDS, field)
    if response["schema_version"] != CAPABILITIES_SCHEMA_VERSION or response["operation"] != "capabilities":
        raise ContractError(f"{field} has an unsupported schema or operation")
    if response["status"] != "ok":
        raise ContractError(f"{field} is not successful")
    _validate_shared_response_facts(response, config, field)
    if not _require_bool(response["raw_token_ids"], f"{field}.raw_token_ids"):
        raise ContractError(f"{field} did not preserve caller-supplied token ids")
    if response["llama_c_api_version"] != "not-exposed-by-stable-c-api/v1":
        raise ContractError(f"{field} has an unsupported C API identity")
    for name in ("supports_gpu_offload", "supports_mmap", "supports_mlock"):
        _require_bool(response[name], f"{field}.{name}")
    vocabulary_size = require_int(response["vocabulary_size"], f"{field}.vocabulary_size", minimum=1)
    probe_token = require_int(response["probe_token"], f"{field}.probe_token", minimum=0)
    if probe_token >= vocabulary_size:
        raise ContractError(f"{field} probe token is outside the reported vocabulary")
    state_components = response["state_components"]
    if not isinstance(state_components, list) or tuple(state_components) != _FULL_STATE_COMPONENTS:
        raise ContractError(f"{field} has incomplete model-memory state coverage")
    probe_digest = _require_sha256(response["probe_state_sha256"], f"{field}.probe_state_sha256")
    restored_digest = _require_sha256(response["restored_state_sha256"], f"{field}.restored_state_sha256")
    probe_bytes = require_int(response["probe_state_bytes"], f"{field}.probe_state_bytes", minimum=1)
    restored_bytes = require_int(response["restored_state_bytes"], f"{field}.restored_state_bytes", minimum=1)
    if not _require_bool(response["state_round_trip_byte_exact"], f"{field}.state_round_trip_byte_exact"):
        raise ContractError(f"{field} did not complete a byte-exact model-memory state round trip")
    if probe_digest != restored_digest or probe_bytes != restored_bytes:
        raise ContractError(f"{field} model-memory state round trip changed bytes")
    operations = response["operations"]
    supported_operation_sets = {
        ("capabilities", "reference", "verify", "profile"),
        ("capabilities", "reference", "verify", "verify_pending", "profile"),
        ("capabilities", "reference", "verify", "verify_pending", "profile", "profile_pending"),
        (
            "capabilities",
            "reference",
            "reference_pending",
            "verify",
            "verify_pending",
            "trajectory_pending",
            "benchmark_pending",
            "live_pending",
            "single_context_live_pending",
            "profile",
            "profile_pending",
        ),
    }
    if not isinstance(operations, list) or tuple(operations) not in supported_operation_sets:
        raise ContractError(f"{field} has an incomplete operation set")

    loaded_library = _absolute_regular_path(
        response["loaded_llama_cpp_library_path"], f"{field}.loaded_llama_cpp_library_path"
    )
    if loaded_library != config.llama_cpp_library:
        raise ContractError(f"{field} loaded library path differs from configured library")
    loaded_library_sha256 = _file_sha256(loaded_library)
    if loaded_library_sha256 != _file_sha256(config.llama_cpp_library):
        raise ContractError(f"{field} loaded library hash differs from configured library")
    return loaded_library, loaded_library_sha256


def _response_token(value: object, field: str, vocabulary_size: int) -> int:
    token = require_int(value, field, minimum=0)
    if token >= vocabulary_size:
        raise ContractError(f"{field} is outside the reported vocabulary")
    return token


def _validate_verify_response(
    response: dict[str, Any],
    config: _BridgeConfig,
    case: VerifierCase,
    seed_id: str,
    *,
    commit_protocol: str = "serial",
) -> AdapterOutcome:
    field = "pcss_llama_bridge verify response"
    if commit_protocol == "serial":
        expected_fields = _VERIFY_RESPONSE_FIELDS
        expected_schema = VERIFY_SCHEMA_VERSION
        expected_operation = "verify"
        expected_reconstruction = "batch-score+serial-canonical-commit/v2"
    elif commit_protocol == "pending_native":
        expected_fields = _PENDING_VERIFY_RESPONSE_FIELDS
        expected_schema = PENDING_VERIFY_SCHEMA_VERSION
        expected_operation = "verify_pending"
        expected_reconstruction = {
            "pending-anchor+batch-score+native-tail-trim/v1",
            "pending-anchor+batch-score+serial-state-fallback/v2",
        }
    else:
        raise ContractError("unsupported llama.cpp verifier commit protocol")
    _require_exact_keys(response, expected_fields, field)
    if response["schema_version"] != expected_schema or response["operation"] != expected_operation:
        raise ContractError(f"{field} has an unsupported schema or operation")
    if response["status"] not in {"ok", "mismatch"}:
        raise ContractError(f"{field} has an unsupported status")
    _validate_shared_response_facts(response, config, field)
    if (
        response["reconstruction"] not in expected_reconstruction
        if isinstance(expected_reconstruction, set)
        else response["reconstruction"] != expected_reconstruction
    ):
        raise ContractError(f"{field} has an unsupported reconstruction method")
    if response["tokenization_boundary"] != "caller-supplied-raw-token-ids/v1":
        raise ContractError(f"{field} has an unexpected tokenization boundary")
    if response["seed_id"] != seed_id or response["case_id"] != case.case_id:
        raise ContractError(f"{field} seed or case identity differs from request")

    vocabulary_size = require_int(response["vocabulary_size"], f"{field}.vocabulary_size", minimum=1)
    target_tokens = _require_tokens(response["target_tokens"], f"{field}.target_tokens", nonempty=True)
    verification_target_tokens = _require_tokens(
        response["verification_target_tokens"], f"{field}.verification_target_tokens", nonempty=True
    )
    if len(target_tokens) != case.k + 1:
        raise ContractError(f"{field} target token count differs from draft length plus one")
    if len(verification_target_tokens) != len(target_tokens):
        raise ContractError(f"{field} verification target count differs from serial reference")
    if any(token >= vocabulary_size for token in target_tokens + verification_target_tokens):
        raise ContractError(f"{field} target tokens are outside the reported vocabulary")
    fixture_target = tuple(case.target_tokens)
    if len(fixture_target) < case.k or len(fixture_target) > len(target_tokens):
        raise ContractError("fixture target token count is incompatible with native verification")
    if target_tokens[: len(fixture_target)] != fixture_target:
        raise ContractError("native serial greedy target prefix differs from fixture target tokens")

    accepted = require_int(response["accepted_draft_tokens"], f"{field}.accepted_draft_tokens", minimum=0)
    reference_accepted = require_int(
        response["reference_accepted_draft_tokens"], f"{field}.reference_accepted_draft_tokens", minimum=0
    )
    if accepted > case.k or reference_accepted > case.k:
        raise ContractError(f"{field} accepts more draft tokens than k")
    expected_accepted = 0
    for draft_token, target_token in zip(case.draft_tokens, verification_target_tokens):
        if draft_token != target_token:
            break
        expected_accepted += 1
    if accepted != expected_accepted:
        raise ContractError(f"{field} accepted-draft count disagrees with its verification targets")
    expected_reference_accepted = 0
    for draft_token, target_token in zip(case.draft_tokens, target_tokens):
        if draft_token != target_token:
            break
        expected_reference_accepted += 1
    if reference_accepted != expected_reference_accepted:
        raise ContractError(f"{field} reference accepted-draft count disagrees with serial targets")
    authoritative_target_count = accepted + 1 if accepted < case.k else len(target_tokens)
    verification_matches = _require_bool(
        response["verification_targets_match"], f"{field}.verification_targets_match"
    )
    if verification_matches != (
        verification_target_tokens[:authoritative_target_count] == target_tokens[:authoritative_target_count]
    ):
        raise ContractError(f"{field} verification-target match claim is inconsistent")
    correction = response["correction_token"]
    bonus = response["bonus_token"]
    if accepted < case.k:
        expected_correction = verification_target_tokens[accepted]
        if correction != expected_correction or bonus is not None:
            raise ContractError(f"{field} correction or bonus disagrees with target tokens")
        committed_expected = case.draft_tokens[:accepted] + (expected_correction,)
    else:
        if correction is not None:
            raise ContractError(f"{field} full acceptance has a correction token")
        expected_bonus = verification_target_tokens[case.k] if case.include_bonus_on_full_accept else None
        if bonus != expected_bonus:
            raise ContractError(f"{field} bonus token disagrees with target tokens")
        committed_expected = case.draft_tokens + (() if expected_bonus is None else (expected_bonus,))
    if correction is not None:
        _response_token(correction, f"{field}.correction_token", vocabulary_size)
    if bonus is not None:
        _response_token(bonus, f"{field}.bonus_token", vocabulary_size)
    committed = _require_tokens(response["committed_tokens"], f"{field}.committed_tokens", nonempty=True)
    reference_committed = _require_tokens(
        response["reference_committed_tokens"], f"{field}.reference_committed_tokens", nonempty=True
    )
    if committed != committed_expected:
        raise ContractError(f"{field} committed tokens disagree with its acceptance outcome")
    if reference_committed != case.expected_committed_tokens:
        raise ContractError("native serial reference commit differs from the fixture expectation")

    _require_sha256(response["prompt_state_sha256"], f"{field}.prompt_state_sha256")
    require_int(response["prompt_state_bytes"], f"{field}.prompt_state_bytes", minimum=1)
    reference_digest = _require_sha256(response["reference_state_sha256"], f"{field}.reference_state_sha256")
    committed_digest = _require_sha256(response["committed_state_sha256"], f"{field}.committed_state_sha256")
    reference_bytes = require_int(response["reference_state_bytes"], f"{field}.reference_state_bytes", minimum=1)
    committed_bytes = require_int(response["committed_state_bytes"], f"{field}.committed_state_bytes", minimum=1)
    _require_sha256(response["verifier_draft_state_sha256"], f"{field}.verifier_draft_state_sha256")
    require_int(response["verifier_draft_state_bytes"], f"{field}.verifier_draft_state_bytes", minimum=1)
    reference_mutable = _require_sha256(
        response["reference_mutable_state_digest"], f"{field}.reference_mutable_state_digest"
    )
    committed_mutable = _require_sha256(
        response["committed_mutable_state_digest"], f"{field}.committed_mutable_state_digest"
    )
    mutable_state_digest = _require_sha256(response["mutable_state_digest"], f"{field}.mutable_state_digest")
    reconstruction_matches = _require_bool(response["state_matches"], f"{field}.state_matches")
    continuation_matches = _require_bool(response["continuation_matches"], f"{field}.continuation_matches")
    if response["status"] == "ok" and not (
        verification_matches and reconstruction_matches and continuation_matches
    ):
        raise ContractError(f"{field} status=ok contradicts its reconstruction result")
    if response["status"] == "mismatch" and verification_matches and reconstruction_matches and continuation_matches:
        raise ContractError(f"{field} status=mismatch contradicts its reconstruction result")
    if reconstruction_matches and (reference_digest != committed_digest or reference_bytes != committed_bytes):
        raise ContractError(f"{field} matching reference and committed state digests differ")
    if (
        reference_mutable != reference_digest
        or committed_mutable != committed_digest
        or mutable_state_digest != committed_digest
    ):
        raise ContractError(f"{field} mutable-state digests disagree with exported state digests")
    if reference_mutable != case.expected_state_digest:
        raise ContractError("native serial reference state differs from the fixture expectation")
    if commit_protocol == "pending_native":
        pending_digest = _require_sha256(response["pending_state_sha256"], f"{field}.pending_state_sha256")
        reference_pending_digest = _require_sha256(
            response["reference_pending_state_sha256"], f"{field}.reference_pending_state_sha256"
        )
        pending_bytes = require_int(response["pending_state_bytes"], f"{field}.pending_state_bytes", minimum=1)
        reference_pending_bytes = require_int(
            response["reference_pending_state_bytes"], f"{field}.reference_pending_state_bytes", minimum=1
        )
        pending_matches = _require_bool(response["pending_state_matches"], f"{field}.pending_state_matches")
        tail_trimmed = _require_bool(response["tail_trimmed"], f"{field}.tail_trimmed")
        if response["reconstruction"] == "pending-anchor+batch-score+native-tail-trim/v1" and not tail_trimmed:
            raise ContractError(f"{field} did not trim the rejected verification tail")
        if response["reconstruction"] == "pending-anchor+batch-score+serial-state-fallback/v2" and tail_trimmed:
            raise ContractError(f"{field} contradicts its serial-state fallback")
        if pending_matches != (
            pending_digest == reference_pending_digest and pending_bytes == reference_pending_bytes
        ):
            raise ContractError(f"{field} pending-state match claim is inconsistent")
        expected_pending = correction if correction is not None else bonus
        if response["pending_token"] != expected_pending:
            raise ContractError(f"{field} pending token differs from its correction or bonus")
        if response["status"] == "ok" and not pending_matches:
            raise ContractError(f"{field} status=ok contradicts its pending-state result")
    reference_next = _response_token(
        response["reference_next_token"], f"{field}.reference_next_token", vocabulary_size
    )
    committed_next = _response_token(
        response["committed_next_token"], f"{field}.committed_next_token", vocabulary_size
    )
    if continuation_matches and reference_next != committed_next:
        raise ContractError(f"{field} matching continuation tokens differ")

    return AdapterOutcome(
        accepted_draft_tokens=accepted,
        correction_token=correction,
        bonus_token=bonus,
        committed_tokens=committed,
        mutable_state_digest=mutable_state_digest,
        verification_matches=verification_matches,
        reconstruction_matches=reconstruction_matches,
        continuation_matches=continuation_matches,
    )


class LlamaCppBridgeAdapter:
    """Target adapter that freezes one successful native capability probe per instance."""

    name = "pcss_llama_bridge"

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        commit_protocol: str = "serial",
    ) -> None:
        if commit_protocol not in {"serial", "pending_native"}:
            raise ContractError("unsupported llama.cpp verifier commit protocol")
        self._environment = dict(os.environ if environment is None else environment)
        self._commit_protocol = commit_protocol
        self._availability: AdapterAvailability | None = None
        self._prepared: _PreparedBridge | None = None

    def _prepare(self) -> _PreparedBridge:
        config = _load_config(self._environment)
        capabilities = _run_bridge(config, ("--pcss-capabilities",), _capability_request(config))
        loaded_library_path, loaded_library_sha256 = _validate_capabilities(capabilities, config)
        return _PreparedBridge(
            config=config,
            capabilities=capabilities,
            bridge_sha256=_file_sha256(config.bridge_executable),
            model_sha256=_file_sha256(config.model_artifact),
            loaded_library_path=loaded_library_path,
            loaded_library_sha256=loaded_library_sha256,
        )

    def availability(self) -> AdapterAvailability:
        if self._availability is not None:
            return self._availability
        try:
            self._prepared = self._prepare()
        except ContractError as error:
            self._availability = AdapterAvailability("NOT_RUN", str(error))
        else:
            self._availability = AdapterAvailability(
                "TARGET_BACKEND", "native llama.cpp model-memory state capability probe verified"
            )
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
        if "runtime_configuration_fingerprint" in getattr(TargetAdapterBinding, "__dataclass_fields__", {}):
            values["schema_version"] = "pcss.verifier-adapter-binding/v2"
            values["runtime_configuration_fingerprint"] = prepared.config.runtime_configuration_fingerprint
        return TargetAdapterBinding(**values)

    def verify(self, case: VerifierCase) -> AdapterOutcome:
        expected_reference_method = (
            "pending-token-serial-greedy-memory-state/v1"
            if self._commit_protocol == "pending_native"
            else "serial-greedy-memory-state/v2"
        )
        if (
            case.reference_provenance is not None
            and case.reference_provenance.reference_method != expected_reference_method
        ):
            raise ContractError("fixture reference method is incompatible with adapter commit protocol")
        prepared = self._frozen_prepared()
        operation = "verify" if self._commit_protocol == "serial" else "verify_pending"
        argument = "--pcss-verify-case" if self._commit_protocol == "serial" else "--pcss-verify-pending-case"
        request, seed_id = _verify_request(prepared.config, case, operation=operation)
        response = _run_bridge(prepared.config, (argument,), request)
        return _validate_verify_response(
            response,
            prepared.config,
            case,
            seed_id,
            commit_protocol=self._commit_protocol,
        )


def create_adapter() -> LlamaCppBridgeAdapter:
    """Factory for ``pcss.verifier_runner --adapter pcss.llamacpp_adapter:create_adapter``."""

    return LlamaCppBridgeAdapter()


def create_pending_adapter() -> LlamaCppBridgeAdapter:
    """Factory for the pending-token native-tail-trim correctness experiment."""

    return LlamaCppBridgeAdapter(commit_protocol="pending_native")
