"""Build target-bound verifier fixtures from a real llama.cpp reference bridge.

This module deliberately has no tokenizer, model, or synthetic token fallback.
Every target token and every expected mutable-state digest in the generated
fixtures comes from a separate ``pcss_llama_bridge --pcss-reference`` process.
It is a fixture source only: it makes no timing, capacity, provider-economics,
or live-decode claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, require_string, sha256
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


CONFIG_SCHEMA_VERSION = "pcss.llamacpp-adapter-config/v1"
SEEDS_SCHEMA_VERSION = "pcss.llamacpp-fixture-seeds/v1"
FIXTURE_SCHEMA_VERSION = "pcss.m4-verifier-cases/v2"
FIXTURE_PROVENANCE_SCHEMA_VERSION = "pcss.target-verifier-fixture-provenance/v1"
BRIDGE_REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
BRIDGE_RESPONSE_SCHEMA_VERSION = "pcss.llama-bridge-reference/v1"
REFERENCE_METHOD = "serial-greedy-full-state/v1"

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
_SEED_FILE_FIELDS = {
    "schema_version",
    "k_values",
    "provider_floor_k",
    "forced_rejection_k",
    "long_mixed",
    "seeds",
}
_SEED_FIELDS = {"seed_id", "initial_context_tokens"}
_LONG_MIXED_FIELDS = {"context_steps", "k", "rejection_position"}
_BRIDGE_REQUEST_FIELDS = {
    "schema_version",
    "operation",
    "model_artifact",
    "llama_cpp_library",
    "llama_cpp_revision",
    "runtime",
    "seed",
    "reference",
}
_BRIDGE_SEED_FIELDS = {"seed_id", "initial_context_tokens", "steps"}
_BRIDGE_REFERENCE_FIELDS = {"method"}
_BRIDGE_RESPONSE_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "generation_mode",
    "greedy_tie_break",
    "seed_id",
    "target_tokens",
    "vocabulary_size",
    "state_export",
    "mutable_state_digest",
    "completion_state_sha256",
    "completion_state_bytes",
    "prompt_state_sha256",
    "prompt_state_bytes",
    "configuration",
    "observed_context",
    "tokenization_boundary",
    "state_api",
    "sampler_rng_state",
    "declared_identity_fields",
    "chat_template_applied",
    "thinking_mode_applied",
}
_BRIDGE_CONFIGURATION_FIELDS = {"model_artifact", "llama_cpp_library", "llama_cpp_revision", "runtime"}
_BRIDGE_OBSERVED_CONTEXT_FIELDS = {
    "actual_n_ctx",
    "actual_n_batch",
    "actual_n_threads",
    "actual_n_threads_batch",
}
_SUITE_FILENAMES = {
    "k_sweep": "k-sweep.v2.json",
    "perfect_draft": "perfect-draft.v2.json",
    "provider_floor": "provider-floor.v2.json",
    "forced_rejection": "forced-rejection.v2.json",
    "long_mixed": "long-mixed.v2.json",
}


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
    if not isinstance(payload, dict):
        raise ContractError(f"{field} must be one strict JSON object")
    return payload


def _read_json_object(path: Path, field: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        if not resolved.is_file():
            raise OSError("not a regular file")
        return resolved, _strict_json_object(resolved.read_text(encoding="utf-8"), field)
    except OSError as error:
        raise ContractError(f"unable to read {field}: {path}") from error


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


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    config_path: Path
    bridge_executable: Path
    model_artifact: Path
    llama_cpp_library: Path
    llama_cpp_revision: str
    runtime: dict[str, Any]
    request_timeout_seconds: int
    runtime_configuration_fingerprint: str


def load_bridge_config(path: Path) -> BridgeConfig:
    config_path, payload = _read_json_object(path, "llama.cpp fixture-builder config")
    require_exact_keys(payload, _CONFIG_FIELDS, "llama.cpp fixture-builder config")
    if payload["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ContractError("unsupported llama.cpp fixture-builder config schema")
    if payload["bridge_protocol"] != "pcss.llama-bridge/v1":
        raise ContractError("unsupported llama.cpp bridge protocol")

    runtime = require_mapping(payload["runtime"], "llama.cpp fixture-builder config.runtime")
    require_exact_keys(runtime, _RUNTIME_FIELDS, "llama.cpp fixture-builder config.runtime")
    normalized_runtime = dict(runtime)
    require_int(normalized_runtime["n_ctx"], "llama.cpp fixture-builder config.runtime.n_ctx", minimum=1)
    require_int(normalized_runtime["n_batch"], "llama.cpp fixture-builder config.runtime.n_batch", minimum=1)
    if normalized_runtime["n_batch"] > normalized_runtime["n_ctx"]:
        raise ContractError("llama.cpp fixture-builder config.runtime.n_batch must not exceed n_ctx")
    require_int(normalized_runtime["n_gpu_layers"], "llama.cpp fixture-builder config.runtime.n_gpu_layers", minimum=0)
    if normalized_runtime["flash_attn"] not in {"auto", "enabled", "disabled"}:
        raise ContractError("llama.cpp fixture-builder config.runtime.flash_attn must be auto, enabled, or disabled")
    require_int(normalized_runtime["seed"], "llama.cpp fixture-builder config.runtime.seed", minimum=0)
    require_int(normalized_runtime["n_threads"], "llama.cpp fixture-builder config.runtime.n_threads", minimum=1)
    require_int(normalized_runtime["n_threads_batch"], "llama.cpp fixture-builder config.runtime.n_threads_batch", minimum=1)
    require_string(normalized_runtime["chat_template"], "llama.cpp fixture-builder config.runtime.chat_template")
    if normalized_runtime["thinking_mode"] != "disabled":
        raise ContractError("llama.cpp fixture-builder config.runtime.thinking_mode must be disabled")

    bridge_executable = _absolute_regular_path(payload["bridge_executable"], "bridge_executable", executable=True)
    model_artifact = _absolute_regular_path(payload["model_artifact"], "model_artifact")
    llama_cpp_library = _absolute_regular_path(payload["llama_cpp_library"], "llama_cpp_library")
    llama_cpp_revision = require_string(payload["llama_cpp_revision"], "llama_cpp_revision")
    fingerprint = sha256(
        {
            "schema_version": "pcss.llamacpp-runtime-config/v1",
            "bridge_protocol": payload["bridge_protocol"],
            "model_artifact": str(model_artifact),
            "llama_cpp_library": str(llama_cpp_library),
            "llama_cpp_revision": llama_cpp_revision,
            "runtime": normalized_runtime,
        }
    )
    return BridgeConfig(
        config_path=config_path,
        bridge_executable=bridge_executable,
        model_artifact=model_artifact,
        llama_cpp_library=llama_cpp_library,
        llama_cpp_revision=llama_cpp_revision,
        runtime=normalized_runtime,
        request_timeout_seconds=require_int(
            payload["request_timeout_seconds"], "request_timeout_seconds", minimum=1
        ),
        runtime_configuration_fingerprint=fingerprint,
    )


@dataclass(frozen=True, slots=True)
class LongMixedPlan:
    context_steps: int
    k: int
    rejection_position: int


@dataclass(frozen=True, slots=True)
class FixtureSeeds:
    seed_path: Path
    k_values: tuple[int, ...]
    provider_floor_k: int
    forced_rejection_k: int
    long_mixed: LongMixedPlan
    contexts: tuple[tuple[str, tuple[int, ...]], ...]


def load_fixture_seeds(path: Path) -> FixtureSeeds:
    seed_path, payload = _read_json_object(path, "llama.cpp fixture seeds")
    require_exact_keys(payload, _SEED_FILE_FIELDS, "llama.cpp fixture seeds")
    if payload["schema_version"] != SEEDS_SCHEMA_VERSION:
        raise ContractError("unsupported llama.cpp fixture-seeds schema")

    k_values_value = payload["k_values"]
    if not isinstance(k_values_value, list) or not k_values_value:
        raise ContractError("fixture seeds.k_values must be a non-empty array")
    k_values = tuple(require_int(value, f"fixture seeds.k_values[{index}]", minimum=1) for index, value in enumerate(k_values_value))
    if len(set(k_values)) != len(k_values):
        raise ContractError("fixture seeds.k_values must not contain duplicates")

    long_mixed_payload = require_mapping(payload["long_mixed"], "fixture seeds.long_mixed")
    require_exact_keys(long_mixed_payload, _LONG_MIXED_FIELDS, "fixture seeds.long_mixed")
    long_mixed = LongMixedPlan(
        context_steps=require_int(long_mixed_payload["context_steps"], "fixture seeds.long_mixed.context_steps", minimum=1),
        k=require_int(long_mixed_payload["k"], "fixture seeds.long_mixed.k", minimum=2),
        rejection_position=require_int(
            long_mixed_payload["rejection_position"], "fixture seeds.long_mixed.rejection_position", minimum=1
        ),
    )
    if long_mixed.rejection_position >= long_mixed.k:
        raise ContractError("fixture seeds.long_mixed.rejection_position must be below long_mixed.k")

    seed_values = payload["seeds"]
    if not isinstance(seed_values, list) or not seed_values:
        raise ContractError("fixture seeds.seeds must be a non-empty array")
    contexts: list[tuple[str, tuple[int, ...]]] = []
    for index, item in enumerate(seed_values):
        seed = require_mapping(item, f"fixture seeds.seeds[{index}]")
        require_exact_keys(seed, _SEED_FIELDS, f"fixture seeds.seeds[{index}]")
        contexts.append(
            (
                require_string(seed["seed_id"], f"fixture seeds.seeds[{index}].seed_id"),
                _require_tokens(seed["initial_context_tokens"], f"fixture seeds.seeds[{index}].initial_context_tokens", nonempty=True),
            )
        )
    if len({seed_id for seed_id, _ in contexts}) != len(contexts):
        raise ContractError("fixture seed ids must be unique")

    return FixtureSeeds(
        seed_path=seed_path,
        k_values=tuple(sorted(k_values)),
        provider_floor_k=require_int(payload["provider_floor_k"], "fixture seeds.provider_floor_k", minimum=1),
        forced_rejection_k=require_int(
            payload["forced_rejection_k"], "fixture seeds.forced_rejection_k", minimum=1
        ),
        long_mixed=long_mixed,
        contexts=tuple(contexts),
    )


@dataclass(frozen=True, slots=True)
class BridgeReference:
    seed_id: str
    target_tokens: tuple[int, ...]
    vocabulary_size: int
    mutable_state_digest: str


def _bridge_request(config: BridgeConfig, *, seed_id: str, context: tuple[int, ...], steps: int) -> dict[str, Any]:
    if steps < 1:
        raise ContractError("reference steps must be positive")
    if len(context) + steps > config.runtime["n_ctx"]:
        raise ContractError("reference context plus steps exceeds configured n_ctx")
    request = {
        "schema_version": BRIDGE_REQUEST_SCHEMA_VERSION,
        "operation": "reference",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": config.runtime,
        "seed": {
            "seed_id": seed_id,
            "initial_context_tokens": list(context),
            "steps": steps,
        },
        "reference": {"method": REFERENCE_METHOD},
    }
    require_exact_keys(request, _BRIDGE_REQUEST_FIELDS, "fixture-builder bridge request")
    require_exact_keys(request["seed"], _BRIDGE_SEED_FIELDS, "fixture-builder bridge request.seed")
    require_exact_keys(request["reference"], _BRIDGE_REFERENCE_FIELDS, "fixture-builder bridge request.reference")
    return request


def _run_reference(
    config: BridgeConfig,
    *,
    seed_id: str,
    context: tuple[int, ...],
    steps: int,
) -> BridgeReference:
    request = _bridge_request(config, seed_id=seed_id, context=context, steps=steps)
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-reference"),
            input=canonical_json(request) + "\n",
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("pcss_llama_bridge reference process did not complete") from error
    if completed.returncode != 0:
        raise ContractError("pcss_llama_bridge reference process returned a non-zero exit status")
    response = _strict_json_object(completed.stdout, "pcss_llama_bridge reference response")
    require_exact_keys(response, _BRIDGE_RESPONSE_FIELDS, "pcss_llama_bridge reference response")
    if response["schema_version"] != BRIDGE_RESPONSE_SCHEMA_VERSION:
        raise ContractError("pcss_llama_bridge reference response schema is unsupported")
    if response["operation"] != "reference":
        raise ContractError("pcss_llama_bridge reference response operation is invalid")
    if response["status"] != "ok" or response["backend"] != "llama.cpp":
        raise ContractError("pcss_llama_bridge reference response is not a successful llama.cpp result")
    if response["generation_mode"] != "greedy_argmax" or response["greedy_tie_break"] != "lowest-token-id/v1":
        raise ContractError("pcss_llama_bridge reference response is not deterministic serial greedy generation")
    if response["state_export"] != "full_context_state/v1":
        raise ContractError("pcss_llama_bridge reference response does not declare full-context state export")
    if response["tokenization_boundary"] != "caller-supplied-raw-token-ids/v1":
        raise ContractError("pcss_llama_bridge reference response has an unexpected tokenization boundary")
    if response["state_api"] != "llama_state_get_data+llama_state_set_data/full-context/v1":
        raise ContractError("pcss_llama_bridge reference response has an unexpected state API")
    if response["sampler_rng_state"] != "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1":
        raise ContractError("pcss_llama_bridge reference response has an unexpected sampler/RNG state")
    if response["declared_identity_fields"] != "caller-declared-not-verified-by-bridge/v1":
        raise ContractError("pcss_llama_bridge reference response does not disclose declared identity fields")
    if _require_bool(response["chat_template_applied"], "pcss_llama_bridge reference response.chat_template_applied"):
        raise ContractError("pcss_llama_bridge reference response unexpectedly applied a chat template")
    if _require_bool(response["thinking_mode_applied"], "pcss_llama_bridge reference response.thinking_mode_applied"):
        raise ContractError("pcss_llama_bridge reference response unexpectedly applied a thinking-mode policy")
    if response["seed_id"] != seed_id:
        raise ContractError("pcss_llama_bridge reference response seed id differs from request")
    target_tokens = _require_tokens(response["target_tokens"], "pcss_llama_bridge reference response.target_tokens", nonempty=True)
    if len(target_tokens) != steps:
        raise ContractError("pcss_llama_bridge reference response target token count differs from requested steps")
    vocabulary_size = require_int(
        response["vocabulary_size"], "pcss_llama_bridge reference response.vocabulary_size", minimum=1
    )
    if any(token >= vocabulary_size for token in target_tokens):
        raise ContractError("pcss_llama_bridge reference response contains a token outside its vocabulary")
    mutable_state_digest = _require_sha256(
        response["mutable_state_digest"], "pcss_llama_bridge reference response.mutable_state_digest"
    )
    completion_digest = _require_sha256(
        response["completion_state_sha256"], "pcss_llama_bridge reference response.completion_state_sha256"
    )
    if mutable_state_digest != completion_digest:
        raise ContractError("pcss_llama_bridge reference mutable-state digest differs from completion-state digest")
    require_int(
        response["completion_state_bytes"], "pcss_llama_bridge reference response.completion_state_bytes", minimum=1
    )
    _require_sha256(response["prompt_state_sha256"], "pcss_llama_bridge reference response.prompt_state_sha256")
    require_int(response["prompt_state_bytes"], "pcss_llama_bridge reference response.prompt_state_bytes", minimum=1)
    configuration = require_mapping(response["configuration"], "pcss_llama_bridge reference response.configuration")
    require_exact_keys(configuration, _BRIDGE_CONFIGURATION_FIELDS, "pcss_llama_bridge reference response.configuration")
    if configuration["model_artifact"] != str(config.model_artifact):
        raise ContractError("pcss_llama_bridge reference response model path differs from explicit config")
    if configuration["llama_cpp_library"] != str(config.llama_cpp_library):
        raise ContractError("pcss_llama_bridge reference response library path differs from explicit config")
    if configuration["llama_cpp_revision"] != config.llama_cpp_revision:
        raise ContractError("pcss_llama_bridge reference response llama.cpp revision differs from explicit config")
    echoed_runtime = require_mapping(configuration["runtime"], "pcss_llama_bridge reference response.configuration.runtime")
    if dict(echoed_runtime) != config.runtime:
        raise ContractError("pcss_llama_bridge reference response runtime differs from explicit config")
    observed_context = require_mapping(response["observed_context"], "pcss_llama_bridge reference response.observed_context")
    require_exact_keys(observed_context, _BRIDGE_OBSERVED_CONTEXT_FIELDS, "pcss_llama_bridge reference response.observed_context")
    observed_expected = {
        "actual_n_ctx": config.runtime["n_ctx"],
        "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }
    for field, expected in observed_expected.items():
        if require_int(observed_context[field], f"pcss_llama_bridge reference response.observed_context.{field}", minimum=1) != expected:
            raise ContractError("pcss_llama_bridge reference observed context differs from explicit config")
    return BridgeReference(
        seed_id=seed_id,
        target_tokens=target_tokens,
        vocabulary_size=vocabulary_size,
        mutable_state_digest=mutable_state_digest,
    )


def _require_prefix(reference: BridgeReference, expected: tuple[int, ...], field: str) -> None:
    if reference.target_tokens != expected:
        raise ContractError(f"serial greedy reference changed while deriving {field}")


def _distinct_rejection_token(target_token: int, vocabulary_size: int) -> int:
    require_int(target_token, "target token", minimum=0)
    vocabulary_size = require_int(vocabulary_size, "vocabulary size", minimum=1)
    candidate = (target_token + 1) % vocabulary_size
    if not 0 <= candidate < vocabulary_size or candidate == target_token:
        raise ContractError("unable to derive a distinct vocabulary-valid forced-rejection token")
    return candidate


def _provenance(*, model_artifact_sha256: str, runtime_configuration_fingerprint: str) -> dict[str, str]:
    return {
        "schema_version": FIXTURE_PROVENANCE_SCHEMA_VERSION,
        "reference_method": REFERENCE_METHOD,
        "model_artifact_sha256": _require_sha256(model_artifact_sha256, "model artifact sha256"),
        "runtime_configuration_fingerprint": _require_sha256(
            runtime_configuration_fingerprint, "runtime configuration fingerprint"
        ),
    }


def _case(
    *,
    case_id: str,
    context: tuple[int, ...],
    draft_tokens: tuple[int, ...],
    target_tokens: tuple[int, ...],
    expected_committed_tokens: tuple[int, ...],
    expected_state_digest: str,
    include_bonus_on_full_accept: bool,
) -> dict[str, Any]:
    if not context:
        raise ContractError("target verifier fixture case must have a non-empty initial context")
    if len(draft_tokens) < 1 or len(target_tokens) < len(draft_tokens):
        raise ContractError("target verifier fixture case must have a complete non-empty target sequence")
    if not expected_committed_tokens:
        raise ContractError("target verifier fixture case must commit at least one token")
    return {
        "case_id": require_string(case_id, "fixture case id"),
        "draft_tokens": list(draft_tokens),
        "target_tokens": list(target_tokens),
        "expected_committed_tokens": list(expected_committed_tokens),
        "expected_state_digest": _require_sha256(expected_state_digest, "fixture expected state digest"),
        "k": len(draft_tokens),
        "include_bonus_on_full_accept": include_bonus_on_full_accept,
        "initial_context_tokens": list(context),
    }


def _fixture(*, provenance: dict[str, str], cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ContractError("target verifier fixture suite must contain at least one case")
    ordered_cases = sorted(cases, key=lambda item: item["case_id"])
    if len({item["case_id"] for item in ordered_cases}) != len(ordered_cases):
        raise ContractError("target verifier fixture case ids must be unique")
    return {"schema_version": FIXTURE_SCHEMA_VERSION, "provenance": provenance, "cases": ordered_cases}


def _build_suite_payloads(config: BridgeConfig, seeds: FixtureSeeds, *, model_artifact_sha256: str) -> dict[str, dict[str, Any]]:
    provenance = _provenance(
        model_artifact_sha256=model_artifact_sha256,
        runtime_configuration_fingerprint=config.runtime_configuration_fingerprint,
    )
    k_sweep_cases: list[dict[str, Any]] = []
    perfect_draft_cases: list[dict[str, Any]] = []
    provider_floor_cases: list[dict[str, Any]] = []
    forced_rejection_cases: list[dict[str, Any]] = []
    long_mixed_cases: list[dict[str, Any]] = []

    for seed_id, initial_context in seeds.contexts:
        for k in seeds.k_values:
            exact = _run_reference(
                config,
                seed_id=seed_id,
                context=initial_context,
                steps=k,
            )
            k_sweep_cases.append(
                _case(
                    case_id=f"k-sweep:{seed_id}:k={k}",
                    context=initial_context,
                    draft_tokens=exact.target_tokens,
                    target_tokens=exact.target_tokens,
                    expected_committed_tokens=exact.target_tokens,
                    expected_state_digest=exact.mutable_state_digest,
                    include_bonus_on_full_accept=False,
                )
            )

            with_bonus = _run_reference(
                config,
                seed_id=seed_id,
                context=initial_context,
                steps=k + 1,
            )
            _require_prefix(
                BridgeReference(
                    seed_id=seed_id,
                    target_tokens=with_bonus.target_tokens[:k],
                    vocabulary_size=with_bonus.vocabulary_size,
                    mutable_state_digest=with_bonus.mutable_state_digest,
                ),
                exact.target_tokens,
                f"perfect-draft seed {seed_id} k={k}",
            )
            perfect_draft_cases.append(
                _case(
                    case_id=f"perfect-draft:{seed_id}:k={k}",
                    context=initial_context,
                    draft_tokens=with_bonus.target_tokens[:k],
                    target_tokens=with_bonus.target_tokens,
                    expected_committed_tokens=with_bonus.target_tokens,
                    expected_state_digest=with_bonus.mutable_state_digest,
                    include_bonus_on_full_accept=True,
                )
            )

        provider = _run_reference(
            config,
            seed_id=seed_id,
            context=initial_context,
            steps=seeds.provider_floor_k,
        )
        provider_wrong = _distinct_rejection_token(provider.target_tokens[0], provider.vocabulary_size)
        provider_prefix = _run_reference(
            config,
            seed_id=seed_id,
            context=initial_context,
            steps=1,
        )
        _require_prefix(provider_prefix, provider.target_tokens[:1], f"provider-floor seed {seed_id}")
        provider_floor_cases.append(
            _case(
                case_id=f"provider-floor:{seed_id}:k={seeds.provider_floor_k}",
                context=initial_context,
                draft_tokens=(provider_wrong,) + provider.target_tokens[1:],
                target_tokens=provider.target_tokens,
                expected_committed_tokens=provider.target_tokens[:1],
                expected_state_digest=provider_prefix.mutable_state_digest,
                include_bonus_on_full_accept=False,
            )
        )

        forced = _run_reference(
            config,
            seed_id=seed_id,
            context=initial_context,
            steps=seeds.forced_rejection_k,
        )
        for position, target_token in enumerate(forced.target_tokens):
            forced_wrong = _distinct_rejection_token(target_token, forced.vocabulary_size)
            draft = forced.target_tokens[:position] + (forced_wrong,) + forced.target_tokens[position + 1 :]
            committed_reference = _run_reference(
                config,
                seed_id=seed_id,
                context=initial_context,
                steps=position + 1,
            )
            _require_prefix(
                committed_reference,
                forced.target_tokens[: position + 1],
                f"forced-rejection seed {seed_id} position {position}",
            )
            forced_rejection_cases.append(
                _case(
                    case_id=f"forced-rejection:{seed_id}:k={seeds.forced_rejection_k}:position={position}",
                    context=initial_context,
                    draft_tokens=draft,
                    target_tokens=forced.target_tokens,
                    expected_committed_tokens=forced.target_tokens[: position + 1],
                    expected_state_digest=committed_reference.mutable_state_digest,
                    include_bonus_on_full_accept=False,
                )
            )

        plan = seeds.long_mixed
        trajectory = _run_reference(
            config,
            seed_id=seed_id,
            context=initial_context,
            steps=plan.context_steps + plan.k,
        )
        reconstructed_context = initial_context + trajectory.target_tokens[: plan.context_steps]
        continuation = trajectory.target_tokens[plan.context_steps :]
        reconstructed = _run_reference(
            config,
            seed_id=f"{seed_id}:reconstructed",
            context=reconstructed_context,
            steps=plan.k,
        )
        _require_prefix(reconstructed, continuation, f"long-mixed reconstructed seed {seed_id}")
        mixed_wrong = _distinct_rejection_token(
            continuation[plan.rejection_position], reconstructed.vocabulary_size
        )
        mixed_draft = (
            continuation[: plan.rejection_position]
            + (mixed_wrong,)
            + continuation[plan.rejection_position + 1 :]
        )
        mixed_commit = _run_reference(
            config,
            seed_id=f"{seed_id}:reconstructed",
            context=reconstructed_context,
            steps=plan.rejection_position + 1,
        )
        _require_prefix(
            mixed_commit,
            continuation[: plan.rejection_position + 1],
            f"long-mixed committed prefix seed {seed_id}",
        )
        long_mixed_cases.append(
            _case(
                case_id=(
                    f"long-mixed:{seed_id}:context-steps={plan.context_steps}:k={plan.k}:"
                    f"position={plan.rejection_position}"
                ),
                context=reconstructed_context,
                draft_tokens=mixed_draft,
                target_tokens=continuation,
                expected_committed_tokens=continuation[: plan.rejection_position + 1],
                expected_state_digest=mixed_commit.mutable_state_digest,
                include_bonus_on_full_accept=False,
            )
        )

    return {
        "k_sweep": _fixture(provenance=provenance, cases=k_sweep_cases),
        "perfect_draft": _fixture(provenance=provenance, cases=perfect_draft_cases),
        "provider_floor": _fixture(provenance=provenance, cases=provider_floor_cases),
        "forced_rejection": _fixture(provenance=provenance, cases=forced_rejection_cases),
        "long_mixed": _fixture(provenance=provenance, cases=long_mixed_cases),
    }


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise ContractError(f"fixture output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
    except OSError as error:
        raise ContractError(f"unable to write fixture output: {path}") from error
    return file_sha256(path)


def _prepare_output_paths(fixture_directory: Path, raw_result_path: Path, report_path: Path) -> dict[str, Path]:
    fixture_directory = fixture_directory.expanduser().resolve()
    outputs = {suite: fixture_directory / filename for suite, filename in _SUITE_FILENAMES.items()}
    all_paths = [*outputs.values(), raw_result_path.expanduser().resolve(), report_path.expanduser().resolve()]
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ContractError("fixture, raw-result, and report paths must be distinct")
    for path in all_paths:
        if path.exists():
            raise ContractError(f"refuse to overwrite an existing fixture-builder artifact: {path}")
    return outputs


def _input_fingerprints(config: BridgeConfig, seeds: FixtureSeeds) -> dict[str, str]:
    return {
        "bridge_binary": file_sha256(config.bridge_executable),
        "config": file_sha256(config.config_path),
        "llama_cpp_library": file_sha256(config.llama_cpp_library),
        "model_artifact": file_sha256(config.model_artifact),
        "seed_contexts": file_sha256(seeds.seed_path),
    }


def build_target_fixtures(
    *,
    config_path: Path,
    seeds_path: Path,
    fixture_directory: Path,
    raw_result_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
) -> ResultReceipt:
    """Write suite-specific v2 fixtures, then an immutable raw receipt and report.

    The target observer runs before any bridge invocation.  Tests pass a local
    fake observer; the CLI always uses the strict M4 observer from the runner.
    """

    config = load_bridge_config(config_path)
    seeds = load_fixture_seeds(seeds_path)
    outputs = _prepare_output_paths(fixture_directory, raw_result_path, report_path)
    git_revision = resolve_git_revision(repository_root)
    command = recorded_module_command("pcss.llamacpp_fixture_builder", command_arguments, repository_root)
    host_identity = target_observer()
    if not isinstance(host_identity, TargetHostIdentity):
        raise ContractError("target observer must return TargetHostIdentity")
    input_fingerprints = _input_fingerprints(config, seeds)
    model_artifact_sha256 = input_fingerprints["model_artifact"]
    payloads = _build_suite_payloads(config, seeds, model_artifact_sha256=model_artifact_sha256)
    if _input_fingerprints(config, seeds) != input_fingerprints:
        raise ContractError("configured bridge/model/config/seed inputs changed while building fixtures")

    output_details: dict[str, dict[str, Any]] = {}
    for suite in sorted(payloads):
        output_path = outputs[suite]
        output_details[suite] = {
            "path": str(output_path),
            "sha256": _write_json_new(output_path, payloads[suite]),
            "case_count": len(payloads[suite]["cases"]),
        }

    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=command,
        git_revision=git_revision,
        input_fingerprints=input_fingerprints,
        status="FIXTURES_BUILT",
        payload={
            "schema_version": "pcss.llamacpp-fixture-build/v1",
            "claim_boundary": {
                "fixture_source": "serial-greedy-full-state/v1",
                "not_claimed": [
                    "hardware_performance",
                    "timing",
                    "provider_economics",
                    "live_speculative_decode",
                ],
            },
            "target_host": host_identity.as_dict(),
            "provenance": _provenance(
                model_artifact_sha256=model_artifact_sha256,
                runtime_configuration_fingerprint=config.runtime_configuration_fingerprint,
            ),
            "fixtures": output_details,
        },
    )
    write_raw_result(raw_result_path, receipt)
    write_compact_report(report_path, raw_result=raw_result_path, title="llama.cpp target verifier fixture build")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build target-bound PCSS verifier fixtures from a real llama.cpp reference bridge."
    )
    parser.add_argument("--config", required=True, type=Path, help="explicit llama.cpp adapter config JSON")
    parser.add_argument("--seeds", required=True, type=Path, help="explicit token-context seed JSON")
    parser.add_argument("--fixture-directory", required=True, type=Path, help="new directory entries for suite fixtures")
    parser.add_argument("--raw-result", required=True, type=Path, help="new immutable machine-readable result JSON")
    parser.add_argument("--report", required=True, type=Path, help="new compact report derived from --raw-result")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    build_target_fixtures(
        config_path=args.config,
        seeds_path=args.seeds,
        fixture_directory=args.fixture_directory,
        raw_result_path=args.raw_result,
        report_path=args.report,
        repository_root=args.repository_root.resolve(),
        command_arguments=command_arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
