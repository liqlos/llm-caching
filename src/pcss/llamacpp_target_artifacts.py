"""Target-only seed and manifest artifacts for a future llama.cpp verifier run.

The seed command calls the native bridge's capability probe on the strict M4
target and copies its observed ``probe_token`` into the fixture-builder's
token-context format.  It deliberately has no tokenizer or fallback token.
The manifest command only derives a fully bound manifest from an immutable
preflight receipt, the same explicit config, and an explicit model descriptor.
Neither command runs a verifier case or measures runtime performance.
"""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, require_string, sha256
from .llamacpp_fixture_builder import BridgeConfig, SEEDS_SCHEMA_VERSION, load_bridge_config, load_fixture_seeds
from .manifest import MANIFEST_SCHEMA_VERSION, validate_manifest
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
from .verifier_preflight import OBSERVATION_SCHEMA_VERSION
from .verifier_runner import observe_target_host


CAPABILITY_REQUEST_SCHEMA_VERSION = "pcss.llama-bridge-request/v1"
CAPABILITY_SCHEMA_VERSION = "pcss.llama-bridge-capabilities/v2"
SEED_PLAN_SCHEMA_VERSION = "pcss.llamacpp-capability-seed-plan/v1"
MODEL_DESCRIPTOR_SCHEMA_VERSION = "pcss.target-model-descriptor/v1"
SEED_OBSERVATION_SCHEMA_VERSION = "pcss.llamacpp-capability-seed-observation/v1"
MANIFEST_BUILD_SCHEMA_VERSION = "pcss.observed-verifier-manifest-build/v1"

_SEED_PLAN_FIELDS = {
    "schema_version",
    "seed_id",
    "k_values",
    "provider_floor_k",
    "forced_rejection_k",
    "long_mixed",
}
_LONG_MIXED_FIELDS = {"context_steps", "k", "rejection_position"}
_MODEL_DESCRIPTOR_FIELDS = {
    "schema_version",
    "run_id",
    "model_artifact",
    "quantization",
    "tokenizer",
    "chat_template",
    "generation_protocol",
    "llama_cpp",
    "context_kv",
    "benchmark",
}
_DESCRIPTOR_MODEL_FIELDS = {"artifact_uri", "sha256", "source"}
_SOURCE_FIELDS = {"source_class", "locator", "revision", "artifact_name"}
_PROTOCOL_FIELDS = {"fingerprint", "parameters"}
_CAPABILITY_FIELDS = {
    "schema_version",
    "operation",
    "status",
    "backend",
    "state_export",
    "generation_mode",
    "greedy_tie_break",
    "raw_token_ids",
    "llama_c_api_version",
    "loaded_llama_cpp_library_path",
    "supports_gpu_offload",
    "supports_mmap",
    "supports_mlock",
    "vocabulary_size",
    "configuration",
    "observed_context",
    "state_components",
    "state_api",
    "declared_identity_fields",
    "chat_template_applied",
    "thinking_mode_applied",
    "sampler_rng_state",
    "probe_token",
    "probe_state_sha256",
    "probe_state_bytes",
    "restored_state_sha256",
    "restored_state_bytes",
    "state_round_trip_byte_exact",
    "operations",
}
_CAPABILITY_CONFIGURATION_FIELDS = {"model_artifact", "llama_cpp_library", "llama_cpp_revision", "runtime"}
_OBSERVED_CONTEXT_FIELDS = {"actual_n_ctx", "actual_n_batch", "actual_n_threads", "actual_n_threads_batch"}
_RECEIPT_FIELDS = {"schema_version", "evidence_class", "command", "git_revision", "input_fingerprints", "status", "payload"}
_PREFLIGHT_PAYLOAD_FIELDS = {
    "schema_version",
    "observation_status",
    "adapter",
    "declared_execution_class",
    "reason",
    "host_identity",
    "adapter_binding",
}
_ADAPTER_BINDING_FIELDS = {
    "schema_version",
    "adapter_source_sha256",
    "backend_binary_sha256",
    "model_artifact_sha256",
    "llama_cpp_build_fingerprint",
    "runtime_configuration_fingerprint",
}
_LLAMA_CPP_DESCRIPTOR_FIELDS = {"llama_cpp_revision", "llama_cpp_build_fingerprint", "llama_cpp_library_sha256"}
_CONTEXT_KV_DESCRIPTOR_FIELDS = {"runtime_configuration_fingerprint", "runtime"}
_CHAT_TEMPLATE_DESCRIPTOR_FIELDS = {"template"}
_GENERATION_PROTOCOL_DESCRIPTOR_FIELDS = {"generation_mode", "token_input", "thinking_mode"}
_BENCHMARK_DESCRIPTOR_FIELDS = {"measurement_status", "parameters"}


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
    if not isinstance(value, dict):
        raise ContractError(f"{field} must be one strict JSON object")
    return value


def _read_json_object(path: Path, field: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    try:
        if not resolved.is_file():
            raise OSError("not a regular file")
        return resolved, _strict_json_object(resolved.read_text(encoding="utf-8"), field)
    except OSError as error:
        raise ContractError(f"unable to read {field}: {path}") from error


def _require_sha256(value: object, field: str) -> str:
    value = require_string(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    return value


def _absolute_regular_path(value: object, field: str) -> Path:
    path = Path(require_string(value, field))
    if not path.is_absolute():
        raise ContractError(f"{field} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise OSError("not a regular file")
    except OSError as error:
        raise ContractError(f"{field} must resolve to a readable regular file") from error
    return resolved


@dataclass(frozen=True, slots=True)
class SeedPlan:
    plan_path: Path
    seed_id: str
    k_values: tuple[int, ...]
    provider_floor_k: int
    forced_rejection_k: int
    long_mixed: dict[str, int]


def load_seed_plan(path: Path) -> SeedPlan:
    plan_path, payload = _read_json_object(path, "llama.cpp capability seed plan")
    require_exact_keys(payload, _SEED_PLAN_FIELDS, "llama.cpp capability seed plan")
    if payload["schema_version"] != SEED_PLAN_SCHEMA_VERSION:
        raise ContractError("unsupported llama.cpp capability seed-plan schema")
    seed_id = require_string(payload["seed_id"], "llama.cpp capability seed plan.seed_id")
    values = payload["k_values"]
    if not isinstance(values, list) or not values:
        raise ContractError("llama.cpp capability seed plan.k_values must be a non-empty array")
    k_values = tuple(require_int(value, f"llama.cpp capability seed plan.k_values[{index}]", minimum=1) for index, value in enumerate(values))
    if len(set(k_values)) != len(k_values):
        raise ContractError("llama.cpp capability seed plan.k_values must not contain duplicates")
    long_mixed_value = require_mapping(payload["long_mixed"], "llama.cpp capability seed plan.long_mixed")
    require_exact_keys(long_mixed_value, _LONG_MIXED_FIELDS, "llama.cpp capability seed plan.long_mixed")
    long_mixed = {
        "context_steps": require_int(
            long_mixed_value["context_steps"], "llama.cpp capability seed plan.long_mixed.context_steps", minimum=1
        ),
        "k": require_int(long_mixed_value["k"], "llama.cpp capability seed plan.long_mixed.k", minimum=2),
        "rejection_position": require_int(
            long_mixed_value["rejection_position"],
            "llama.cpp capability seed plan.long_mixed.rejection_position",
            minimum=1,
        ),
    }
    if long_mixed["rejection_position"] >= long_mixed["k"]:
        raise ContractError("llama.cpp capability seed plan.long_mixed.rejection_position must be below k")
    return SeedPlan(
        plan_path=plan_path,
        seed_id=seed_id,
        k_values=tuple(sorted(k_values)),
        provider_floor_k=require_int(payload["provider_floor_k"], "llama.cpp capability seed plan.provider_floor_k", minimum=1),
        forced_rejection_k=require_int(payload["forced_rejection_k"], "llama.cpp capability seed plan.forced_rejection_k", minimum=1),
        long_mixed=long_mixed,
    )


def _capability_request(config: BridgeConfig) -> dict[str, Any]:
    return {
        "schema_version": CAPABILITY_REQUEST_SCHEMA_VERSION,
        "operation": "capabilities",
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
    }


def _run_capabilities(config: BridgeConfig) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            (str(config.bridge_executable), "--pcss-capabilities"),
            input=canonical_json(_capability_request(config)),
            capture_output=True,
            check=False,
            text=True,
            timeout=config.request_timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("pcss_llama_bridge capability process did not complete") from error
    if completed.returncode != 0:
        raise ContractError("pcss_llama_bridge capability process returned a non-zero exit status")
    return _strict_json_object(completed.stdout, "pcss_llama_bridge capability response")


def _validate_capability_response(response: Mapping[str, Any], config: BridgeConfig) -> int:
    require_exact_keys(response, _CAPABILITY_FIELDS, "pcss_llama_bridge capability response")
    if response["schema_version"] != CAPABILITY_SCHEMA_VERSION:
        raise ContractError("pcss_llama_bridge capability schema is unsupported")
    if response["operation"] != "capabilities" or response["status"] != "ok" or response["backend"] != "llama.cpp":
        raise ContractError("pcss_llama_bridge did not return a successful llama.cpp capability observation")
    if response["state_export"] != "model-architecture+memory-state/v2":
        raise ContractError("pcss_llama_bridge capability does not declare model-memory state export")
    if response["generation_mode"] != "greedy_argmax" or response["greedy_tie_break"] != "lowest-token-id/v1":
        raise ContractError("pcss_llama_bridge capability is not deterministic greedy argmax")
    if not _require_bool(response["raw_token_ids"], "pcss_llama_bridge capability response.raw_token_ids"):
        raise ContractError("pcss_llama_bridge capability does not use caller-supplied raw token ids")
    if response["state_api"] != "llama_state_get_data+llama_state_set_data/model-architecture+memory/v2":
        raise ContractError("pcss_llama_bridge capability does not declare the required model-memory state API")
    if response["chat_template_applied"] is not False or response["thinking_mode_applied"] is not False:
        raise ContractError("pcss_llama_bridge capability unexpectedly applied a template or thinking mode")
    if not _require_bool(response["state_round_trip_byte_exact"], "pcss_llama_bridge capability response.state_round_trip_byte_exact"):
        raise ContractError("pcss_llama_bridge capability did not prove a byte-exact state round trip")

    vocabulary_size = require_int(response["vocabulary_size"], "pcss_llama_bridge capability response.vocabulary_size", minimum=1)
    probe_token = require_int(response["probe_token"], "pcss_llama_bridge capability response.probe_token", minimum=0)
    if probe_token >= vocabulary_size:
        raise ContractError("pcss_llama_bridge capability probe_token is outside the observed vocabulary")
    for field in ("probe_state_sha256", "restored_state_sha256"):
        _require_sha256(response[field], f"pcss_llama_bridge capability response.{field}")
    if response["probe_state_sha256"] != response["restored_state_sha256"]:
        raise ContractError("pcss_llama_bridge capability state round trip has mismatched digests")
    for field in ("probe_state_bytes", "restored_state_bytes"):
        require_int(response[field], f"pcss_llama_bridge capability response.{field}", minimum=1)
    if response["probe_state_bytes"] != response["restored_state_bytes"]:
        raise ContractError("pcss_llama_bridge capability state round trip has mismatched byte counts")

    configuration = require_mapping(response["configuration"], "pcss_llama_bridge capability response.configuration")
    require_exact_keys(configuration, _CAPABILITY_CONFIGURATION_FIELDS, "pcss_llama_bridge capability response.configuration")
    expected_configuration = {
        "model_artifact": str(config.model_artifact),
        "llama_cpp_library": str(config.llama_cpp_library),
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(config.runtime),
    }
    if dict(configuration) != expected_configuration:
        raise ContractError("pcss_llama_bridge capability configuration differs from the explicit config")

    loaded_library = _absolute_regular_path(
        response["loaded_llama_cpp_library_path"], "pcss_llama_bridge capability response.loaded_llama_cpp_library_path"
    )
    if loaded_library != config.llama_cpp_library:
        raise ContractError("pcss_llama_bridge loaded a different llama.cpp library than the configured library")
    if file_sha256(loaded_library) != file_sha256(config.llama_cpp_library):
        raise ContractError("pcss_llama_bridge loaded llama.cpp library hash differs from the configured library")

    observed_context = require_mapping(response["observed_context"], "pcss_llama_bridge capability response.observed_context")
    require_exact_keys(observed_context, _OBSERVED_CONTEXT_FIELDS, "pcss_llama_bridge capability response.observed_context")
    expected_context = {
        "actual_n_ctx": config.runtime["n_ctx"],
        "actual_n_batch": config.runtime["n_batch"],
        "actual_n_threads": config.runtime["n_threads"],
        "actual_n_threads_batch": config.runtime["n_threads_batch"],
    }
    if dict(observed_context) != expected_context:
        raise ContractError("pcss_llama_bridge observed context differs from the explicit runtime config")
    if response["state_components"] != ["model_architecture_identity", "llama_memory"]:
        raise ContractError("pcss_llama_bridge capability model-memory state coverage is incomplete")
    if response["operations"] != ["capabilities", "reference", "verify"]:
        raise ContractError("pcss_llama_bridge capability operation set is incomplete")
    for field in (
        "llama_c_api_version",
        "declared_identity_fields",
        "sampler_rng_state",
    ):
        require_string(response[field], f"pcss_llama_bridge capability response.{field}")
    for field in ("supports_gpu_offload", "supports_mmap", "supports_mlock"):
        _require_bool(response[field], f"pcss_llama_bridge capability response.{field}")
    return probe_token


def _write_json_new(path: Path, payload: Mapping[str, Any], *, kind: str) -> str:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        raise ContractError(f"{kind} output path already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
    except OSError as error:
        raise ContractError(f"unable to write {kind} output: {resolved}") from error
    return file_sha256(resolved)


def _prepare_outputs(*, artifact: Path, raw_result: Path, report: Path, label: str) -> tuple[Path, Path, Path]:
    resolved = tuple(path.expanduser().resolve() for path in (artifact, raw_result, report))
    if len({str(path) for path in resolved}) != len(resolved):
        raise ContractError(f"{label}, raw-result, and report paths must be distinct")
    for path in resolved:
        if path.exists():
            raise ContractError(f"refuse to overwrite an existing {label} artifact: {path}")
    return resolved


def _input_fingerprints(config: BridgeConfig, *, additional: Mapping[str, str]) -> dict[str, str]:
    fingerprints = {
        "bridge_binary": file_sha256(config.bridge_executable),
        "config": file_sha256(config.config_path),
        "llama_cpp_library": file_sha256(config.llama_cpp_library),
        "model_artifact": file_sha256(config.model_artifact),
    }
    fingerprints.update(additional)
    return dict(sorted(fingerprints.items()))


def build_capability_seed_context(
    *,
    config_path: Path,
    plan_path: Path,
    output_path: Path,
    raw_result_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
    capability_runner: Callable[[BridgeConfig], Mapping[str, Any]] = _run_capabilities,
) -> ResultReceipt:
    """Produce fixture-builder seed contexts from one observed capability token."""

    config = load_bridge_config(config_path)
    plan = load_seed_plan(plan_path)
    input_fingerprints = _input_fingerprints(config, additional={"seed_plan": file_sha256(plan.plan_path)})
    output_path, raw_result_path, report_path = _prepare_outputs(
        artifact=output_path, raw_result=raw_result_path, report=report_path, label="seed-context"
    )
    git_revision = resolve_git_revision(repository_root)
    command = recorded_module_command("pcss.llamacpp_target_artifacts", command_arguments, repository_root)
    host_identity = target_observer()
    if not isinstance(host_identity, TargetHostIdentity):
        raise ContractError("target observer must return TargetHostIdentity")
    response = dict(capability_runner(config))
    probe_token = _validate_capability_response(response, config)
    if _input_fingerprints(config, additional={"seed_plan": file_sha256(plan.plan_path)}) != input_fingerprints:
        raise ContractError("configured bridge/model/config/seed plan changed while observing capability")

    seed_payload = {
        "schema_version": SEEDS_SCHEMA_VERSION,
        "k_values": list(plan.k_values),
        "provider_floor_k": plan.provider_floor_k,
        "forced_rejection_k": plan.forced_rejection_k,
        "long_mixed": dict(plan.long_mixed),
        "seeds": [{"seed_id": plan.seed_id, "initial_context_tokens": [probe_token]}],
    }
    seed_sha256 = _write_json_new(output_path, seed_payload, kind="seed-context")
    # Re-open through the fixture-builder parser: this output is its exact input
    # contract, rather than an adjacent lookalike schema.
    loaded_seeds = load_fixture_seeds(output_path)
    if loaded_seeds.contexts != ((plan.seed_id, (probe_token,)),):
        raise ContractError("written seed-context artifact does not preserve the observed capability token")

    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=command,
        git_revision=git_revision,
        input_fingerprints=input_fingerprints,
        status="SEED_CONTEXT_OBSERVED",
        payload={
            "schema_version": SEED_OBSERVATION_SCHEMA_VERSION,
            "claim_boundary": {
                "not_claimed": [
                    "hardware_performance",
                    "timing",
                    "provider_economics",
                    "live_speculative_decode",
                ]
            },
            "target_host": host_identity.as_dict(),
            "capability": response,
            "seed_artifact": {"path": str(output_path), "sha256": seed_sha256},
        },
    )
    write_raw_result(raw_result_path, receipt)
    write_compact_report(report_path, raw_result=raw_result_path, title="llama.cpp capability-derived PCSS seed context")
    return receipt


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    descriptor_path: Path
    run_id: str
    model_artifact: dict[str, Any]
    protocols: dict[str, dict[str, Any]]


def _load_protocol_descriptor(value: object, field: str) -> dict[str, Any]:
    protocol = dict(require_mapping(value, field))
    require_exact_keys(protocol, _PROTOCOL_FIELDS, field)
    fingerprint = _require_sha256(protocol["fingerprint"], f"{field}.fingerprint")
    parameters = dict(require_mapping(protocol["parameters"], f"{field}.parameters"))
    if fingerprint != sha256(parameters):
        raise ContractError(f"{field}.fingerprint does not match its parameters")
    return {"fingerprint": fingerprint, "parameters": parameters}


def load_model_descriptor(path: Path) -> ModelDescriptor:
    descriptor_path, payload = _read_json_object(path, "target model descriptor")
    require_exact_keys(payload, _MODEL_DESCRIPTOR_FIELDS, "target model descriptor")
    if payload["schema_version"] != MODEL_DESCRIPTOR_SCHEMA_VERSION:
        raise ContractError("unsupported target model descriptor schema")
    model_artifact = dict(require_mapping(payload["model_artifact"], "target model descriptor.model_artifact"))
    require_exact_keys(model_artifact, _DESCRIPTOR_MODEL_FIELDS, "target model descriptor.model_artifact")
    artifact_uri = require_string(model_artifact["artifact_uri"], "target model descriptor.model_artifact.artifact_uri")
    artifact_sha256 = _require_sha256(model_artifact["sha256"], "target model descriptor.model_artifact.sha256")
    source = dict(require_mapping(model_artifact["source"], "target model descriptor.model_artifact.source"))
    require_exact_keys(source, _SOURCE_FIELDS, "target model descriptor.model_artifact.source")
    for key in sorted(_SOURCE_FIELDS):
        require_string(source[key], f"target model descriptor.model_artifact.source.{key}")
    protocols = {
        field: _load_protocol_descriptor(payload[field], f"target model descriptor.{field}")
        for field in (
            "quantization",
            "tokenizer",
            "chat_template",
            "generation_protocol",
            "llama_cpp",
            "context_kv",
            "benchmark",
        )
    }
    return ModelDescriptor(
        descriptor_path=descriptor_path,
        run_id=require_string(payload["run_id"], "target model descriptor.run_id"),
        model_artifact={"artifact_uri": artifact_uri, "sha256": artifact_sha256, "source": source},
        protocols=protocols,
    )


def _validate_protocol_descriptor_matches_target(
    descriptor: ModelDescriptor,
    config: BridgeConfig,
    adapter_binding: Mapping[str, Any],
) -> None:
    llama_cpp = descriptor.protocols["llama_cpp"]["parameters"]
    require_exact_keys(llama_cpp, _LLAMA_CPP_DESCRIPTOR_FIELDS, "target model descriptor.llama_cpp.parameters")
    if llama_cpp["llama_cpp_revision"] != config.llama_cpp_revision:
        raise ContractError("target model descriptor llama.cpp revision differs from config")
    if _require_sha256(
        llama_cpp["llama_cpp_build_fingerprint"], "target model descriptor.llama_cpp.parameters.llama_cpp_build_fingerprint"
    ) != adapter_binding["llama_cpp_build_fingerprint"]:
        raise ContractError("target model descriptor llama.cpp build fingerprint differs from preflight binding")
    if _require_sha256(
        llama_cpp["llama_cpp_library_sha256"], "target model descriptor.llama_cpp.parameters.llama_cpp_library_sha256"
    ) != file_sha256(config.llama_cpp_library):
        raise ContractError("target model descriptor llama.cpp library hash differs from config")

    context_kv = descriptor.protocols["context_kv"]["parameters"]
    require_exact_keys(context_kv, _CONTEXT_KV_DESCRIPTOR_FIELDS, "target model descriptor.context_kv.parameters")
    if _require_sha256(
        context_kv["runtime_configuration_fingerprint"],
        "target model descriptor.context_kv.parameters.runtime_configuration_fingerprint",
    ) != config.runtime_configuration_fingerprint:
        raise ContractError("target model descriptor runtime configuration differs from config")
    if context_kv["runtime"] != dict(config.runtime):
        raise ContractError("target model descriptor context/KV runtime differs from config")

    chat_template = descriptor.protocols["chat_template"]["parameters"]
    require_exact_keys(chat_template, _CHAT_TEMPLATE_DESCRIPTOR_FIELDS, "target model descriptor.chat_template.parameters")
    if chat_template["template"] != config.runtime["chat_template"]:
        raise ContractError("target model descriptor chat template differs from config")

    generation_protocol = descriptor.protocols["generation_protocol"]["parameters"]
    require_exact_keys(
        generation_protocol,
        _GENERATION_PROTOCOL_DESCRIPTOR_FIELDS,
        "target model descriptor.generation_protocol.parameters",
    )
    if generation_protocol != {
        "generation_mode": "greedy_argmax",
        "token_input": "caller-supplied-raw-token-ids/v1",
        "thinking_mode": "disabled",
    }:
        raise ContractError("target model descriptor generation protocol is not the raw-token, no-thinking bridge protocol")

    benchmark = descriptor.protocols["benchmark"]["parameters"]
    require_exact_keys(benchmark, _BENCHMARK_DESCRIPTOR_FIELDS, "target model descriptor.benchmark.parameters")
    if benchmark["measurement_status"] != "not_run":
        raise ContractError("target model descriptor benchmark must remain not_run until a real benchmark receipt exists")
    require_mapping(benchmark["parameters"], "target model descriptor.benchmark.parameters.parameters")


def _read_preflight_receipt(path: Path) -> tuple[Path, dict[str, Any], TargetHostIdentity, dict[str, str]]:
    receipt_path, receipt = _read_json_object(path, "target observation preflight receipt")
    require_exact_keys(receipt, _RECEIPT_FIELDS, "target observation preflight receipt")
    if receipt["schema_version"] != "pcss.result/v1":
        raise ContractError("target observation preflight receipt has an unsupported result schema")
    if receipt["evidence_class"] != "NOT_RUN" or receipt["status"] != "NOT_RUN":
        raise ContractError("target observation preflight receipt must be an immutable NOT_RUN preflight")
    require_string(receipt["command"], "target observation preflight receipt.command")
    require_string(receipt["git_revision"], "target observation preflight receipt.git_revision")
    input_fingerprints = dict(require_mapping(receipt["input_fingerprints"], "target observation preflight receipt.input_fingerprints"))
    for key, value in input_fingerprints.items():
        require_string(key, "target observation preflight receipt.input_fingerprints key")
        _require_sha256(value, f"target observation preflight receipt.input_fingerprints.{key}")
    payload = dict(require_mapping(receipt["payload"], "target observation preflight receipt.payload"))
    require_exact_keys(payload, _PREFLIGHT_PAYLOAD_FIELDS, "target observation preflight receipt.payload")
    if payload["schema_version"] != OBSERVATION_SCHEMA_VERSION or payload["observation_status"] != "observed":
        raise ContractError("target observation preflight receipt does not contain observed target facts")
    if payload["declared_execution_class"] != "TARGET_BACKEND" or payload["reason"] is not None:
        raise ContractError("target observation preflight receipt is not a successful target-backend observation")
    require_string(payload["adapter"], "target observation preflight receipt.payload.adapter")
    host_mapping = dict(require_mapping(payload["host_identity"], "target observation preflight receipt.payload.host_identity"))
    host_identity = TargetHostIdentity(**host_mapping)
    binding = dict(require_mapping(payload["adapter_binding"], "target observation preflight receipt.payload.adapter_binding"))
    require_exact_keys(binding, _ADAPTER_BINDING_FIELDS, "target observation preflight receipt.payload.adapter_binding")
    if binding["schema_version"] != "pcss.verifier-adapter-binding/v2":
        raise ContractError("target observation preflight receipt uses an unsupported adapter binding schema")
    for key in _ADAPTER_BINDING_FIELDS - {"schema_version"}:
        _require_sha256(binding[key], f"target observation preflight receipt.payload.adapter_binding.{key}")
    for field, binding_key in (
        ("adapter_source", "adapter_source_sha256"),
        ("backend_binary", "backend_binary_sha256"),
        ("model_artifact", "model_artifact_sha256"),
        ("llama_cpp_build", "llama_cpp_build_fingerprint"),
        ("runtime_configuration", "runtime_configuration_fingerprint"),
    ):
        if input_fingerprints.get(field) != binding[binding_key]:
            raise ContractError(f"target observation preflight receipt input fingerprint {field} differs from adapter binding")
    return receipt_path, receipt, host_identity, binding


def _validate_preflight_matches_config(
    *,
    receipt: Mapping[str, Any],
    host_identity: TargetHostIdentity,
    binding: Mapping[str, str],
    config: BridgeConfig,
    target_observer: Callable[[], TargetHostIdentity],
    git_revision: str,
) -> None:
    if receipt["git_revision"] != git_revision:
        raise ContractError("target observation preflight receipt was not made at the current clean Git revision")
    observed_now = target_observer()
    if not isinstance(observed_now, TargetHostIdentity):
        raise ContractError("target observer must return TargetHostIdentity")
    if observed_now != host_identity:
        raise ContractError("target observation preflight host identity differs from the current target host")
    package_root = Path(__file__).resolve().parents[2]
    adapter_source = package_root / "src" / "pcss" / "llamacpp_adapter.py"
    if file_sha256(adapter_source) != binding["adapter_source_sha256"]:
        raise ContractError("target observation preflight adapter source hash differs from this checkout")
    if file_sha256(config.bridge_executable) != binding["backend_binary_sha256"]:
        raise ContractError("target observation preflight bridge hash differs from config")
    if file_sha256(config.model_artifact) != binding["model_artifact_sha256"]:
        raise ContractError("target observation preflight model hash differs from config")
    if config.runtime_configuration_fingerprint != binding["runtime_configuration_fingerprint"]:
        raise ContractError("target observation preflight runtime configuration differs from config")


def _observed_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "observation_status": "observed",
        "fingerprint": protocol["fingerprint"],
        "parameters": dict(protocol["parameters"]),
    }


def build_observed_verifier_manifest(
    *,
    config_path: Path,
    preflight_receipt_path: Path,
    descriptor_path: Path,
    output_path: Path,
    raw_result_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
) -> ResultReceipt:
    """Derive one immutable observed manifest from exact M4 preflight facts."""

    config = load_bridge_config(config_path)
    descriptor = load_model_descriptor(descriptor_path)
    preflight_path, preflight, host_identity, binding = _read_preflight_receipt(preflight_receipt_path)
    input_fingerprints = _input_fingerprints(
        config,
        additional={
            "preflight_receipt": file_sha256(preflight_path),
            "model_descriptor": file_sha256(descriptor.descriptor_path),
        },
    )
    output_path, raw_result_path, report_path = _prepare_outputs(
        artifact=output_path, raw_result=raw_result_path, report=report_path, label="observed-manifest"
    )
    git_revision = resolve_git_revision(repository_root)
    _validate_preflight_matches_config(
        receipt=preflight,
        host_identity=host_identity,
        binding=binding,
        config=config,
        target_observer=target_observer,
        git_revision=git_revision,
    )
    model_hash = file_sha256(config.model_artifact)
    if descriptor.model_artifact["sha256"] != model_hash:
        raise ContractError("target model descriptor artifact hash differs from the configured model")
    _validate_protocol_descriptor_matches_target(descriptor, config, binding)
    hardware = {
        "target_profile": host_identity.target_profile,
        "architecture": host_identity.architecture,
        "hardware_model": host_identity.hardware_model,
        "chip": host_identity.chip,
        "memory_bytes": host_identity.memory_bytes,
        "host_identity_fingerprint": host_identity.fingerprint,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": descriptor.run_id,
        # The manifest records directly observed target identity, artifact, and
        # binding facts.  It is therefore suitable as the observed input to a
        # later target verifier receipt, even though this command's own result
        # receipt remains NOT_RUN because it runs no verifier case.
        "evidence_class": "LOCAL_MEASURED",
        "target": {
            "hardware": {"observation_status": "observed", "value": hardware},
            "operating_system": {"observation_status": "observed", "value": host_identity.operating_system},
        },
        "model_artifact": {
            "observation_status": "observed",
            "artifact_uri": descriptor.model_artifact["artifact_uri"],
            "sha256": model_hash,
        },
        **{field: _observed_protocol(descriptor.protocols[field]) for field in descriptor.protocols},
        "verifier_gate": {
            "observation_status": "observed",
            "host_identity": host_identity.as_dict(),
            "adapter_binding": dict(binding),
        },
    }
    validate_manifest(manifest)
    manifest_sha256 = _write_json_new(output_path, manifest, kind="observed-manifest")
    if _input_fingerprints(
        config,
        additional={
            "preflight_receipt": file_sha256(preflight_path),
            "model_descriptor": file_sha256(descriptor.descriptor_path),
        },
    ) != input_fingerprints:
        raise ContractError("configured preflight/model/config inputs changed while building observed manifest")
    receipt = make_receipt(
        evidence_class="NOT_RUN",
        command=recorded_module_command("pcss.llamacpp_target_artifacts", command_arguments, repository_root),
        git_revision=git_revision,
        input_fingerprints=input_fingerprints,
        status="OBSERVED_MANIFEST_BUILT",
        payload={
            "schema_version": MANIFEST_BUILD_SCHEMA_VERSION,
            "claim_boundary": {
                "not_claimed": [
                    "hardware_performance",
                    "timing",
                    "provider_economics",
                    "live_speculative_decode",
                    "verifier_correctness",
                ]
            },
            "target_host": host_identity.as_dict(),
            "adapter_binding": dict(binding),
            "source": dict(descriptor.model_artifact["source"]),
            "manifest": {"path": str(output_path), "sha256": manifest_sha256},
        },
    )
    write_raw_result(raw_result_path, receipt)
    write_compact_report(report_path, raw_result=raw_result_path, title="PCSS observed verifier manifest")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create strict target-only PCSS seed or manifest artifacts.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    seed = subcommands.add_parser("seed", help="derive a nonempty fixture seed from native bridge capability probe_token")
    seed.add_argument("--config", required=True, type=Path)
    seed.add_argument("--plan", required=True, type=Path, help="explicit immutable k/trajectory seed plan JSON")
    seed.add_argument("--output", required=True, type=Path, help="new fixture-builder-compatible seed JSON")
    seed.add_argument("--raw-result", required=True, type=Path, help="new immutable capability observation receipt")
    seed.add_argument("--report", required=True, type=Path, help="new compact report derived from --raw-result")
    seed.add_argument("--repository-root", type=Path, default=Path.cwd())
    manifest = subcommands.add_parser("manifest", help="derive an observed verifier manifest from exact preflight facts")
    manifest.add_argument("--config", required=True, type=Path)
    manifest.add_argument("--preflight", required=True, type=Path, help="immutable observed target preflight receipt")
    manifest.add_argument("--descriptor", required=True, type=Path, help="explicit source/model descriptor JSON")
    manifest.add_argument("--output", required=True, type=Path, help="new observed manifest JSON")
    manifest.add_argument("--raw-result", required=True, type=Path, help="new immutable manifest-build receipt")
    manifest.add_argument("--report", required=True, type=Path, help="new compact report derived from --raw-result")
    manifest.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    if args.command == "seed":
        build_capability_seed_context(
            config_path=args.config,
            plan_path=args.plan,
            output_path=args.output,
            raw_result_path=args.raw_result,
            report_path=args.report,
            repository_root=args.repository_root.resolve(),
            command_arguments=command_arguments,
        )
    else:
        build_observed_verifier_manifest(
            config_path=args.config,
            preflight_receipt_path=args.preflight,
            descriptor_path=args.descriptor,
            output_path=args.output,
            raw_result_path=args.raw_result,
            report_path=args.report,
            repository_root=args.repository_root.resolve(),
            command_arguments=command_arguments,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
