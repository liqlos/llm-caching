"""Fail-closed command boundary for a future target-backend verifier run.

This module intentionally ships no Qwen/llama.cpp adapter. A future adapter
cannot self-label a fixture as target evidence: the runner observes the host,
hashes the adapter/backend/model files itself, compares those facts with an
observed manifest, and mints the permit needed by ``VerifierHarness``.
It produces no throughput or speedup metric.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, require_exact_keys, require_mapping, require_string, sha256
from .manifest import validate_manifest
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .verifier import (
    AdapterAvailability,
    HarnessReceipt,
    TargetAdapterBinding,
    TargetExecutionPermit,
    TargetFixtureProvenance,
    TargetHostIdentity,
    VerifierAdapter,
    VerifierCase,
    VerifierHarness,
)


TARGET_PROFILE = "macbook_pro_m4_24gb"
_TARGET_MEMORY_BYTES = 24 * 1024**3
_SUITES = frozenset({"k-sweep", "perfect-draft", "provider-floor", "forced-rejection", "long-mixed"})


class _FrozenAvailabilityAdapter:
    """Prevent an adapter from changing its execution class after authorization."""

    def __init__(self, adapter: VerifierAdapter, availability: AdapterAvailability) -> None:
        self._adapter = adapter
        self._availability = availability
        self.name = require_string(adapter.name, "adapter.name")

    def availability(self) -> AdapterAvailability:
        return self._availability

    def verify(self, case: VerifierCase):
        return self._adapter.verify(case)


_CASE_V1_KEYS = {
    "case_id",
    "draft_tokens",
    "target_tokens",
    "expected_committed_tokens",
    "expected_state_digest",
    "k",
    "include_bonus_on_full_accept",
}
_CASE_V2_KEYS = _CASE_V1_KEYS | {"initial_context_tokens"}
_CASE_V3_KEYS = _CASE_V2_KEYS | {"seed_id"}
_FIXTURE_TARGET_KEYS = {"schema_version", "provenance", "cases"}


def _case(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    provenance: TargetFixtureProvenance | None,
) -> VerifierCase:
    if schema_version == "pcss.m4-verifier-cases/v1":
        required = _CASE_V1_KEYS
    elif schema_version == "pcss.m4-verifier-cases/v2":
        required = _CASE_V2_KEYS
    else:
        required = _CASE_V3_KEYS
    if set(payload) != required:
        raise ContractError("verifier case has missing or unknown fields")
    initial_context_tokens = ()
    seed_id: str | None = None
    if schema_version in {"pcss.m4-verifier-cases/v2", "pcss.m4-verifier-cases/v3"}:
        initial_context_tokens = tuple(payload["initial_context_tokens"])
        if not initial_context_tokens:
            raise ContractError("target verifier cases require a non-empty initial_context_tokens sequence")
    if schema_version == "pcss.m4-verifier-cases/v3":
        seed_id = require_string(payload["seed_id"], "v3 verifier case.seed_id")
    return VerifierCase(
        case_id=payload["case_id"],
        draft_tokens=tuple(payload["draft_tokens"]),
        target_tokens=tuple(payload["target_tokens"]),
        expected_committed_tokens=tuple(payload["expected_committed_tokens"]),
        expected_state_digest=payload["expected_state_digest"],
        k=payload["k"],
        include_bonus_on_full_accept=payload["include_bonus_on_full_accept"],
        initial_context_tokens=initial_context_tokens,
        reference_provenance=provenance,
        seed_id=seed_id,
    )


def _fixture_provenance(payload: Mapping[str, Any]) -> TargetFixtureProvenance:
    require_exact_keys(
        payload,
        {
            "schema_version",
            "reference_method",
            "model_artifact_sha256",
            "runtime_configuration_fingerprint",
        },
        "verifier fixture provenance",
    )
    return TargetFixtureProvenance(
        schema_version=payload["schema_version"],
        reference_method=payload["reference_method"],
        model_artifact_sha256=payload["model_artifact_sha256"],
        runtime_configuration_fingerprint=payload["runtime_configuration_fingerprint"],
    )


def load_cases(path: Path) -> tuple[VerifierCase, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"unable to read verifier fixture: {path}") from error
    payload = require_mapping(payload, "verifier fixture")
    schema_version = payload.get("schema_version")
    provenance: TargetFixtureProvenance | None = None
    if schema_version == "pcss.m4-verifier-cases/v1":
        require_exact_keys(payload, {"schema_version", "cases"}, "verifier fixture")
    elif schema_version in {"pcss.m4-verifier-cases/v2", "pcss.m4-verifier-cases/v3"}:
        require_exact_keys(payload, _FIXTURE_TARGET_KEYS, "verifier fixture")
        provenance = _fixture_provenance(require_mapping(payload["provenance"], "verifier fixture provenance"))
    else:
        raise ContractError("verifier fixture must use pcss.m4-verifier-cases/v1, v2, or v3")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ContractError("verifier fixture must contain at least one case")
    cases = tuple(
        _case(
            require_mapping(item, "verifier case"),
            schema_version=schema_version,
            provenance=provenance,
        )
        for item in payload["cases"]
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise ContractError("verifier case ids must be unique")
    return cases


def _load_adapter(spec: str) -> VerifierAdapter:
    if spec.count(":") != 1:
        raise ContractError("adapter factory must be MODULE:CALLABLE")
    module_name, factory_name = spec.split(":", 1)
    try:
        factory: Callable[[], VerifierAdapter] = getattr(importlib.import_module(module_name), factory_name)
        adapter = factory()
    except (ImportError, AttributeError, TypeError) as error:
        raise ContractError(f"unable to load adapter factory {spec}") from error
    for required in ("name", "availability", "verify"):
        if not hasattr(adapter, required):
            raise ContractError(f"adapter from {spec} is missing {required}")
    return adapter


def _capture_availability(adapter: VerifierAdapter) -> AdapterAvailability:
    availability = adapter.availability()
    if not isinstance(availability, AdapterAvailability):
        raise ContractError("adapter.availability() must return AdapterAvailability")
    return availability


def _require_observed_target_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest["evidence_class"] != "LOCAL_MEASURED":
        raise ContractError("TARGET_BACKEND receipts require manifest.evidence_class=LOCAL_MEASURED")
    target = manifest["target"]
    if target["hardware"]["observation_status"] != "observed" or target["operating_system"]["observation_status"] != "observed":
        raise ContractError("TARGET_BACKEND receipts require observed hardware and OS manifest fields")
    for field in (
        "model_artifact",
        "quantization",
        "tokenizer",
        "chat_template",
        "generation_protocol",
        "llama_cpp",
        "context_kv",
        "benchmark",
    ):
        if manifest[field]["observation_status"] != "observed":
            raise ContractError(f"TARGET_BACKEND receipts require observed manifest.{field}")
    if manifest["verifier_gate"]["observation_status"] != "observed":
        raise ContractError("TARGET_BACKEND receipts require an observed verifier_gate")


def _sysctl_value(name: str) -> str:
    try:
        completed = subprocess.run(
            ("sysctl", "-n", name),
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise ContractError(f"unable to observe target host via sysctl {name}") from error
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise ContractError(f"unable to observe target host via sysctl {name}")
    return value


def _system_profiler_hardware_profile() -> Mapping[str, str]:
    """Read the immutable product class and model identifier from macOS."""

    try:
        completed = subprocess.run(
            ("system_profiler", "-json", "SPHardwareDataType"),
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise ContractError("unable to observe target host via system_profiler") from error
    if completed.returncode != 0 or not completed.stdout:
        raise ContractError("unable to observe target host via system_profiler")
    try:
        payload = require_mapping(json.loads(completed.stdout), "system_profiler hardware profile")
    except json.JSONDecodeError as error:
        raise ContractError("system_profiler hardware profile must be valid JSON") from error
    entries = payload.get("SPHardwareDataType")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ContractError("system_profiler hardware profile must contain exactly one hardware entry")
    entry = require_mapping(entries[0], "system_profiler hardware entry")
    values: dict[str, str] = {}
    for field in ("machine_model", "machine_name"):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"system_profiler hardware entry must contain {field}")
        values[field] = value.strip()
    return values


def observe_target_host() -> TargetHostIdentity:
    """Observe the future user-specified target only after TARGET_BACKEND is requested."""

    memory_text = _sysctl_value("hw.memsize")
    try:
        memory_bytes = int(memory_text)
    except ValueError as error:
        raise ContractError("sysctl hw.memsize must be an integer") from error
    identity = TargetHostIdentity.from_observed(
        target_profile=TARGET_PROFILE,
        operating_system=platform.system(),
        architecture=platform.machine(),
        hardware_model=_sysctl_value("hw.model"),
        chip=_sysctl_value("machdep.cpu.brand_string"),
        memory_bytes=memory_bytes,
    )
    if identity.operating_system != "Darwin":
        raise ContractError("TARGET_BACKEND requires the target macOS host")
    if identity.architecture != "arm64":
        raise ContractError("TARGET_BACKEND requires an arm64 target host")
    hardware_profile = _system_profiler_hardware_profile()
    if hardware_profile["machine_model"] != identity.hardware_model:
        raise ContractError("sysctl and system_profiler hardware model identifiers must agree")
    if hardware_profile["machine_name"] != "MacBook Pro":
        raise ContractError("TARGET_BACKEND requires the target MacBook Pro hardware class")
    if not identity.chip.startswith("Apple M4"):
        raise ContractError("TARGET_BACKEND requires an Apple M4-family chip")
    if identity.memory_bytes != _TARGET_MEMORY_BYTES:
        raise ContractError("TARGET_BACKEND requires the user-specified 24 GiB unified-memory target")
    return identity


def _adapter_source_path(adapter: VerifierAdapter, binding: TargetAdapterBinding) -> Path:
    actual_source = inspect.getsourcefile(type(adapter))
    if actual_source is None:
        raise ContractError("target adapter must have an inspectable Python source file")
    source_path = Path(actual_source).resolve()
    declared_path = Path(binding.adapter_source_path).resolve()
    if source_path != declared_path:
        raise ContractError("target adapter source path differs from runner-inspected adapter source")
    if not source_path.is_file():
        raise ContractError("target adapter source path is not a regular file")
    return source_path


def _observed_binding(adapter: VerifierAdapter, binding: TargetAdapterBinding) -> dict[str, str]:
    source_path = _adapter_source_path(adapter, binding)
    backend_path = Path(binding.backend_binary_path).resolve()
    model_path = Path(binding.model_artifact_path).resolve()
    for label, path in (("backend binary", backend_path), ("model artifact", model_path)):
        if not path.is_file():
            raise ContractError(f"target {label} path is not a regular file")
    return {
        "schema_version": binding.schema_version,
        "adapter_source_sha256": file_sha256(source_path),
        "backend_binary_sha256": file_sha256(backend_path),
        "model_artifact_sha256": file_sha256(model_path),
        "llama_cpp_build_fingerprint": binding.llama_cpp_build_fingerprint,
        "runtime_configuration_fingerprint": binding.runtime_configuration_fingerprint,
    }


def _target_binding(adapter: VerifierAdapter) -> TargetAdapterBinding:
    binding_method = getattr(adapter, "target_binding", None)
    if not callable(binding_method):
        raise ContractError("TARGET_BACKEND adapter must expose target_binding()")
    binding = binding_method()
    if not isinstance(binding, TargetAdapterBinding):
        raise ContractError("target_binding() must return TargetAdapterBinding")
    return binding


def capture_target_observation(adapter: VerifierAdapter) -> tuple[TargetHostIdentity, dict[str, str]]:
    """Capture future target identity and file hashes without calling ``verify()``."""

    binding = _target_binding(adapter)
    return observe_target_host(), _observed_binding(adapter, binding)


def _authorize_target_backend(
    adapter: VerifierAdapter,
    manifest: Mapping[str, Any],
    host_identity: TargetHostIdentity,
    observed_binding: Mapping[str, str],
) -> TargetExecutionPermit:
    gate = manifest["verifier_gate"]
    if gate["host_identity"] != host_identity.as_dict():
        raise ContractError("runner-observed host identity differs from manifest.verifier_gate")
    if gate["adapter_binding"] != observed_binding:
        raise ContractError("runner-computed adapter/backend/model binding differs from manifest.verifier_gate")
    if manifest["model_artifact"]["sha256"] != observed_binding["model_artifact_sha256"]:
        raise ContractError("runner-computed model hash differs from manifest.model_artifact.sha256")
    if manifest["llama_cpp"]["fingerprint"] != observed_binding["llama_cpp_build_fingerprint"]:
        raise ContractError("target adapter llama.cpp build fingerprint differs from manifest.llama_cpp")
    return TargetExecutionPermit(
        adapter_name=require_string(adapter.name, "adapter.name"),
        host_fingerprint=host_identity.fingerprint,
        adapter_binding_fingerprint=sha256(observed_binding),
    )


def _require_fixture_target_compatibility(
    cases: tuple[VerifierCase, ...], observed_binding: Mapping[str, str]
) -> None:
    """Reject a context-bearing fixture generated by a different target runtime."""

    provenances = {case.reference_provenance for case in cases if case.reference_provenance is not None}
    if not provenances:
        return
    if len(provenances) != 1:
        raise ContractError("target fixture cases must share one reference provenance")
    provenance = next(iter(provenances))
    assert provenance is not None
    if provenance.model_artifact_sha256 != observed_binding["model_artifact_sha256"]:
        raise ContractError("target fixture model fingerprint differs from runner-observed model artifact")
    if provenance.runtime_configuration_fingerprint != observed_binding["runtime_configuration_fingerprint"]:
        raise ContractError("target fixture runtime configuration differs from runner-observed binding")
    if any(case.seed_id is None for case in cases):
        raise ContractError("target fixture cases require an explicit seed_id")


def _not_run_receipt(suite: str, adapter: VerifierAdapter, reason: str) -> HarnessReceipt:
    return HarnessReceipt(
        suite=suite,
        adapter=require_string(adapter.name, "adapter.name"),
        status="NOT_RUN",
        execution_class="NOT_RUN",
        reason=reason,
        cases=(),
        metadata={"target_authorization": {"status": "rejected", "reason": reason}},
    )


def _run_suite(harness: VerifierHarness, suite: str, cases: tuple[VerifierCase, ...]) -> HarnessReceipt:
    suites = {
        "k-sweep": harness.k_sweep,
        "perfect-draft": harness.perfect_draft,
        "provider-floor": harness.provider_floor,
        "forced-rejection": harness.forced_rejection_every_position,
        "long-mixed": harness.long_mixed_trajectories,
    }
    return suites[suite](cases)


def execute_suite(
    *,
    adapter: VerifierAdapter,
    suite: str,
    fixture_path: Path,
    manifest_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    if suite not in _SUITES:
        raise ContractError(f"unknown verifier suite: {suite}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    cases = load_cases(fixture_path)
    availability = _capture_availability(adapter)
    frozen_adapter = _FrozenAvailabilityAdapter(adapter, availability)

    if availability.execution_class == "TARGET_BACKEND":
        try:
            _require_observed_target_manifest(manifest)
            host_identity, observed_binding = capture_target_observation(adapter)
            _require_fixture_target_compatibility(cases, observed_binding)
            permit = _authorize_target_backend(adapter, manifest, host_identity, observed_binding)
        except Exception as error:
            # Authorization errors must leave an auditable NOT_RUN receipt and never call verify().
            harness_receipt = _not_run_receipt(suite, adapter, f"target authorization rejected: {error}")
        else:
            harness_receipt = _run_suite(
                VerifierHarness(frozen_adapter, target_execution_permit=permit), suite, cases
            )
    else:
        harness_receipt = _run_suite(VerifierHarness(frozen_adapter), suite, cases)

    evidence_class = {
        "TARGET_BACKEND": "LOCAL_MEASURED",
        "OFFLINE_FIXTURE": "OFFLINE_FIXTURE",
        "NOT_RUN": "NOT_RUN",
    }[harness_receipt.execution_class]
    return make_receipt(
        evidence_class=evidence_class,
        command=recorded_module_command("pcss.verifier_runner", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={"fixture": file_sha256(fixture_path), "manifest": file_sha256(manifest_path)},
        status=harness_receipt.status,
        payload=json.loads(harness_receipt.as_json()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one future PCSS verifier suite through an external adapter.")
    parser.add_argument("--adapter", required=True, help="external MODULE:CALLABLE factory returning a VerifierAdapter")
    parser.add_argument("--suite", required=True, choices=tuple(sorted(_SUITES)))
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    receipt = execute_suite(
        adapter=_load_adapter(args.adapter),
        suite=args.suite,
        fixture_path=args.fixture,
        manifest_path=args.manifest,
        repository_root=args.repository_root.resolve(),
        command_arguments=command_arguments,
    )
    write_raw_result(args.output, receipt)
    if args.report is not None:
        write_compact_report(args.report, raw_result=args.output, title=f"PCSS verifier {args.suite}")
    return 0 if receipt.status in {"VERIFIED", "OFFLINE_FIXTURE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
