"""Future-only target observation without executing a verifier case.

This command can capture the host and adapter file identities needed to fill a
future M4 verifier gate.  It is deliberately not an acceptance run: every
result receipt stays ``NOT_RUN`` and this module never calls ``adapter.verify``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, sha256
from .results import (
    ResultReceipt,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .verifier import AdapterAvailability, VerifierAdapter
from .verifier_runner import capture_target_observation


OBSERVATION_SCHEMA_VERSION = "pcss.m4-target-observation/v1"


def _safe_adapter_name(adapter: VerifierAdapter) -> str:
    name = getattr(adapter, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"{type(adapter).__module__}.{type(adapter).__qualname__}"


def _adapter_spec_fingerprint(adapter: VerifierAdapter, adapter_spec: str | None) -> str:
    identity = adapter_spec or f"{type(adapter).__module__}:{type(adapter).__qualname__}"
    return sha256({"adapter_spec": identity})


def _not_run_payload(
    *,
    adapter: VerifierAdapter,
    availability: AdapterAvailability | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_status": "not_run",
        "adapter": _safe_adapter_name(adapter),
        "declared_execution_class": availability.execution_class if availability is not None else None,
        "reason": reason,
        "host_identity": None,
        "adapter_binding": None,
    }


def _observed_payload(
    *,
    adapter: VerifierAdapter,
    availability: AdapterAvailability,
    host_identity: Mapping[str, object],
    adapter_binding: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_status": "observed",
        "adapter": _safe_adapter_name(adapter),
        "declared_execution_class": availability.execution_class,
        "reason": None,
        "host_identity": dict(host_identity),
        "adapter_binding": dict(adapter_binding),
    }


def preflight_target_observation(
    *,
    adapter: VerifierAdapter,
    adapter_spec: str | None,
    repository_root: Path,
    command_arguments: Sequence[str],
) -> ResultReceipt:
    """Capture future target facts, or record an honest ``NOT_RUN`` boundary.

    The adapter is queried for availability exactly once.  Only a declared
    ``TARGET_BACKEND`` proceeds to the runner-owned capture helper; that helper
    requires a real ``TargetAdapterBinding`` and independently hashes source,
    backend, and model files.  No verifier fixture or decode call is involved.
    """

    availability: AdapterAvailability | None = None
    try:
        candidate = adapter.availability()
        if not isinstance(candidate, AdapterAvailability):
            raise ContractError("adapter.availability() must return AdapterAvailability")
        availability = candidate
    except Exception as error:
        payload = _not_run_payload(
            adapter=adapter,
            availability=None,
            reason=f"adapter availability unavailable: {error}",
        )
    else:
        if availability.execution_class != "TARGET_BACKEND":
            payload = _not_run_payload(
                adapter=adapter,
                availability=availability,
                reason="adapter does not declare TARGET_BACKEND",
            )
        else:
            try:
                host_identity, adapter_binding = capture_target_observation(adapter)
            except Exception as error:
                payload = _not_run_payload(
                    adapter=adapter,
                    availability=availability,
                    reason=f"target observation unavailable: {error}",
                )
            else:
                payload = _observed_payload(
                    adapter=adapter,
                    availability=availability,
                    host_identity=host_identity.as_dict(),
                    adapter_binding=adapter_binding,
                )

    input_fingerprints = {"adapter_spec": _adapter_spec_fingerprint(adapter, adapter_spec)}
    if payload["observation_status"] == "observed":
        binding = payload["adapter_binding"]
        assert isinstance(binding, dict)
        input_fingerprints.update(
            {
                "adapter_source": binding["adapter_source_sha256"],
                "backend_binary": binding["backend_binary_sha256"],
                "model_artifact": binding["model_artifact_sha256"],
            }
        )
    return make_receipt(
        evidence_class="NOT_RUN",
        command=recorded_module_command("pcss.verifier_preflight", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints=input_fingerprints,
        status="NOT_RUN",
        payload=payload,
    )


def write_target_observation(path: Path, receipt: ResultReceipt) -> str:
    """Write one immutable preflight artifact after checking its exact payload shape."""

    if receipt.evidence_class != "NOT_RUN" or receipt.status != "NOT_RUN":
        raise ContractError("target observation receipts must remain NOT_RUN")
    payload = receipt.payload
    required = {
        "schema_version",
        "observation_status",
        "adapter",
        "declared_execution_class",
        "reason",
        "host_identity",
        "adapter_binding",
    }
    if set(payload) != required or payload["schema_version"] != OBSERVATION_SCHEMA_VERSION:
        raise ContractError("target observation payload has an invalid shape")
    if payload["observation_status"] not in {"observed", "not_run"}:
        raise ContractError("target observation payload has an invalid status")
    if payload["observation_status"] == "not_run":
        if payload["host_identity"] is not None or payload["adapter_binding"] is not None:
            raise ContractError("NOT_RUN target observation must not contain observed target facts")
    elif payload["host_identity"] is None or payload["adapter_binding"] is None:
        raise ContractError("observed target observation requires host identity and adapter binding")
    return write_raw_result(path, receipt)


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture future PCSS target identity and binding without running verifier cases."
    )
    parser.add_argument("--adapter", required=True, help="MODULE:CALLABLE factory for a future target adapter")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional compact report written only after the immutable raw observation",
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    command_arguments = tuple(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(command_arguments)
    receipt = preflight_target_observation(
        adapter=_load_adapter(args.adapter),
        adapter_spec=args.adapter,
        repository_root=args.repository_root.resolve(),
        command_arguments=command_arguments,
    )
    if args.report is not None and args.report.resolve() == args.output.resolve():
        raise ContractError("--report must not replace the immutable raw observation")
    write_target_observation(args.output, receipt)
    if args.report is not None:
        write_compact_report(
            args.report,
            raw_result=args.output,
            title="PCSS target observation preflight",
        )
    return 0 if receipt.payload["observation_status"] == "observed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
