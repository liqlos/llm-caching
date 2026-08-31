"""Preregistered comparative benchmark runner for a live ``llama-server``.

The runner deliberately performs no tuning and makes no speedup decision.  It
starts one fresh, long-lived process for each frozen arm, sends the same
chronological OpenAI-compatible request stream, and preserves the raw evidence
needed by a separate preregistered analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MANIFEST_SCHEMA = "pcss.live-server-benchmark-manifest/v1"
WORKLOAD_SCHEMA = "pcss.live-server-workload/v1"
REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA = (
    "pcss.live-server-workload-redacted-current-source/v1"
)
RESULT_SCHEMA = "pcss.live-server-benchmark-result/v1"
ARM_NAMES = ("baseline", "upstream-ngram-cache", "pcss-cache")
NGRAM_MAX_DRAFT_TOKENS = 2
CURRENT_SOURCE_REQUEST_KEY = "pcss_current_source"
CURRENT_SOURCE_SCHEMA = "pcss.current-source/v1"
CURRENT_SOURCE_MAX_TOKENS = 16_384
CURRENT_SOURCE_MAX_TOKEN_ID = 2_147_483_647
CURRENT_SOURCE_COMMITMENT_SCHEMA = "pcss.current-source-token-commitment/v1"
CAPTURE_SOURCE_COMMITMENT_SCHEMA = "pcss.current-source-capture-commitment/v1"
CAPTURE_RECEIPT_SCHEMA_V1 = "pcss.project-workload-capture-receipt/v1"
CAPTURE_RECEIPT_SCHEMA_V2 = "pcss.project-workload-capture-receipt/v2"
CURRENT_SOURCE_MATCHING_MODE = "matching"
CURRENT_SOURCE_MISMATCH_CONTROL_MODE = "compatibility_mismatch_abstention_control"
WORKLOAD_CLASSES = (
    "synthetic_control",
    "project_naturalistic_development",
    "agent_assisted_calibration",
    "owner_representative_holdout",
)
_MANIFEST_KEYS = {
    "schema_version",
    "study_id",
    "frozen",
    "git_revision",
    "patch_sha256",
    "llama_cpp_revision",
    "runtime_fingerprint",
    "hardware_os",
    "workload_sha256",
    "workload_class",
    "workload_provenance",
    "eligibility_rules",
    "calibration_split",
    "holdout_split",
    "clustering",
    "holdout",
    "model_sha256",
    "server_sha256",
    "server_args",
    "request_defaults",
    "arms",
    "arm_order",
    "counterbalance",
    "state_definitions",
    "arm_store_initialization",
    "readiness",
    "request_timeout_seconds",
    "request_count",
    "effect_gates",
    "resource_ceilings",
    "allowed_policy_families",
    "finite_parameter_grids",
    "budgets",
    "stopping_rule",
    "kill_gates",
    "exclusion_rules",
    "claim_boundary",
}
_POLICY_BINDING_KEYS = {"policy_id", "policy_spec_sha256"}
_CANONICAL_POLICY_ID = re.compile(
    r"^pcss\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)*/v[1-9][0-9]*$"
)
_POLICY_INFO_LINE = re.compile(
    r'^llamacpp:pcss_policy_info\{policy_id="(?P<policy_id>[^"\\]+)",'
    r'policy_spec_sha256="(?P<policy_spec_sha256>[0-9a-f]{64})"\} '
    r'(?P<value>[0-9]+(?:\.[0-9]+)?)$'
)
_WORKLOAD_KEYS = {"schema_version", "requests"}
_REQUEST_KEYS = {"request_id", "chronological_index", "endpoint", "body"}
_REDACTED_CURRENT_SOURCE_WORKLOAD_KEYS = {
    "schema_version",
    "original_workload_sha256",
    "requests",
}
_REDACTED_CURRENT_SOURCE_REQUEST_KEYS = _REQUEST_KEYS | {
    "pcss_current_source_commitment",
}
_CURRENT_SOURCE_KEYS = {
    "schema_version",
    "compatibility_fingerprint",
    "source_id",
    "visible_token_ids",
}
_CURRENT_SOURCE_COMMITMENT_KEYS = {
    "schema_version",
    "token_count",
    "tokens_sha256",
    "attachment_mode",
}
_CAPTURE_SOURCE_COMMITMENT_KEYS = {
    "schema_version",
    "source_envelope_sha256",
    "token_count",
    "tokens_sha256",
}
_CAPTURE_RECEIPT_V1_KEYS = {
    "schema_version",
    "source_event_count",
    "selected_event_count",
    "episodes",
    "per_episode",
    "source_sha256",
    "workload_sha256",
    "capture_tool_version",
    "origin",
    "request_episode_map",
    "final_event_sha256",
    "selection_rule",
    "contains_responses",
}
_CAPTURE_RECEIPT_V2_KEYS = _CAPTURE_RECEIPT_V1_KEYS | {
    "source_sidecar_sha256",
    "source_sidecar_event_count",
    "source_attachment_count",
    "source_attachment_commitments",
    "final_sidecar_event_sha256",
}
_READINESS_KEYS = {"path", "timeout_seconds", "max_attempts", "interval_seconds"}
_COUNTERBALANCE_KEYS = {"method", "sequence_id"}
_EFFECT_GATE_KEYS = {
    "primary_metric",
    "ratio_direction",
    "baseline_over_pcss_lcb95_min",
    "miss_heavy_p95_overhead_ucb95_max",
    "confidence_method",
    "exact_output_required",
}
_LADDER_ACTIVATION_GATE_KEYS = {
    "persistent_32_hit_requests_min",
    "persistent_32_hit_episodes_min",
    "persistent_32_accepted_tokens_min",
    "persistent_64_hit_requests_max",
    "persistent_32_suppressions_max",
    "fallback_disable_events_max",
}
_CURRENT_SOURCE_ACTIVATION_GATE_KEYS = {
    "current_source_matching_requests_min",
    "current_source_lookup_requests_min",
    "current_source_hit_requests_min",
    "current_source_hit_episodes_min",
    "current_source_accepted_tokens_min",
    "current_source_peak_index_bytes_min",
    "current_source_rejections_max",
    "current_source_ambiguities_max",
    "current_source_abstentions_max",
    "current_source_mismatch_control_requests_min",
    "current_source_mismatch_abstentions_min",
}
_CURRENT_SOURCE_REQUIRED_METRICS = (
    "pcss_current_source_lookups_total",
    "pcss_current_source_hits_total",
    "pcss_current_source_ambiguities_total",
    "pcss_current_source_abstentions_total",
    "pcss_current_source_draft_tokens_total",
    "pcss_current_source_accepted_tokens_total",
    "pcss_current_source_rejections_total",
    "pcss_current_source_peak_index_bytes",
    "pcss_current_source_mismatch_abstentions_total",
)
_HARDWARE_OS_KEYS = {"hardware", "os"}
_PROVENANCE_KEYS = {
    "source",
    "capture_protocol",
    "captured_before_arm_results",
    "immutable",
}
_AGENT_ASSISTED_PROVENANCE_KEYS = _PROVENANCE_KEYS | {
    "origin",
    "capture_receipt_sha256",
    "source_event_sha256",
}
_OWNER_REPRESENTATIVE_PROVENANCE_KEYS = _PROVENANCE_KEYS | {
    "origin",
    "capture_receipt_sha256",
    "source_event_sha256",
}
_CURRENT_SOURCE_CAPTURE_PROVENANCE_KEYS = _PROVENANCE_KEYS | {
    "origin",
    "capture_receipt_sha256",
    "source_event_sha256",
    "source_sidecar_sha256",
    "source_attachment_count",
}
_CALIBRATION_SPLIT_KEYS = {"request_ids", "tuning_permitted"}
_HOLDOUT_SPLIT_KEYS = {"request_ids", "one_shot"}
_CLUSTERING_KEYS = {"method", "request_cluster_ids"}
_STATE_DEFINITION_KEYS = {"warm", "cold"}
_STORE_INITIALIZATION_KEYS = {"state", "initialization", "snapshot_sha256"}
_RESOURCE_CEILING_KEYS = {
    "ram_mib",
    "disk_gib",
    "startup_seconds",
    "compaction_write_amplification_max",
}
_BUDGET_KEYS = {"max_requests", "max_wall_clock_seconds", "max_artifact_bytes"}
_FORBIDDEN_SERVER_ARGS = {
    "-m",
    "--model",
    "--port",
    "--host",
    "--metrics",
    "--spec-type",
    "--pcss-store",
    "--pcss-fingerprint",
    "--pcss-namespace",
    "--pcss-max-disk-gib",
    "--pcss-ram-mib",
    "--pcss-read-only",
    "--pcss-disable-learning",
    "--pcss-disable-current-context",
    "--pcss-enable-current-source",
    "--pcss-max-draft-tokens",
    "--spec-draft-n-max",
}
_FUTURE_KEYS = {
    "answer",
    "candidate",
    "candidate_tokens",
    "committed_token_ids",
    "completion",
    "expected",
    "expected_output",
    "expected_text",
    "future",
    "future_tokens",
    "output",
    "output_text",
    "target",
    "target_tokens",
}


class BenchmarkContractError(ValueError):
    """Raised when an input or live observation violates the frozen protocol."""


def _assert_exclusive_local_compute() -> None:
    """Fail before hashing or launch when competing local compute is active."""

    try:
        completed = subprocess.run(
            ("ps", "-axo", "pid=,ppid=,command="),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchmarkContractError("cannot verify exclusive local compute") from error
    if completed.returncode != 0:
        raise BenchmarkContractError("cannot verify exclusive local compute")
    own_pid = os.getpid()
    processes: dict[int, tuple[int, str]] = {}
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.strip().split(maxsplit=2)
        if (
            len(fields) != 3
            or not fields[0].isdigit()
            or not fields[1].isdigit()
        ):
            continue
        processes[int(fields[0])] = (int(fields[1]), fields[2])

    # Invoking the runner through a shell/wrapper necessarily puts the literal
    # `--server .../llama-server` in one or more ancestor command lines. Those
    # wrappers are not competing compute. Exclude only this process lineage;
    # every independent llama/build process still fails the preflight.
    own_lineage: set[int] = set()
    lineage_pid = own_pid
    while lineage_pid > 0 and lineage_pid not in own_lineage:
        own_lineage.add(lineage_pid)
        parent = processes.get(lineage_pid)
        if parent is None:
            break
        lineage_pid = parent[0]

    conflicts: list[int] = []
    patterns = ("llama-server", "cmake --build", "/make -f Makefile")
    for pid, (_, command) in processes.items():
        if pid not in own_lineage and any(pattern in command for pattern in patterns):
            conflicts.append(pid)
    if conflicts:
        raise BenchmarkContractError(
            f"competing local compute is active (pids={sorted(set(conflicts))})"
        )


@dataclass(frozen=True)
class LoadedInputs:
    manifest: dict[str, Any]
    workload: list[dict[str, Any]]
    manifest_bytes: bytes
    workload_bytes: bytes
    manifest_sha256: str
    workload_sha256: str
    model_sha256: str
    server_sha256: str


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BenchmarkContractError(
            f"{label} keys differ from schema (missing={missing}, extra={extra})"
        )


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label} must be a JSON object")
    return value


def _string(value: object, label: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise BenchmarkContractError(f"{label} must be a non-empty string")
    return value


def _number(value: object, label: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkContractError(f"{label} must be a number")
    result = float(value)
    if result < minimum:
        raise BenchmarkContractError(f"{label} must be at least {minimum}")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkContractError(f"{label} must be an integer >= {minimum}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _capture_canonical_json_sha256(value: object) -> str:
    """Match the newline-terminated canonical form used by the capture sealer."""

    return _sha256_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkContractError(f"cannot read {label} JSON: {path}") from error
    return _object(value, label), raw


def _validate_sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkContractError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _policy_binding(manifest: Mapping[str, Any]) -> dict[str, str | None]:
    """Return the immutable policy identity, or the explicit legacy absence.

    The initial public benchmark artifacts predate policy-family studies.  They
    deliberately remain readable, but their lack of a policy binding is made
    explicit instead of being silently treated as a current-policy receipt.
    """

    if "policy_id" not in manifest:
        return {"policy_id": None, "policy_spec_sha256": None}
    return {
        "policy_id": manifest["policy_id"],
        "policy_spec_sha256": manifest["policy_spec_sha256"],
    }


def _bound_policy_spec(binding: Mapping[str, str | None]) -> dict[str, Any] | None:
    """Load the repository-owned immutable policy specification for a run.

    The identifier is already restricted to a canonical grammar, so deriving
    its filename cannot escape the repository.  The raw file digest is the
    frozen manifest binding and is copied unchanged into every raw receipt.
    """

    policy_id = binding["policy_id"]
    policy_sha256 = binding["policy_spec_sha256"]
    if policy_id is None or policy_sha256 is None:
        return None
    filename = policy_id.replace(".", "-").replace("/", "-") + ".json"
    relative_path = Path("qualification") / "policies" / filename
    path = Path(__file__).resolve().parents[2] / relative_path
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkContractError(
            f"cannot read canonical policy specification: {relative_path}"
        ) from error
    if _sha256_bytes(raw) != policy_sha256:
        raise BenchmarkContractError(
            "canonical policy specification SHA-256 does not match manifest"
        )
    if not isinstance(document, dict) or document.get("policy_id") != policy_id:
        raise BenchmarkContractError(
            "canonical policy specification identity does not match manifest"
        )
    return {
        "relative_path": str(relative_path),
        "sha256": policy_sha256,
        "bytes": raw,
    }


def _validate_policy_binding(document: Mapping[str, Any]) -> dict[str, str | None]:
    present = _POLICY_BINDING_KEYS & set(document)
    if not present:
        return {"policy_id": None, "policy_spec_sha256": None}
    if present != _POLICY_BINDING_KEYS:
        raise BenchmarkContractError(
            "manifest.policy_id and manifest.policy_spec_sha256 must be declared together"
        )
    policy_id = _string(document["policy_id"], "manifest.policy_id")
    if _CANONICAL_POLICY_ID.fullmatch(policy_id) is None:
        raise BenchmarkContractError(
            "manifest.policy_id must be a canonical non-legacy policy identifier"
        )
    return {
        "policy_id": policy_id,
        "policy_spec_sha256": _validate_sha256(
            document["policy_spec_sha256"],
            "manifest.policy_spec_sha256",
        ),
    }


def _validate_revision(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise BenchmarkContractError(f"{label} must be a full lowercase 40-character revision")
    return text


def _string_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise BenchmarkContractError(f"{label} must be {qualifier} of strings")
    if not all(isinstance(item, str) and item for item in value):
        raise BenchmarkContractError(f"{label} must contain only non-empty strings")
    if len(set(value)) != len(value):
        raise BenchmarkContractError(f"{label} must not contain duplicates")
    return value


def _validate_no_future_fields(value: object, path: str = "body") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BenchmarkContractError(f"{path} contains a non-string key")
            if key.lower() in _FUTURE_KEYS:
                raise BenchmarkContractError(
                    f"{path}.{key} is forbidden future/output material"
                )
            _validate_no_future_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_future_fields(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise BenchmarkContractError(f"{path} is not JSON-compatible")


def _assert_no_embedded_current_source(value: object, path: str) -> None:
    """Reject a current-source envelope outside its one supported request slot."""

    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() == CURRENT_SOURCE_REQUEST_KEY:
                raise BenchmarkContractError(
                    f"{path} contains current-source material outside the request body slot"
                )
            _assert_no_embedded_current_source(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_embedded_current_source(child, f"{path}[{index}]")


def _validate_current_source(value: object, label: str) -> dict[str, Any]:
    """Validate source input while retaining it only in the in-memory workload."""

    source = _object(value, label)
    _exact_keys(source, _CURRENT_SOURCE_KEYS, label)
    if source["schema_version"] != CURRENT_SOURCE_SCHEMA:
        raise BenchmarkContractError(
            f"{label}.schema_version must equal {CURRENT_SOURCE_SCHEMA}"
        )
    _string(source["compatibility_fingerprint"], f"{label}.compatibility_fingerprint")
    source_id = _string(source["source_id"], f"{label}.source_id")
    if (
        len(source_id) != 71
        or not source_id.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in source_id[7:])
    ):
        raise BenchmarkContractError(f"{label}.source_id must be a SHA-256 identifier")
    tokens = source["visible_token_ids"]
    if not isinstance(tokens, list) or len(tokens) > CURRENT_SOURCE_MAX_TOKENS:
        raise BenchmarkContractError(
            f"{label}.visible_token_ids must contain at most {CURRENT_SOURCE_MAX_TOKENS} token IDs"
        )
    for position, token in enumerate(tokens):
        if (
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token > CURRENT_SOURCE_MAX_TOKEN_ID
        ):
            raise BenchmarkContractError(
                f"{label}.visible_token_ids[{position}] must be a non-negative int32 token ID"
            )
    return source


def _current_source_from_body(
    body: Mapping[str, Any], label: str
) -> dict[str, Any] | None:
    matching_keys = [
        key
        for key in body
        if isinstance(key, str) and key.casefold() == CURRENT_SOURCE_REQUEST_KEY
    ]
    if not matching_keys:
        _assert_no_embedded_current_source(body, label)
        return None
    if matching_keys != [CURRENT_SOURCE_REQUEST_KEY]:
        raise BenchmarkContractError(
            f"{label} must use exactly one lowercase {CURRENT_SOURCE_REQUEST_KEY} field"
        )
    for key, child in body.items():
        if key != CURRENT_SOURCE_REQUEST_KEY:
            _assert_no_embedded_current_source(child, f"{label}.{key}")
    return _validate_current_source(
        body[CURRENT_SOURCE_REQUEST_KEY], f"{label}.{CURRENT_SOURCE_REQUEST_KEY}"
    )


def _current_source_attachment_mode(
    source: Mapping[str, Any], pcss_fingerprint: str
) -> str:
    return (
        CURRENT_SOURCE_MATCHING_MODE
        if source["compatibility_fingerprint"] == pcss_fingerprint
        else CURRENT_SOURCE_MISMATCH_CONTROL_MODE
    )


def _current_source_commitment(
    source: Mapping[str, Any], pcss_fingerprint: str
) -> dict[str, Any]:
    """Return the non-reversible evidence retained outside the live request."""

    return {
        "schema_version": CURRENT_SOURCE_COMMITMENT_SCHEMA,
        "token_count": len(source["visible_token_ids"]),
        "tokens_sha256": _canonical_json_sha256(source["visible_token_ids"]),
        "attachment_mode": _current_source_attachment_mode(source, pcss_fingerprint),
    }


def _capture_source_commitment(source: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the private-sidecar commitment without retaining the source."""

    return {
        "schema_version": CAPTURE_SOURCE_COMMITMENT_SCHEMA,
        "source_envelope_sha256": _capture_canonical_json_sha256(source),
        "token_count": len(source["visible_token_ids"]),
        "tokens_sha256": _canonical_json_sha256(source["visible_token_ids"]),
    }


def _validate_capture_source_commitment(value: object, label: str) -> dict[str, Any]:
    commitment = _object(value, label)
    _exact_keys(commitment, _CAPTURE_SOURCE_COMMITMENT_KEYS, label)
    if commitment["schema_version"] != CAPTURE_SOURCE_COMMITMENT_SCHEMA:
        raise BenchmarkContractError("unsupported capture source commitment schema")
    _validate_sha256(commitment["source_envelope_sha256"], f"{label}.source_envelope_sha256")
    _integer(commitment["token_count"], f"{label}.token_count", minimum=1)
    _validate_sha256(commitment["tokens_sha256"], f"{label}.tokens_sha256")
    return dict(commitment)


def _redact_current_source_body(
    body: Mapping[str, Any], label: str, pcss_fingerprint: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = _current_source_from_body(body, label)
    if source is None:
        return dict(body), None
    return (
        {
            key: value
            for key, value in body.items()
            if key != CURRENT_SOURCE_REQUEST_KEY
        },
        _current_source_commitment(source, pcss_fingerprint),
    )


def _redacted_current_source_workload(
    workload: Sequence[Mapping[str, Any]],
    original_workload_sha256: str,
    pcss_fingerprint: str,
) -> dict[str, Any] | None:
    """Create the source-safe archival workload receipt, if source was supplied."""

    requests: list[dict[str, Any]] = []
    source_request_count = 0
    for position, row in enumerate(workload):
        body, commitment = _redact_current_source_body(
            _object(row["body"], f"workload.requests[{position}].body"),
            f"workload.requests[{position}].body",
            pcss_fingerprint,
        )
        receipt_row = {
            "request_id": row["request_id"],
            "chronological_index": row["chronological_index"],
            "endpoint": row["endpoint"],
            "body": body,
        }
        if commitment is not None:
            receipt_row["pcss_current_source_commitment"] = commitment
            source_request_count += 1
        requests.append(receipt_row)
    if source_request_count == 0:
        return None
    return {
        "schema_version": REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
        "original_workload_sha256": original_workload_sha256,
        "requests": requests,
    }


def _validate_current_source_commitment(value: object, label: str) -> dict[str, Any]:
    commitment = _object(value, label)
    _exact_keys(commitment, _CURRENT_SOURCE_COMMITMENT_KEYS, label)
    if commitment["schema_version"] != CURRENT_SOURCE_COMMITMENT_SCHEMA:
        raise BenchmarkContractError("unsupported current-source commitment schema")
    _integer(commitment["token_count"], f"{label}.token_count")
    _validate_sha256(commitment["tokens_sha256"], f"{label}.tokens_sha256")
    if commitment["attachment_mode"] not in {
        CURRENT_SOURCE_MATCHING_MODE,
        CURRENT_SOURCE_MISMATCH_CONTROL_MODE,
    }:
        raise BenchmarkContractError("unsupported current-source attachment mode")
    return commitment


def _validate_workload(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    _exact_keys(document, _WORKLOAD_KEYS, "workload")
    if document["schema_version"] != WORKLOAD_SCHEMA:
        raise BenchmarkContractError("unsupported workload schema_version")
    requests = document["requests"]
    if not isinstance(requests, list) or not requests:
        raise BenchmarkContractError("workload.requests must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_index: int | None = None
    for position, raw in enumerate(requests):
        row = _object(raw, f"workload.requests[{position}]")
        _exact_keys(row, _REQUEST_KEYS, f"workload.requests[{position}]")
        request_id = _string(row["request_id"], f"request[{position}].request_id")
        if request_id in seen_ids:
            raise BenchmarkContractError(f"duplicate request_id: {request_id}")
        seen_ids.add(request_id)
        chronological_index = _integer(
            row["chronological_index"],
            f"request[{position}].chronological_index",
        )
        if previous_index is not None and chronological_index <= previous_index:
            raise BenchmarkContractError(
                "workload requests must be in strictly increasing chronological order"
            )
        previous_index = chronological_index
        endpoint = _string(row["endpoint"], f"request[{position}].endpoint")
        if endpoint not in {"/v1/chat/completions", "/v1/completions"}:
            raise BenchmarkContractError(
                "request endpoint must be /v1/chat/completions or /v1/completions"
            )
        body = _object(row["body"], f"request[{position}].body")
        _validate_no_future_fields(body)
        _current_source_from_body(body, f"request[{position}].body")
        if endpoint == "/v1/chat/completions":
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise BenchmarkContractError("chat requests require non-empty messages")
        elif not isinstance(body.get("prompt"), (str, list)):
            raise BenchmarkContractError("completion requests require a string/list prompt")
        normalized.append(dict(row))
    return normalized


def _validate_redacted_current_source_workload(
    document: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Validate an archival receipt without accepting any raw source material."""

    _exact_keys(document, _REDACTED_CURRENT_SOURCE_WORKLOAD_KEYS, "redacted workload")
    if document["schema_version"] != REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA:
        raise BenchmarkContractError("unsupported redacted current-source workload schema")
    original_workload_sha256 = _validate_sha256(
        document["original_workload_sha256"],
        "redacted workload.original_workload_sha256",
    )
    raw_requests = document["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        raise BenchmarkContractError("redacted workload.requests must be a non-empty array")
    ordinary_requests: list[dict[str, Any]] = []
    commitments: list[dict[str, Any] | None] = []
    for position, raw in enumerate(raw_requests):
        row = _object(raw, f"redacted workload.requests[{position}]")
        if (
            set(row) != _REQUEST_KEYS
            and set(row) != _REDACTED_CURRENT_SOURCE_REQUEST_KEYS
        ):
            raise BenchmarkContractError(
                f"redacted workload.requests[{position}] has an unsupported receipt shape"
            )
        body = _object(row["body"], f"redacted workload.requests[{position}].body")
        _assert_no_embedded_current_source(
            body, f"redacted workload.requests[{position}].body"
        )
        ordinary_requests.append({key: row[key] for key in _REQUEST_KEYS})
        commitments.append(
            _validate_current_source_commitment(
                row["pcss_current_source_commitment"],
                f"redacted workload.requests[{position}].pcss_current_source_commitment",
            )
            if "pcss_current_source_commitment" in row
            else None
        )
    if not any(commitment is not None for commitment in commitments):
        raise BenchmarkContractError(
            "redacted current-source workload must contain a source commitment"
        )
    normalized = _validate_workload(
        {"schema_version": WORKLOAD_SCHEMA, "requests": ordinary_requests}
    )
    for row, commitment in zip(normalized, commitments, strict=True):
        if commitment is not None:
            row["pcss_current_source_commitment"] = commitment
    return normalized, original_workload_sha256


def _validate_manifest(
    document: Mapping[str, Any],
    *,
    workload_sha256: str,
    workload_request_ids: Sequence[str],
    requires_current_source_activation: bool = False,
    current_source_request_count: int = 0,
) -> dict[str, Any]:
    policy_binding = _validate_policy_binding(document)
    expected_manifest_keys = (
        _MANIFEST_KEYS | _POLICY_BINDING_KEYS
        if policy_binding["policy_id"] is not None
        else _MANIFEST_KEYS
    )
    _exact_keys(document, expected_manifest_keys, "manifest")
    if document["schema_version"] != MANIFEST_SCHEMA:
        raise BenchmarkContractError("unsupported manifest schema_version")
    _string(document["study_id"], "manifest.study_id")
    if document["frozen"] is not True:
        raise BenchmarkContractError("manifest must explicitly set frozen=true")
    _validate_revision(document["git_revision"], "manifest.git_revision")
    _validate_sha256(document["patch_sha256"], "manifest.patch_sha256")
    _validate_revision(document["llama_cpp_revision"], "manifest.llama_cpp_revision")
    _validate_sha256(document["runtime_fingerprint"], "manifest.runtime_fingerprint")
    hardware_os = _object(document["hardware_os"], "manifest.hardware_os")
    _exact_keys(hardware_os, _HARDWARE_OS_KEYS, "manifest.hardware_os")
    _string(hardware_os["hardware"], "hardware_os.hardware")
    _string(hardware_os["os"], "hardware_os.os")
    if _validate_sha256(document["workload_sha256"], "manifest.workload_sha256") != workload_sha256:
        raise BenchmarkContractError("workload SHA-256 does not match frozen manifest")
    workload_class = document["workload_class"]
    if workload_class not in WORKLOAD_CLASSES:
        raise BenchmarkContractError("manifest.workload_class is unsupported")
    holdout = document["holdout"]
    if not isinstance(holdout, bool):
        raise BenchmarkContractError("manifest.holdout must be boolean")
    if holdout != (workload_class == "owner_representative_holdout"):
        raise BenchmarkContractError(
            "holdout=true is reserved for owner_representative_holdout"
        )
    provenance = _object(document["workload_provenance"], "manifest.workload_provenance")
    source_capture_required = (
        requires_current_source_activation
        and workload_class in {"agent_assisted_calibration", "owner_representative_holdout"}
    )
    expected_provenance_keys = (
        _CURRENT_SOURCE_CAPTURE_PROVENANCE_KEYS
        if source_capture_required
        else (
            _AGENT_ASSISTED_PROVENANCE_KEYS
            if workload_class == "agent_assisted_calibration"
            else (
                _OWNER_REPRESENTATIVE_PROVENANCE_KEYS
                if workload_class == "owner_representative_holdout"
                else _PROVENANCE_KEYS
            )
        )
    )
    _exact_keys(provenance, expected_provenance_keys, "manifest.workload_provenance")
    _string(provenance["source"], "workload_provenance.source")
    _string(provenance["capture_protocol"], "workload_provenance.capture_protocol")
    if provenance["captured_before_arm_results"] is not True or provenance["immutable"] is not True:
        raise BenchmarkContractError(
            "workload provenance must be immutable and captured before arm results"
        )
    if source_capture_required or workload_class in {
        "agent_assisted_calibration",
        "owner_representative_holdout",
    }:
        expected_origin = (
            "agent_assisted_local_llama_server"
            if workload_class == "agent_assisted_calibration"
            else "ordinary_local_llama_server"
        )
        if provenance["origin"] != expected_origin:
            raise BenchmarkContractError(
                "captured workload must declare the required local origin"
            )
        _validate_sha256(
            provenance["capture_receipt_sha256"],
            "workload_provenance.capture_receipt_sha256",
        )
        _validate_sha256(
            provenance["source_event_sha256"],
            "workload_provenance.source_event_sha256",
        )
    if source_capture_required:
        _validate_sha256(
            provenance["source_sidecar_sha256"],
            "workload_provenance.source_sidecar_sha256",
        )
        attachment_count = _integer(
            provenance["source_attachment_count"],
            "workload_provenance.source_attachment_count",
            minimum=1,
        )
        if attachment_count != current_source_request_count:
            raise BenchmarkContractError(
                "workload provenance source_attachment_count does not bind the frozen workload"
            )
    _string_list(document["eligibility_rules"], "manifest.eligibility_rules")
    calibration = _object(document["calibration_split"], "manifest.calibration_split")
    _exact_keys(calibration, _CALIBRATION_SPLIT_KEYS, "manifest.calibration_split")
    calibration_ids = _string_list(
        calibration["request_ids"],
        "calibration_split.request_ids",
        allow_empty=True,
    )
    if not isinstance(calibration["tuning_permitted"], bool):
        raise BenchmarkContractError("calibration_split.tuning_permitted must be boolean")
    holdout_split = _object(document["holdout_split"], "manifest.holdout_split")
    _exact_keys(holdout_split, _HOLDOUT_SPLIT_KEYS, "manifest.holdout_split")
    holdout_ids = _string_list(
        holdout_split["request_ids"],
        "holdout_split.request_ids",
        allow_empty=True,
    )
    if not isinstance(holdout_split["one_shot"], bool):
        raise BenchmarkContractError("holdout_split.one_shot must be boolean")
    workload_ids = list(workload_request_ids)
    if set(calibration_ids) & set(holdout_ids):
        raise BenchmarkContractError("calibration and holdout splits must be disjoint")
    if calibration_ids + holdout_ids != workload_ids:
        raise BenchmarkContractError(
            "calibration_split followed by holdout_split must preserve and cover workload order"
        )
    if holdout:
        if calibration_ids or holdout_ids != workload_ids or holdout_split["one_shot"] is not True:
            raise BenchmarkContractError(
                "owner representative holdout must contain the full workload and be one-shot"
            )
        if calibration["tuning_permitted"] is not False:
            raise BenchmarkContractError("owner representative holdout cannot permit tuning")
    elif holdout_ids or holdout_split["one_shot"] is not False:
        raise BenchmarkContractError(
            "synthetic/development studies cannot declare a holdout split or one-shot opening"
        )
    if workload_class == "agent_assisted_calibration" and calibration["tuning_permitted"] is not False:
        raise BenchmarkContractError(
            "agent-assisted calibration must freeze its sealed workload before arm results"
        )
    clustering = _object(document["clustering"], "manifest.clustering")
    _exact_keys(clustering, _CLUSTERING_KEYS, "manifest.clustering")
    _string(clustering["method"], "clustering.method")
    cluster_ids = _object(clustering["request_cluster_ids"], "clustering.request_cluster_ids")
    if set(cluster_ids) != set(workload_ids):
        raise BenchmarkContractError("clustering must bind every workload request exactly once")
    for request_id, cluster_id in cluster_ids.items():
        _string(cluster_id, f"clustering.request_cluster_ids.{request_id}")
    _validate_sha256(document["model_sha256"], "manifest.model_sha256")
    _validate_sha256(document["server_sha256"], "manifest.server_sha256")
    server_args = document["server_args"]
    if not isinstance(server_args, list) or not all(
        isinstance(argument, str) and argument for argument in server_args
    ):
        raise BenchmarkContractError("manifest.server_args must be an array of strings")
    for argument in server_args:
        option = argument.split("=", 1)[0]
        if option in _FORBIDDEN_SERVER_ARGS:
            raise BenchmarkContractError(
                f"manifest.server_args cannot override harness-owned option {option}"
            )
    request_defaults = _object(document["request_defaults"], "manifest.request_defaults")
    _validate_no_future_fields(request_defaults, "manifest.request_defaults")
    _assert_no_embedded_current_source(
        request_defaults, "manifest.request_defaults"
    )
    if request_defaults.get("stream") not in {True, False, None}:
        raise BenchmarkContractError("request_defaults.stream must be boolean when present")
    arms = document["arms"]
    if not isinstance(arms, list) or len(arms) != len(ARM_NAMES):
        raise BenchmarkContractError("manifest.arms must define exactly three arms")
    arm_names: list[str] = []
    for index, raw_arm in enumerate(arms):
        arm = _object(raw_arm, f"manifest.arms[{index}]")
        name = arm.get("name")
        arm_names.append(name if isinstance(name, str) else "")
        if name == "pcss-cache":
            _exact_keys(arm, {"name", "pcss"}, f"manifest.arms[{index}]")
            pcss = _object(arm["pcss"], "pcss arm options")
            required_pcss_keys = {"fingerprint", "namespace", "max_disk_gib", "ram_mib", "read_only", "disable_learning"}
            optional_pcss_keys = {
                "disable_current_context",
                "enable_current_source",
                "max_draft_tokens",
            }
            if not required_pcss_keys <= set(pcss) <= required_pcss_keys | optional_pcss_keys:
                raise BenchmarkContractError("pcss arm options contain unsupported or missing fields")
            _string(pcss["fingerprint"], "pcss.fingerprint")
            _string(pcss["namespace"], "pcss.namespace")
            _integer(pcss["max_disk_gib"], "pcss.max_disk_gib", minimum=1)
            _integer(pcss["ram_mib"], "pcss.ram_mib", minimum=1)
            if (
                not isinstance(pcss["read_only"], bool)
                or not isinstance(pcss["disable_learning"], bool)
                or not isinstance(pcss.get("disable_current_context", False), bool)
                or not isinstance(pcss.get("enable_current_source", False), bool)
            ):
                raise BenchmarkContractError("pcss boolean options are invalid")
            _integer(pcss.get("max_draft_tokens", 2), "pcss.max_draft_tokens", minimum=1)
            if pcss.get("max_draft_tokens", 2) > 2:
                raise BenchmarkContractError("pcss.max_draft_tokens must not exceed the proven-safe cap of 2")
        else:
            _exact_keys(arm, {"name"}, f"manifest.arms[{index}]")
    if tuple(arm_names) != ARM_NAMES:
        raise BenchmarkContractError(
            f"manifest.arms must be declared exactly as {list(ARM_NAMES)}"
        )
    arm_order = document["arm_order"]
    if not isinstance(arm_order, list) or sorted(arm_order) != sorted(ARM_NAMES):
        raise BenchmarkContractError("arm_order must be one permutation of all arms")
    counterbalance = _object(document["counterbalance"], "manifest.counterbalance")
    _exact_keys(counterbalance, _COUNTERBALANCE_KEYS, "manifest.counterbalance")
    if counterbalance["method"] not in {
        "latin_square",
        "randomized_preregistered",
        "single_preregistered_order",
    }:
        raise BenchmarkContractError("unsupported counterbalance method")
    _string(counterbalance["sequence_id"], "counterbalance.sequence_id")
    state_definitions = _object(document["state_definitions"], "manifest.state_definitions")
    _exact_keys(state_definitions, _STATE_DEFINITION_KEYS, "manifest.state_definitions")
    _string(state_definitions["warm"], "state_definitions.warm")
    _string(state_definitions["cold"], "state_definitions.cold")
    initialization = _object(
        document["arm_store_initialization"],
        "manifest.arm_store_initialization",
    )
    if set(initialization) != set(ARM_NAMES):
        raise BenchmarkContractError("arm_store_initialization must bind every arm exactly once")
    for arm_name in ARM_NAMES:
        arm_initialization = _object(
            initialization[arm_name],
            f"arm_store_initialization.{arm_name}",
        )
        _exact_keys(
            arm_initialization,
            _STORE_INITIALIZATION_KEYS,
            f"arm_store_initialization.{arm_name}",
        )
        if arm_initialization["state"] not in {"warm", "cold"}:
            raise BenchmarkContractError("arm store state must be warm or cold")
        _string(
            arm_initialization["initialization"],
            f"arm_store_initialization.{arm_name}.initialization",
        )
        snapshot = arm_initialization["snapshot_sha256"]
        if snapshot is not None:
            _validate_sha256(
                snapshot,
                f"arm_store_initialization.{arm_name}.snapshot_sha256",
            )
        if arm_initialization["state"] == "warm" and snapshot is None:
            raise BenchmarkContractError("warm arm initialization requires a snapshot SHA-256")
        if arm_initialization["state"] != "cold" or snapshot is not None:
            raise BenchmarkContractError(
                "live benchmark v1 currently supports only verifiable cold arm initialization"
            )
    readiness = _object(document["readiness"], "manifest.readiness")
    _exact_keys(readiness, _READINESS_KEYS, "manifest.readiness")
    path = _string(readiness["path"], "readiness.path")
    if not path.startswith("/"):
        raise BenchmarkContractError("readiness.path must be an absolute HTTP path")
    _number(readiness["timeout_seconds"], "readiness.timeout_seconds", minimum=0.001)
    _integer(readiness["max_attempts"], "readiness.max_attempts", minimum=1)
    _number(readiness["interval_seconds"], "readiness.interval_seconds", minimum=0)
    _number(
        document["request_timeout_seconds"],
        "manifest.request_timeout_seconds",
        minimum=0.001,
    )
    if _integer(document["request_count"], "manifest.request_count", minimum=1) != len(workload_ids):
        raise BenchmarkContractError("manifest.request_count differs from workload")
    gates = _object(document["effect_gates"], "manifest.effect_gates")
    allowed_effect_keys = (
        _EFFECT_GATE_KEYS
        | {"exact_output_arms"}
        | _LADDER_ACTIVATION_GATE_KEYS
        | _CURRENT_SOURCE_ACTIVATION_GATE_KEYS
    )
    if not _EFFECT_GATE_KEYS <= set(gates) <= allowed_effect_keys:
        raise BenchmarkContractError("effect_gates contains unsupported or missing fields")
    if gates["primary_metric"] != "paired_end_to_end_latency":
        raise BenchmarkContractError("primary_metric must be paired_end_to_end_latency")
    if gates["ratio_direction"] != "baseline_over_arm_gt_1_is_faster":
        raise BenchmarkContractError("effect gate ratio direction is invalid")
    _number(
        gates["baseline_over_pcss_lcb95_min"],
        "effect_gates.baseline_over_pcss_lcb95_min",
        minimum=1,
    )
    miss_overhead = _number(
        gates["miss_heavy_p95_overhead_ucb95_max"],
        "effect_gates.miss_heavy_p95_overhead_ucb95_max",
    )
    if miss_overhead > 0.03:
        raise BenchmarkContractError("miss-heavy p95 overhead UCB95 gate cannot exceed 3%")
    _string(gates["confidence_method"], "effect_gates.confidence_method")
    if gates["exact_output_required"] is not True:
        raise BenchmarkContractError("exact_output_required must be true")
    required_exact_arms = gates.get("exact_output_arms", list(ARM_NAMES[1:]))
    if not isinstance(required_exact_arms, list) or not required_exact_arms or any(
        arm not in ARM_NAMES[1:] for arm in required_exact_arms
    ) or len(set(required_exact_arms)) != len(required_exact_arms):
        raise BenchmarkContractError("effect_gates.exact_output_arms must be a non-empty unique subset of speculative arms")
    if workload_class == "agent_assisted_calibration":
        if not _LADDER_ACTIVATION_GATE_KEYS <= set(gates):
            raise BenchmarkContractError(
                "agent-assisted calibration requires suffix-ladder activation gates"
            )
        _integer(
            gates["persistent_32_hit_requests_min"],
            "effect_gates.persistent_32_hit_requests_min",
            minimum=1,
        )
        _integer(
            gates["persistent_32_hit_episodes_min"],
            "effect_gates.persistent_32_hit_episodes_min",
            minimum=1,
        )
        _integer(
            gates["persistent_32_accepted_tokens_min"],
            "effect_gates.persistent_32_accepted_tokens_min",
            minimum=1,
        )
        _integer(
            gates["persistent_64_hit_requests_max"],
            "effect_gates.persistent_64_hit_requests_max",
        )
        _integer(
            gates["persistent_32_suppressions_max"],
            "effect_gates.persistent_32_suppressions_max",
        )
        _integer(
            gates["fallback_disable_events_max"],
            "effect_gates.fallback_disable_events_max",
        )
    if requires_current_source_activation:
        if not _CURRENT_SOURCE_ACTIVATION_GATE_KEYS <= set(gates):
            raise BenchmarkContractError(
                "source-bearing workload requires current-source activation and mismatch-control gates"
            )
        minimum_one = (
            "current_source_matching_requests_min",
            "current_source_lookup_requests_min",
            "current_source_hit_requests_min",
            "current_source_hit_episodes_min",
            "current_source_accepted_tokens_min",
            "current_source_peak_index_bytes_min",
            "current_source_mismatch_control_requests_min",
            "current_source_mismatch_abstentions_min",
        )
        for key in minimum_one:
            _integer(gates[key], f"effect_gates.{key}", minimum=1)
        for key in (
            "current_source_rejections_max",
            "current_source_ambiguities_max",
            "current_source_abstentions_max",
        ):
            _integer(gates[key], f"effect_gates.{key}")
    elif _CURRENT_SOURCE_ACTIVATION_GATE_KEYS & set(gates):
        raise BenchmarkContractError(
            "current-source activation gates require a source-bearing workload"
        )
    ceilings = _object(document["resource_ceilings"], "manifest.resource_ceilings")
    _exact_keys(ceilings, _RESOURCE_CEILING_KEYS, "manifest.resource_ceilings")
    _integer(ceilings["ram_mib"], "resource_ceilings.ram_mib", minimum=1)
    _number(ceilings["disk_gib"], "resource_ceilings.disk_gib", minimum=0.001)
    _number(ceilings["startup_seconds"], "resource_ceilings.startup_seconds", minimum=0.001)
    _number(
        ceilings["compaction_write_amplification_max"],
        "resource_ceilings.compaction_write_amplification_max",
        minimum=1,
    )
    _string_list(
        document["allowed_policy_families"],
        "manifest.allowed_policy_families",
    )
    grids = _object(document["finite_parameter_grids"], "manifest.finite_parameter_grids")
    if not grids:
        raise BenchmarkContractError("finite_parameter_grids must not be empty")
    for grid_name, values in grids.items():
        _string(grid_name, "finite_parameter_grids key")
        if not isinstance(values, list) or not values or len(values) > 1000:
            raise BenchmarkContractError(
                f"finite_parameter_grids.{grid_name} must have 1..1000 values"
            )
        serialized: list[str] = []
        for value in values:
            if value is None or isinstance(value, (str, int, float, bool)):
                serialized.append(json.dumps(value, sort_keys=True))
            else:
                raise BenchmarkContractError(
                    f"finite_parameter_grids.{grid_name} values must be JSON scalars"
                )
        if len(set(serialized)) != len(serialized):
            raise BenchmarkContractError(
                f"finite_parameter_grids.{grid_name} must not contain duplicates"
            )
    pcss_arm = _arm_by_name(document, "pcss-cache")
    pcss_options = _object(pcss_arm["pcss"], "pcss arm options")
    effective_draft_tokens = pcss_options.get("max_draft_tokens", 2)
    declared_drafts = grids.get("draft_tokens")
    if not isinstance(declared_drafts, list) or effective_draft_tokens not in declared_drafts:
        raise BenchmarkContractError(
            "effective pcss.max_draft_tokens must be declared in finite_parameter_grids.draft_tokens"
        )
    budgets = _object(document["budgets"], "manifest.budgets")
    _exact_keys(budgets, _BUDGET_KEYS, "manifest.budgets")
    max_requests = _integer(budgets["max_requests"], "budgets.max_requests", minimum=1)
    if max_requests < len(workload_ids) * len(ARM_NAMES) or max_requests > 10_000:
        raise BenchmarkContractError(
            "max_requests must cover all arm requests and cannot exceed 10000"
        )
    _number(budgets["max_wall_clock_seconds"], "budgets.max_wall_clock_seconds", minimum=0.001)
    _integer(budgets["max_artifact_bytes"], "budgets.max_artifact_bytes", minimum=1)
    _string(document["stopping_rule"], "manifest.stopping_rule")
    _string_list(document["kill_gates"], "manifest.kill_gates")
    _string_list(document["exclusion_rules"], "manifest.exclusion_rules")
    _string(document["claim_boundary"], "manifest.claim_boundary")
    return dict(document)


def load_inputs(
    manifest_path: Path,
    workload_path: Path,
    model_path: Path,
    server_path: Path,
) -> LoadedInputs:
    for path, label in ((model_path, "model"), (server_path, "server")):
        if not path.is_file():
            raise BenchmarkContractError(f"{label} is not a file: {path}")
    if not os.access(server_path, os.X_OK):
        raise BenchmarkContractError(f"server is not executable: {server_path}")
    workload_document, workload_bytes = _read_json(workload_path, "workload")
    workload = _validate_workload(workload_document)
    source_bearing = _workload_has_current_source(workload)
    workload_sha256 = _sha256_bytes(workload_bytes)
    manifest_document, manifest_bytes = _read_json(manifest_path, "manifest")
    manifest = _validate_manifest(
        manifest_document,
        workload_sha256=workload_sha256,
        workload_request_ids=[row["request_id"] for row in workload],
        requires_current_source_activation=source_bearing,
        current_source_request_count=sum(
            CURRENT_SOURCE_REQUEST_KEY in row["body"] for row in workload
        ),
    )
    _validate_source_workload_compatibility(workload, manifest)
    model_sha256 = _file_sha256(model_path)
    server_sha256 = _file_sha256(server_path)
    if model_sha256 != manifest["model_sha256"]:
        raise BenchmarkContractError("model SHA-256 does not match frozen manifest")
    if server_sha256 != manifest["server_sha256"]:
        raise BenchmarkContractError("server SHA-256 does not match frozen manifest")
    return LoadedInputs(
        manifest=manifest,
        workload=workload,
        manifest_bytes=manifest_bytes,
        workload_bytes=workload_bytes,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        workload_sha256=workload_sha256,
        model_sha256=model_sha256,
        server_sha256=server_sha256,
    )


def _holdout_marker_path(manifest_path: Path, inputs: LoadedInputs) -> Path:
    safe_study_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in inputs.manifest["study_id"]
    )
    return manifest_path.parent / (
        f".{safe_study_id}.{inputs.manifest_sha256}.owner-holdout-opened"
    )


def _claim_holdout_once(manifest_path: Path, inputs: LoadedInputs) -> Path | None:
    if not inputs.manifest["holdout"]:
        return None
    marker_path = _holdout_marker_path(manifest_path, inputs)
    payload = json.dumps(
        {
            "study_id": inputs.manifest["study_id"],
            "manifest_sha256": inputs.manifest_sha256,
            "one_shot": True,
        },
        sort_keys=True,
    ).encode() + b"\n"
    try:
        descriptor = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise BenchmarkContractError(
            f"owner representative holdout was already opened: {marker_path}"
        ) from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return marker_path


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _process_resource_snapshot(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return {"pid": pid, "rss_bytes": None, "cpu_time": None, "error": "invalid pid"}
    try:
        completed = subprocess.run(
            ("ps", "-o", "rss=,time=", "-p", str(pid)),
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "pid": pid,
            "rss_bytes": None,
            "cpu_time": None,
            "error": f"{type(error).__name__}: {error}",
        }
    fields = completed.stdout.strip().split()
    if completed.returncode != 0 or len(fields) != 2:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        return {
            "pid": pid,
            "rss_bytes": None,
            "cpu_time": None,
            "error": f"ps failed ({completed.returncode}): {diagnostic}",
        }
    try:
        rss_bytes = int(fields[0]) * 1024
    except ValueError:
        rss_bytes = None
    return {
        "pid": pid,
        "rss_bytes": rss_bytes,
        "cpu_time": fields[1],
        "error": None if rss_bytes is not None else "invalid ps RSS",
    }


def _directory_usage(path: Path, *, max_entries: int = 100_000) -> dict[str, Any]:
    total_bytes = 0
    entries = 0
    truncated = False
    error: str | None = None
    try:
        for root, directories, files in os.walk(path):
            for name in directories + files:
                entries += 1
                if entries > max_entries:
                    truncated = True
                    break
                candidate = Path(root) / name
                if candidate.is_file():
                    total_bytes += candidate.stat().st_size
            if truncated:
                break
    except OSError as caught:
        error = f"{type(caught).__name__}: {caught}"
    return {
        "path": str(path),
        "bytes": total_bytes,
        "entries": entries,
        "max_entries": max_entries,
        "truncated": truncated,
        "error": error,
    }


def _tree_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


def _arm_by_name(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(arm for arm in manifest["arms"] if arm["name"] == name)


def _workload_has_current_source(workload: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        _current_source_from_body(
            _object(row["body"], f"workload.requests[{position}].body"),
            f"workload.requests[{position}].body",
        )
        is not None
        for position, row in enumerate(workload)
    )


def _current_source_catalog(
    workload: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        source
        for position, row in enumerate(workload)
        if (
            source := _current_source_from_body(
                _object(row["body"], f"workload.requests[{position}].body"),
                f"workload.requests[{position}].body",
            )
        )
        is not None
    )


def _validate_source_workload_compatibility(
    workload: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> None:
    """Require a source-bearing workload to activate the exact native tier."""

    sources = [
        _current_source_from_body(
            _object(row["body"], f"workload.requests[{position}].body"),
            f"workload.requests[{position}].body",
        )
        for position, row in enumerate(workload)
    ]
    if not any(source is not None for source in sources):
        return
    pcss = _arm_by_name(manifest, "pcss-cache")["pcss"]
    if pcss.get("enable_current_source") is not True:
        raise BenchmarkContractError(
            "source-bearing workload requires pcss.enable_current_source=true"
        )
    modes = [
        _current_source_attachment_mode(source, pcss["fingerprint"])
        for source in sources
        if source is not None
    ]
    if CURRENT_SOURCE_MATCHING_MODE not in modes:
        raise BenchmarkContractError(
            "source-bearing workload requires at least one matching current-source row"
        )
    if CURRENT_SOURCE_MISMATCH_CONTROL_MODE not in modes:
        raise BenchmarkContractError(
            "source-bearing workload requires a compatibility-mismatch abstention control row"
        )


def _server_command(
    server_path: Path,
    model_path: Path,
    manifest: Mapping[str, Any],
    arm_name: str,
    port: int,
    store_path: Path,
) -> list[str]:
    command = [
        str(server_path),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--metrics",
        *manifest["server_args"],
    ]
    if arm_name == "upstream-ngram-cache":
        command.extend(("--spec-type", "ngram-cache", "--spec-draft-n-max", str(NGRAM_MAX_DRAFT_TOKENS)))
    elif arm_name == "pcss-cache":
        pcss = _arm_by_name(manifest, arm_name)["pcss"]
        policy_binding = _policy_binding(manifest)
        command.extend(
            (
                "--spec-type",
                "pcss-cache",
                "--pcss-store",
                str(store_path),
                "--pcss-fingerprint",
                pcss["fingerprint"],
                "--pcss-namespace",
                pcss["namespace"],
                "--pcss-max-disk-gib",
                str(pcss["max_disk_gib"]),
                "--pcss-ram-mib",
                str(pcss["ram_mib"]),
            )
        )
        if pcss["read_only"]:
            command.append("--pcss-read-only")
        if pcss["disable_learning"]:
            command.append("--pcss-disable-learning")
        if pcss.get("disable_current_context", False):
            command.append("--pcss-disable-current-context")
        if pcss.get("enable_current_source", False):
            command.append("--pcss-enable-current-source")
        if "max_draft_tokens" in pcss:
            command.extend(("--pcss-max-draft-tokens", str(pcss["max_draft_tokens"])))
        if policy_binding["policy_id"] is not None:
            command.extend(
                (
                    "--pcss-policy-id",
                    policy_binding["policy_id"],
                    "--pcss-policy-spec-sha256",
                    policy_binding["policy_spec_sha256"],
                )
            )
    return command


def _wait_ready(
    process: subprocess.Popen[bytes],
    base_url: str,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    started_ns = time.monotonic_ns()
    deadline = time.monotonic() + float(readiness["timeout_seconds"])
    for attempt in range(1, int(readiness["max_attempts"]) + 1):
        returncode = process.poll()
        if returncode is not None:
            raise BenchmarkContractError(
                f"server exited before readiness with status {returncode}"
            )
        status: int | None = None
        error: str | None = None
        try:
            with urlopen(
                f"{base_url}{readiness['path']}",
                timeout=min(1.0, max(0.05, deadline - time.monotonic())),
            ) as response:
                status = response.status
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as caught:
            error = f"{type(caught).__name__}: {caught}"
        attempts.append({"attempt": attempt, "status": status, "error": error})
        if status == 200:
            return {
                "ready": True,
                "latency_ns": time.monotonic_ns() - started_ns,
                "attempts": attempts,
            }
        if attempt >= int(readiness["max_attempts"]) or time.monotonic() >= deadline:
            break
        time.sleep(min(float(readiness["interval_seconds"]), max(0, deadline - time.monotonic())))
    raise BenchmarkContractError("server readiness bounds exhausted")


def _fetch_metrics(base_url: str, timeout: float) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    try:
        with urlopen(f"{base_url}/metrics", timeout=timeout) as response:
            body = response.read()
            return {
                "status": response.status,
                "latency_ns": time.monotonic_ns() - started_ns,
                "content_type": response.headers.get("Content-Type"),
                "body": body.decode("utf-8", errors="replace"),
                "error": None,
            }
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        return {
            "status": error.code,
            "latency_ns": time.monotonic_ns() - started_ns,
            "content_type": error.headers.get("Content-Type"),
            "body": body,
            "error": f"HTTPError: {error}",
        }
    except (URLError, TimeoutError, OSError) as error:
        return {
            "status": None,
            "latency_ns": time.monotonic_ns() - started_ns,
            "content_type": None,
            "body": "",
            "error": f"{type(error).__name__}: {error}",
        }


def _chunk_text(event: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            pieces.append(text)
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            pieces.append(delta["content"])
    return "".join(pieces)


def _response_fields(document: Mapping[str, Any]) -> tuple[str, int | None, str | None, Any]:
    text = _chunk_text(document)
    choices = document.get("choices")
    finish_reason: str | None = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            text = message["content"]
        finish = choices[0].get("finish_reason")
        finish_reason = finish if isinstance(finish, str) else None
    usage = document.get("usage")
    output_tokens = None
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        output_tokens = usage["completion_tokens"]
    return text, output_tokens, finish_reason, document.get("timings")


def _dispatch_request(
    base_url: str,
    endpoint: str,
    body: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    started_ns = time.monotonic_ns()
    first_token_ns: int | None = None
    raw_body = b""
    events: list[Any] = []
    status: int | None = None
    error: str | None = None
    text = ""
    output_tokens: int | None = None
    finish_reason: str | None = None
    timings: Any = None
    try:
        request = Request(f"{base_url}{endpoint}", data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            if body.get("stream") is True:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    raw_body += line
                    stripped = line.strip()
                    if not stripped.startswith(b"data:"):
                        continue
                    data = stripped[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    events.append(event)
                    piece = _chunk_text(event) if isinstance(event, dict) else ""
                    if piece and first_token_ns is None:
                        first_token_ns = time.monotonic_ns()
                    text += piece
                    if isinstance(event, dict):
                        _, event_tokens, event_finish, event_timings = _response_fields(event)
                        if event_tokens is not None:
                            output_tokens = event_tokens
                        if event_finish is not None:
                            finish_reason = event_finish
                        if event_timings is not None:
                            timings = event_timings
            else:
                raw_body = response.read()
                document = _object(json.loads(raw_body), "OpenAI response")
                events.append(document)
                text, output_tokens, finish_reason, timings = _response_fields(document)
    except HTTPError as caught:
        status = caught.code
        raw_body = caught.read()
        error = f"HTTPError: {caught}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError, BenchmarkContractError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    ended_ns = time.monotonic_ns()
    return {
        "status": status,
        "latency_ns": ended_ns - started_ns,
        "ttft_ns": None if first_token_ns is None else first_token_ns - started_ns,
        "streaming": body.get("stream") is True,
        "output_text": text,
        "output_tokens": output_tokens,
        "finish_reason": finish_reason,
        "server_timings": timings,
        "events": events,
        "raw_body_utf8": raw_body.decode("utf-8", errors="replace"),
        "error": error,
    }


def _terminate_process(process: subprocess.Popen[bytes], timeout: float = 10.0) -> dict[str, Any]:
    pgid = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    action = "already_exited" if process.poll() is not None else "terminate_group"
    if group_exists():
        os.killpg(pgid, signal.SIGTERM)
    elif process.poll() is None:
        process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        action = "kill_group"
        if group_exists():
            os.killpg(pgid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)
    for _ in range(10):
        if not group_exists():
            break
        time.sleep(0.05)
    if group_exists():
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(0.05)
    return {
        "action": action,
        "returncode": process.returncode,
        "process_group": pgid,
        "descendants_absent": not group_exists(),
    }


def _prometheus_value(receipt: Mapping[str, Any] | None, name: str) -> float | None:
    if receipt is None or not isinstance(receipt.get("body"), str):
        return None
    prefix = f"llamacpp:{name} "
    for line in receipt["body"].splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :])
            except ValueError:
                return None
    return None


def _current_source_metric_attestation(
    receipt: Mapping[str, Any] | None, *, required: bool
) -> dict[str, Any]:
    """Fail closed when a source study is run against an older server metric set."""

    observed = {
        name: _prometheus_value(receipt, name) for name in _CURRENT_SOURCE_REQUIRED_METRICS
    }
    missing = [name for name, value in observed.items() if value is None]
    return {
        "required": required,
        "verified": not missing if required else False,
        "missing_metrics": missing if required else [],
    }


def _policy_metric_attestation(
    receipt: Mapping[str, Any] | None,
    binding: Mapping[str, str | None],
) -> dict[str, Any]:
    """Extract the server's live PCSS policy identity from a metrics receipt."""

    expected = {
        "policy_id": binding["policy_id"],
        "policy_spec_sha256": binding["policy_spec_sha256"],
    }
    if expected["policy_id"] is None:
        return {
            "required": False,
            "verified": False,
            "observed": None,
            "matching_lines": 0,
        }
    if (
        receipt is None
        or receipt.get("status") != 200
        or receipt.get("error") is not None
        or not isinstance(receipt.get("body"), str)
    ):
        return {
            "required": True,
            "verified": False,
            "observed": None,
            "matching_lines": 0,
        }
    observed: list[dict[str, str]] = []
    for line in receipt["body"].splitlines():
        match = _POLICY_INFO_LINE.fullmatch(line)
        if match is not None and float(match["value"]) == 1.0:
            observed.append(
                {
                    "policy_id": match["policy_id"],
                    "policy_spec_sha256": match["policy_spec_sha256"],
                }
            )
    return {
        "required": True,
        "verified": observed == [expected],
        "observed": observed[0] if len(observed) == 1 else None,
        "matching_lines": len(observed),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _receipt_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _redact_current_source_value(
    value: object, sources: Sequence[Mapping[str, Any]]
) -> object:
    """Defensively remove source echoes from a source-bearing response receipt."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _CURRENT_SOURCE_KEYS:
                continue
            redacted[key] = _redact_current_source_value(child, sources)
        return redacted
    if isinstance(value, list):
        if any(value == source["visible_token_ids"] for source in sources):
            return {"redacted_current_source_tokens": True}
        return [_redact_current_source_value(child, sources) for child in value]
    if isinstance(value, str):
        for source in sources:
            value = value.replace(source["source_id"], "[redacted-current-source]")
            value = value.replace(
                source["compatibility_fingerprint"], "[redacted-current-source]"
            )
        return value
    return value


def _redact_current_source_response(
    response: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Keep analysable response fields without persisting raw protocol bytes."""

    safe = _redact_current_source_value(dict(response), sources)
    if not isinstance(safe, dict):  # pragma: no cover - guarded by dict(response)
        raise BenchmarkContractError("current-source response redaction failed")
    # Raw bodies and parsed SSE events could echo request data on a buggy or
    # nonconforming server. The exact-output gate uses output_text instead.
    safe["events"] = []
    safe["raw_body_utf8"] = None
    return safe


def _exact_output_gate(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_arms: Sequence[str] = ARM_NAMES[1:],
) -> dict[str, Any]:
    baseline = {row["request_id"]: row for row in results.get("baseline", ())}
    comparisons: list[dict[str, Any]] = []
    passed = True
    for arm_name in ARM_NAMES[1:]:
        rows = {row["request_id"]: row for row in results.get(arm_name, ())}
        for request_id, baseline_row in baseline.items():
            arm_row = rows.get(request_id)
            reasons: list[str] = []
            if arm_row is None:
                reasons.append("missing_arm_result")
            else:
                baseline_response = baseline_row["response"]
                arm_response = arm_row["response"]
                if baseline_response["error"] is not None or arm_response["error"] is not None:
                    reasons.append("request_error")
                if baseline_response["status"] != arm_response["status"]:
                    reasons.append("http_status_mismatch")
                if baseline_response["output_text"] != arm_response["output_text"]:
                    reasons.append("output_text_mismatch")
                if baseline_response["finish_reason"] != arm_response["finish_reason"]:
                    reasons.append("finish_reason_mismatch")
                baseline_tokens = baseline_response["output_tokens"]
                arm_tokens = arm_response["output_tokens"]
                if baseline_tokens is not None and arm_tokens is not None and baseline_tokens != arm_tokens:
                    reasons.append("output_token_count_mismatch")
            row_passed = not reasons
            required = arm_name in required_arms
            passed = passed and (row_passed or not required)
            comparisons.append(
                {
                    "arm": arm_name,
                    "request_id": request_id,
                    "passed": row_passed,
                    "required": required,
                    "reasons": reasons,
                }
            )
    expected_comparisons = len(baseline) * (len(ARM_NAMES) - 1)
    if len(baseline) == 0 or len(comparisons) != expected_comparisons:
        passed = False
    return {"passed": passed, "required_arms": list(required_arms), "comparisons": comparisons}


def _descriptive_latency(results: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        latencies = [
            row["accounted_latency_ns"]
            for row in results.get(arm_name, ())
            if row["response"]["error"] is None
        ]
        summary[arm_name] = {
            "successful_requests": len(latencies),
            "median_latency_ns": median(latencies) if latencies else None,
        }
    baseline_median = summary["baseline"]["median_latency_ns"]
    for arm_name in ARM_NAMES[1:]:
        arm_median = summary[arm_name]["median_latency_ns"]
        summary[arm_name]["descriptive_baseline_over_arm_median_ratio"] = (
            baseline_median / arm_median
            if baseline_median is not None and arm_median not in {None, 0}
            else None
        )
    return summary


def _bound_capture_receipt(
    capture_receipt_path: Path | None,
    manifest: Mapping[str, Any],
    workload: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Bind a trusted proxy receipt before opening a captured study."""

    workload_class = manifest["workload_class"]
    source_bearing = _workload_has_current_source(workload)
    is_agent_assisted = workload_class == "agent_assisted_calibration"
    is_owner_holdout = workload_class == "owner_representative_holdout"
    requires_receipt = is_agent_assisted or is_owner_holdout
    if not requires_receipt:
        if capture_receipt_path is not None:
            raise BenchmarkContractError(
                "capture receipt is only valid for trusted captured workloads"
            )
        return None
    if capture_receipt_path is None:
        raise BenchmarkContractError(
            "trusted captured workload requires a sealed capture receipt"
        )
    path = capture_receipt_path.resolve()
    document, raw = _read_json(path, "capture receipt")
    source_capture = source_bearing
    expected_keys = (
        _CAPTURE_RECEIPT_V2_KEYS if source_capture else _CAPTURE_RECEIPT_V1_KEYS
    )
    _exact_keys(document, expected_keys, "capture receipt")
    expected_schema = (
        CAPTURE_RECEIPT_SCHEMA_V2 if source_capture else CAPTURE_RECEIPT_SCHEMA_V1
    )
    if document["schema_version"] != expected_schema:
        raise BenchmarkContractError("unsupported capture receipt schema")
    provenance = manifest["workload_provenance"]
    if _sha256_bytes(raw) != provenance["capture_receipt_sha256"]:
        raise BenchmarkContractError("capture receipt SHA-256 does not match manifest")
    if document["origin"] != provenance["origin"]:
        raise BenchmarkContractError("capture receipt origin does not match manifest")
    if document["source_sha256"] != provenance["source_event_sha256"]:
        raise BenchmarkContractError("capture receipt source SHA-256 does not match manifest")
    if document["workload_sha256"] != manifest["workload_sha256"]:
        raise BenchmarkContractError("capture receipt workload SHA-256 does not match manifest")
    if document["contains_responses"] is not False:
        raise BenchmarkContractError("capture receipt must attest that responses are absent")
    _integer(document["source_event_count"], "capture receipt.source_event_count", minimum=1)
    if _integer(
        document["selected_event_count"], "capture receipt.selected_event_count", minimum=1
    ) != len(workload):
        raise BenchmarkContractError("capture receipt selected event count does not bind workload")
    _integer(document["per_episode"], "capture receipt.per_episode", minimum=1)
    _validate_sha256(document["source_sha256"], "capture receipt.source_sha256")
    _validate_sha256(document["workload_sha256"], "capture receipt.workload_sha256")
    _validate_sha256(document["final_event_sha256"], "capture receipt.final_event_sha256")
    request_episode_map = _object(
        document["request_episode_map"], "capture receipt.request_episode_map"
    )
    workload_ids = [row["request_id"] for row in workload]
    if set(request_episode_map) != set(workload_ids):
        raise BenchmarkContractError(
            "capture receipt must bind every frozen workload request exactly once"
        )
    if request_episode_map != manifest["clustering"]["request_cluster_ids"]:
        raise BenchmarkContractError(
            "manifest clustering must exactly match sealed request-to-episode mapping"
        )
    if not all(isinstance(value, str) and value for value in request_episode_map.values()):
        raise BenchmarkContractError("capture receipt episode identifiers must be non-empty strings")
    evidence: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256_bytes(raw),
        "bytes": raw,
        "origin": document["origin"],
        "source_event_sha256": document["source_sha256"],
    }
    if source_capture:
        if document["capture_tool_version"] != "pcss-project-capture-proxy/2":
            raise BenchmarkContractError("source capture receipt requires capture proxy v2")
        if document["origin"] != (
            "agent_assisted_local_llama_server"
            if is_agent_assisted
            else "ordinary_local_llama_server"
        ):
            raise BenchmarkContractError("source capture receipt origin does not match workload class")
        if document["source_sidecar_sha256"] != provenance["source_sidecar_sha256"]:
            raise BenchmarkContractError(
                "capture receipt source sidecar SHA-256 does not match manifest"
            )
        _validate_sha256(
            document["source_sidecar_sha256"],
            "capture receipt.source_sidecar_sha256",
        )
        attachment_count = _integer(
            document["source_attachment_count"],
            "capture receipt.source_attachment_count",
            minimum=1,
        )
        if attachment_count != provenance["source_attachment_count"]:
            raise BenchmarkContractError(
                "capture receipt source attachment count does not match manifest"
            )
        sidecar_event_count = _integer(
            document["source_sidecar_event_count"],
            "capture receipt.source_sidecar_event_count",
            minimum=1,
        )
        if sidecar_event_count < attachment_count:
            raise BenchmarkContractError(
                "capture receipt source sidecar cannot omit selected source attachments"
            )
        _validate_sha256(
            document["final_sidecar_event_sha256"],
            "capture receipt.final_sidecar_event_sha256",
        )
        commitments = _object(
            document["source_attachment_commitments"],
            "capture receipt.source_attachment_commitments",
        )
        source_rows = [
            row
            for row in workload
            if _current_source_from_body(
                _object(row["body"], f"workload.{row['request_id']}.body"),
                f"workload.{row['request_id']}.body",
            )
            is not None
        ]
        source_ids = {row["request_id"] for row in source_rows}
        if attachment_count != len(source_ids) or set(commitments) != source_ids:
            raise BenchmarkContractError(
                "capture receipt source commitments must bind every source-bearing request exactly once"
            )
        for row in source_rows:
            request_id = row["request_id"]
            source = _current_source_from_body(
                _object(row["body"], f"workload.{request_id}.body"),
                f"workload.{request_id}.body",
            )
            assert source is not None  # narrowed by source_rows above
            actual_commitment = _validate_capture_source_commitment(
                commitments[request_id],
                f"capture receipt.source_attachment_commitments.{request_id}",
            )
            if actual_commitment != _capture_source_commitment(source):
                raise BenchmarkContractError(
                    "capture receipt source commitment does not bind frozen workload"
                )
        evidence.update(
            {
                "source_sidecar_sha256": document["source_sidecar_sha256"],
                "source_attachment_count": attachment_count,
            }
        )
    elif document["capture_tool_version"] != "pcss-project-capture-proxy/1":
        raise BenchmarkContractError("capture receipt requires capture proxy v1")
    return evidence


def run_benchmark(
    *,
    manifest_path: Path,
    workload_path: Path,
    model_path: Path,
    server_path: Path,
    output_path: Path,
    capture_receipt_path: Path | None = None,
) -> dict[str, Any]:
    _assert_exclusive_local_compute()
    manifest_path = manifest_path.resolve()
    workload_path = workload_path.resolve()
    model_path = model_path.resolve()
    server_path = server_path.resolve()
    output_path = output_path.resolve()
    inputs = load_inputs(manifest_path, workload_path, model_path, server_path)
    capture_evidence = _bound_capture_receipt(
        capture_receipt_path, inputs.manifest, inputs.workload
    )
    policy_binding = _policy_binding(inputs.manifest)
    policy_spec = _bound_policy_spec(policy_binding)
    current_source_catalog = _current_source_catalog(inputs.workload)
    source_pcss_fingerprint = _arm_by_name(inputs.manifest, "pcss-cache")["pcss"][
        "fingerprint"
    ]
    redacted_source_workload = _redacted_current_source_workload(
        inputs.workload, inputs.workload_sha256, source_pcss_fingerprint
    )
    workload_receipt_bytes = (
        _receipt_json_bytes(redacted_source_workload)
        if redacted_source_workload is not None
        else inputs.workload_bytes
    )
    campaign_started = time.monotonic()
    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise BenchmarkContractError(f"refusing to overwrite output: {output_path}") from error
    holdout_marker = _claim_holdout_once(manifest_path, inputs)
    (output_path / "manifest.json").write_bytes(inputs.manifest_bytes)
    (output_path / "workload.json").write_bytes(workload_receipt_bytes)
    if policy_spec is not None:
        (output_path / "policy-spec.json").write_bytes(policy_spec["bytes"])
    if capture_evidence is not None:
        (output_path / "capture-receipt.json").write_bytes(capture_evidence["bytes"])
    # Native PCSS rejects group/world-accessible stores to keep persisted
    # material scoped to the benchmark owner. The harness creates fresh stores,
    # so make both the parent and each arm directory private from the start.
    stores_path = output_path / "stores"
    stores_path.mkdir(mode=0o700)
    results: dict[str, list[dict[str, Any]]] = {}
    arm_receipts: list[dict[str, Any]] = []
    measured_request_count = 0
    for order_index, arm_name in enumerate(inputs.manifest["arm_order"]):
        arm_path = output_path / "arms" / arm_name
        samples_path = arm_path / "samples"
        samples_path.mkdir(parents=True)
        store_path = stores_path / arm_name
        store_path.mkdir(mode=0o700)
        port = _reserve_loopback_port()
        base_url = f"http://127.0.0.1:{port}"
        command = _server_command(
            server_path,
            model_path,
            inputs.manifest,
            arm_name,
            port,
            store_path,
        )
        stdout_handle = (arm_path / "server.stdout.log").open("wb")
        stderr_handle = (arm_path / "server.stderr.log").open("wb")
        process: subprocess.Popen[bytes] | None = None
        readiness_receipt: dict[str, Any] | None = None
        cleanup_receipt: dict[str, Any] | None = None
        arm_results: list[dict[str, Any]] = []
        arm_error: str | None = None
        metrics_at_ready: dict[str, Any] | None = None
        metrics_at_end: dict[str, Any] | None = None
        server_policy_attestation: dict[str, Any] | None = None
        current_source_metric_attestation: dict[str, Any] | None = None
        resource_snapshots: list[dict[str, Any]] = []
        store_usage_start: dict[str, Any] | None = None
        store_usage_end: dict[str, Any] | None = None
        try:
            wall_budget = float(inputs.manifest["budgets"]["max_wall_clock_seconds"])
            if time.monotonic() - campaign_started >= wall_budget:
                raise BenchmarkContractError("campaign wall-clock budget exhausted")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=str(output_path),
                start_new_session=True,
            )
            readiness_receipt = _wait_ready(
                process,
                base_url,
                inputs.manifest["readiness"],
            )
            startup_ceiling_ns = int(
                float(inputs.manifest["resource_ceilings"]["startup_seconds"]) * 1e9
            )
            if readiness_receipt["latency_ns"] > startup_ceiling_ns:
                raise BenchmarkContractError("startup readiness ceiling exceeded")
            request_timeout = float(inputs.manifest["request_timeout_seconds"])
            metrics_at_ready = _fetch_metrics(base_url, request_timeout)
            if arm_name == "pcss-cache":
                server_policy_attestation = _policy_metric_attestation(
                    metrics_at_ready, policy_binding
                )
                if (
                    policy_binding["policy_id"] is not None
                    and not server_policy_attestation["verified"]
                ):
                    raise BenchmarkContractError(
                        "server policy metric does not attest to the frozen policy"
                    )
                current_source_metric_attestation = _current_source_metric_attestation(
                    metrics_at_ready,
                    required=redacted_source_workload is not None,
                )
                if (
                    redacted_source_workload is not None
                    and not current_source_metric_attestation["verified"]
                ):
                    raise BenchmarkContractError(
                        "server lacks the required current-source activation metrics"
                    )
            else:
                server_policy_attestation = {
                    "required": False,
                    "verified": False,
                    "observed": None,
                    "matching_lines": 0,
                }
            resource_snapshots.append(
                {"stage": "ready", **_process_resource_snapshot(process)}
            )
            rss_ceiling = int(inputs.manifest["resource_ceilings"]["ram_mib"]) * 1024 * 1024
            ready_rss = resource_snapshots[-1]["rss_bytes"]
            if ready_rss is not None and ready_rss > rss_ceiling:
                raise BenchmarkContractError("process RSS ceiling exceeded at readiness")
            store_usage_start = _directory_usage(store_path)
            for request_row in inputs.workload:
                if time.monotonic() - campaign_started >= wall_budget:
                    raise BenchmarkContractError("campaign wall-clock budget exhausted")
                if measured_request_count >= int(inputs.manifest["budgets"]["max_requests"]):
                    raise BenchmarkContractError("campaign request budget exhausted")
                if _tree_size(output_path) >= int(inputs.manifest["budgets"]["max_artifact_bytes"]):
                    raise BenchmarkContractError("campaign artifact budget exhausted")
                body = dict(inputs.manifest["request_defaults"])
                body.update(request_row["body"])
                body.pop("model", None)
                _validate_no_future_fields(body)
                resource_before = _process_resource_snapshot(process)
                metrics_before = _fetch_metrics(base_url, request_timeout)
                response = _dispatch_request(
                    base_url,
                    request_row["endpoint"],
                    body,
                    request_timeout,
                )
                measured_request_count += 1
                metrics_after = _fetch_metrics(base_url, request_timeout)
                resource_after = _process_resource_snapshot(process)
                receipt_body, source_commitment = _redact_current_source_body(
                    body, "dispatched request body", source_pcss_fingerprint
                )
                sample = {
                    "arm": arm_name,
                    "request_id": request_row["request_id"],
                    "chronological_index": request_row["chronological_index"],
                    "endpoint": request_row["endpoint"],
                    "request_body": receipt_body,
                    "process_resource_before": resource_before,
                    "metrics_before": metrics_before,
                    "response": (
                        _redact_current_source_response(response, current_source_catalog)
                        if redacted_source_workload is not None
                        else response
                    ),
                    "metrics_after": metrics_after,
                    "accounted_latency_ns": response["latency_ns"] + metrics_after["latency_ns"],
                    "process_resource_after": resource_after,
                }
                if source_commitment is not None:
                    sample["pcss_current_source_commitment"] = source_commitment
                arm_results.append(sample)
                _write_json(samples_path / f"{len(arm_results) - 1:06d}.json", sample)
                resource_snapshots.extend(
                    (
                        {"stage": f"before:{request_row['request_id']}", **resource_before},
                        {"stage": f"after:{request_row['request_id']}", **resource_after},
                    )
                )
                if resource_after["rss_bytes"] is not None and resource_after["rss_bytes"] > rss_ceiling:
                    raise BenchmarkContractError("process RSS ceiling exceeded")
                current_store_usage = _directory_usage(store_path)
                disk_ceiling = int(float(inputs.manifest["resource_ceilings"]["disk_gib"]) * 1024**3)
                if current_store_usage["bytes"] > disk_ceiling:
                    raise BenchmarkContractError("store disk ceiling exceeded")
                if _tree_size(output_path) > int(inputs.manifest["budgets"]["max_artifact_bytes"]):
                    raise BenchmarkContractError("campaign artifact budget exceeded")
            metrics_at_end = _fetch_metrics(base_url, request_timeout)
            resource_snapshots.append(
                {"stage": "before_cleanup", **_process_resource_snapshot(process)}
            )
            store_usage_end = _directory_usage(store_path)
        except (OSError, BenchmarkContractError) as error:
            arm_error = f"{type(error).__name__}: {error}"
        finally:
            if store_usage_end is None:
                store_usage_end = _directory_usage(store_path)
            if process is not None:
                try:
                    cleanup_receipt = _terminate_process(process)
                except (OSError, subprocess.TimeoutExpired) as error:
                    cleanup_receipt = {
                        "action": "cleanup_error",
                        "returncode": process.poll(),
                        "error": f"{type(error).__name__}: {error}",
                    }
            stdout_handle.close()
            stderr_handle.close()
        results[arm_name] = arm_results
        bytes_written_start = _prometheus_value(metrics_at_ready, "pcss_store_bytes_written_total")
        bytes_written_end = _prometheus_value(metrics_at_end, "pcss_store_bytes_written_total")
        written_delta = (
            bytes_written_end - bytes_written_start
            if bytes_written_start is not None and bytes_written_end is not None
            else None
        )
        store_growth = (
            store_usage_end["bytes"] - store_usage_start["bytes"]
            if store_usage_start is not None and store_usage_end is not None
            else None
        )
        compaction_amplification = (
            written_delta / max(store_growth, 1)
            if written_delta is not None and store_growth is not None
            else None
        )
        startup_evidence = {
            # A policy is an immutable study property, so it is recorded
            # beside every actual arm invocation rather than inferred from
            # a later human-readable label.  The command digest binds that
            # declaration to the exact launch receipt without requiring an
            # unrecognised server CLI flag.
            "policy_binding": policy_binding,
            "command_sha256": _canonical_json_sha256(command),
            "server_policy_attestation": server_policy_attestation,
        }
        if arm_name == "pcss-cache" and redacted_source_workload is not None:
            startup_evidence["current_source_metric_attestation"] = (
                current_source_metric_attestation
            )
        arm_receipt = {
            "arm": arm_name,
            "order_index": order_index,
            "command": command,
            "startup_evidence": startup_evidence,
            "declared_state": inputs.manifest["arm_store_initialization"][arm_name]["state"],
            "readiness": readiness_receipt,
            "cold_start_readiness_latency_ns": (
                readiness_receipt["latency_ns"]
                if readiness_receipt is not None
                and inputs.manifest["arm_store_initialization"][arm_name]["state"] == "cold"
                else None
            ),
            "metrics_at_ready": metrics_at_ready,
            "metrics_at_end": metrics_at_end,
            "process_resource_snapshots": resource_snapshots,
            "store_usage_start": store_usage_start,
            "store_usage_end": store_usage_end,
            "store_growth_bytes": store_growth,
            "store_bytes_written_delta": written_delta,
            "compaction_write_amplification": compaction_amplification,
            "request_results": len(arm_results),
            "error": arm_error,
            "cleanup": cleanup_receipt,
        }
        arm_receipts.append(arm_receipt)
        _write_json(arm_path / "arm.json", arm_receipt)
    exact_gate = _exact_output_gate(
        results,
        required_arms=inputs.manifest["effect_gates"].get("exact_output_arms", list(ARM_NAMES[1:])),
    )
    metrics_complete = all(
        sample[side]["status"] == 200 and sample[side]["error"] is None
        for rows in results.values()
        for sample in rows
        for side in ("metrics_before", "metrics_after")
    ) and all(
        receipt[side] is not None
        and receipt[side]["status"] == 200
        and receipt[side]["error"] is None
        for receipt in arm_receipts
        for side in ("metrics_at_ready", "metrics_at_end")
    )
    request_matrix_complete = all(
        len(results.get(arm_name, ())) == len(inputs.workload) for arm_name in ARM_NAMES
    )
    no_arm_errors = all(receipt["error"] is None for receipt in arm_receipts)
    cleanup_complete = all(
        receipt["cleanup"] is not None
        and receipt["cleanup"].get("returncode") is not None
        and receipt["cleanup"].get("descendants_absent") is True
        for receipt in arm_receipts
    )
    amplification_ceiling = float(
        inputs.manifest["resource_ceilings"]["compaction_write_amplification_max"]
    )
    resource_ceilings_passed = all(
        receipt["compaction_write_amplification"] is not None
        and receipt["compaction_write_amplification"] <= amplification_ceiling
        for receipt in arm_receipts
    )
    resource_evidence_complete = all(
        receipt["readiness"] is not None
        and receipt["metrics_at_ready"] is not None
        and receipt["metrics_at_end"] is not None
        and receipt["store_usage_start"] is not None
        and receipt["store_usage_start"]["error"] is None
        and not receipt["store_usage_start"]["truncated"]
        and receipt["store_usage_end"] is not None
        and receipt["store_usage_end"]["error"] is None
        and not receipt["store_usage_end"]["truncated"]
        and receipt["process_resource_snapshots"]
        and all(snapshot["error"] is None for snapshot in receipt["process_resource_snapshots"])
        for receipt in arm_receipts
    )
    source_metrics_complete = True
    if redacted_source_workload is not None:
        pcss_receipt = next(
            (receipt for receipt in arm_receipts if receipt["arm"] == "pcss-cache"),
            None,
        )
        source_metrics_complete = bool(
            isinstance(pcss_receipt, dict)
            and isinstance(pcss_receipt.get("startup_evidence"), dict)
            and pcss_receipt["startup_evidence"].get(
                "current_source_metric_attestation"
            )
            == _current_source_metric_attestation(
                pcss_receipt.get("metrics_at_ready"), required=True
            )
            and pcss_receipt["startup_evidence"]["current_source_metric_attestation"][
                "verified"
            ]
        )
    representative_protocol_declared = bool(
        inputs.manifest["holdout"]
        and inputs.manifest["workload_class"] == "owner_representative_holdout"
        and inputs.manifest["effect_gates"]
    )
    elapsed_seconds = time.monotonic() - campaign_started
    budget_status = {
        "measured_requests": measured_request_count,
        "max_requests": inputs.manifest["budgets"]["max_requests"],
        "requests_within_budget": measured_request_count
        <= inputs.manifest["budgets"]["max_requests"],
        "elapsed_seconds": elapsed_seconds,
        "max_wall_clock_seconds": inputs.manifest["budgets"]["max_wall_clock_seconds"],
        "wall_clock_within_budget": elapsed_seconds
        <= inputs.manifest["budgets"]["max_wall_clock_seconds"],
        "artifact_bytes": _tree_size(output_path),
        "max_artifact_bytes": inputs.manifest["budgets"]["max_artifact_bytes"],
        "artifact_within_budget": True,
    }
    budgets_complete = bool(
        budget_status["requests_within_budget"]
        and budget_status["wall_clock_within_budget"]
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "study_id": inputs.manifest["study_id"],
        "manifest_sha256": inputs.manifest_sha256,
        "workload_sha256": inputs.workload_sha256,
        "model_sha256": inputs.model_sha256,
        "server_sha256": inputs.server_sha256,
        "policy_binding": policy_binding,
        "policy_spec": (
            {
                "relative_path": policy_spec["relative_path"],
                "sha256": policy_spec["sha256"],
            }
            if policy_spec is not None
            else None
        ),
        "capture_receipt_sha256": (
            capture_evidence["sha256"] if capture_evidence is not None else None
        ),
        "holdout_opening_marker": str(holdout_marker) if holdout_marker is not None else None,
        "arm_order": inputs.manifest["arm_order"],
        "arm_receipts": arm_receipts,
        "exact_output_gate": exact_gate,
        "metrics_complete": metrics_complete,
        "request_matrix_complete": request_matrix_complete,
        "cleanup_complete": cleanup_complete,
        "resource_evidence_complete": resource_evidence_complete,
        "resource_ceilings_passed": resource_ceilings_passed,
        "measured_request_count": measured_request_count,
        "budget_status": budget_status,
        "descriptive_latency_only": _descriptive_latency(results),
        "analysis_status": "raw_samples_only_no_confidence_analysis",
        "general_speedup_claim": False,
        "representative_effect_evaluation_permitted": bool(
            representative_protocol_declared and policy_binding["policy_id"] is not None
        ),
        "claim_boundary": inputs.manifest["claim_boundary"],
        "valid_raw_run": bool(
            exact_gate["passed"]
            and metrics_complete
            and request_matrix_complete
            and no_arm_errors
            and cleanup_complete
            and resource_evidence_complete
            and resource_ceilings_passed
            and budgets_complete
            and source_metrics_complete
        ),
    }
    if redacted_source_workload is not None:
        summary["source_workload_redaction"] = {
            "schema_version": REDACTED_CURRENT_SOURCE_WORKLOAD_SCHEMA,
            "original_workload_sha256": inputs.workload_sha256,
            "redacted_workload_sha256": _sha256_bytes(workload_receipt_bytes),
            "source_request_count": sum(
                "pcss_current_source_commitment" in row
                for row in redacted_source_workload["requests"]
            ),
            "source_metrics_complete": source_metrics_complete,
        }
    _write_json(output_path / "result.json", summary)
    for _ in range(3):
        artifact_bytes = _tree_size(output_path)
        artifact_within_budget = artifact_bytes <= int(
            inputs.manifest["budgets"]["max_artifact_bytes"]
        )
        if (
            summary["budget_status"]["artifact_bytes"] == artifact_bytes
            and summary["budget_status"]["artifact_within_budget"] == artifact_within_budget
        ):
            break
        summary["budget_status"]["artifact_bytes"] = artifact_bytes
        summary["budget_status"]["artifact_within_budget"] = artifact_within_budget
        if not artifact_within_budget:
            summary["valid_raw_run"] = False
        _write_json(output_path / "result.json", summary)
    for artifact in output_path.rglob("*"):
        if artifact.is_file():
            artifact.chmod(0o444)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capture-receipt",
        type=Path,
        help="required sealed receipt for agent-assisted calibration workloads",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = run_benchmark(
            manifest_path=arguments.manifest,
            workload_path=arguments.workload,
            model_path=arguments.model,
            server_path=arguments.server,
            output_path=arguments.output,
            capture_receipt_path=arguments.capture_receipt,
        )
    except BenchmarkContractError as error:
        print(f"benchmark contract error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["valid_raw_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
