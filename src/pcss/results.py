"""Reproducible result receipts for real offline runs and future hardware runs."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .canonical import ContractError, canonical_json, require_string, sha256

EvidenceClass = Literal["SIMULATOR", "LOCAL_MEASURED", "NOT_RUN", "OFFLINE_FIXTURE"]


def _runtime_repository_root() -> Path:
    """Return the checkout that supplied this receipt implementation."""

    return Path(__file__).resolve().parents[2]


def _require_runtime_repository(repository_root: Path) -> Path:
    resolved = repository_root.resolve()
    if resolved != _runtime_repository_root():
        raise ContractError(
            "receipt repository_root must be the checkout that supplied the running pcss package"
        )
    return resolved


def resolve_git_revision(repository_root: Path) -> str:
    """Read full Git HEAD only from the clean checkout supplying this package."""

    resolved = _require_runtime_repository(repository_root)
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=resolved,
            capture_output=True,
            check=False,
            text=True,
        )
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=resolved,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError as error:
        raise ContractError("git is required to create a reproducible PCSS receipt") from error
    if status.returncode != 0:
        raise ContractError(f"unable to inspect Git worktree for receipt: {resolved}")
    if status.stdout:
        raise ContractError("refuse receipt from a dirty or staged worktree")
    revision = completed.stdout.strip()
    if completed.returncode != 0 or len(revision) != 40:
        raise ContractError(f"unable to resolve Git HEAD for receipt: {resolved}")
    return revision


def recorded_module_command(
    module: str, command_arguments: Sequence[str], repository_root: Path
) -> str:
    """Make a rerunnable Python-module command from the actual parsed argv."""

    module = require_string(module, "module")
    source_path = _require_runtime_repository(repository_root) / "src"
    invocation = shlex.join((sys.executable, "-m", module, *command_arguments))
    return f"PYTHONPATH={shlex.quote(str(source_path))} {invocation}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ResultReceipt:
    schema_version: str
    evidence_class: EvidenceClass
    command: str
    git_revision: str
    input_fingerprints: dict[str, str]
    status: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != "pcss.result/v1":
            raise ContractError("unsupported result.schema_version")
        require_string(self.command, "result.command")
        require_string(self.git_revision, "result.git_revision")
        require_string(self.status, "result.status")
        if not self.input_fingerprints:
            raise ContractError("result.input_fingerprints must not be empty")
        for label, fingerprint in self.input_fingerprints.items():
            require_string(label, "result.input_fingerprints key")
            if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
                raise ContractError(f"result fingerprint is not SHA-256: {label}")

    @property
    def fingerprint(self) -> str:
        return sha256(asdict(self))

    def as_json(self) -> str:
        return canonical_json(asdict(self))


def make_receipt(
    *,
    evidence_class: EvidenceClass,
    command: str,
    git_revision: str,
    input_fingerprints: dict[str, str],
    status: str,
    payload: dict[str, Any],
) -> ResultReceipt:
    return ResultReceipt(
        schema_version="pcss.result/v1",
        evidence_class=evidence_class,
        command=command,
        git_revision=git_revision,
        input_fingerprints=dict(sorted(input_fingerprints.items())),
        status=status,
        payload=payload,
    )


def write_raw_result(path: Path, receipt: ResultReceipt) -> str:
    """Write one canonical immutable raw receipt and return its file SHA-256."""

    if path.exists():
        raise ContractError("raw result path already exists; preserve the prior receipt and choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(receipt.as_json() + "\n", encoding="utf-8")
    return file_sha256(path)


def _offline_replay_summary_lines(payload: Mapping[str, Any]) -> list[str]:
    """Render only measured model-free replay quantities already in a receipt."""

    if payload.get("kind") != "model_free_chronological_replay":
        return []
    lines = [
        "- Replay result:",
        f"  - Trace records: `{payload.get('trace_records')}`; proposal positions: `{payload.get('proposal_positions')}`; future-token retrieval: `{payload.get('future_tokens_used_for_retrieval')}`",
    ]
    capacity = payload.get("capacity_simulator")
    if isinstance(capacity, Mapping) and isinstance(capacity.get("strategy_summary"), Mapping):
        lines.append("  - Logical active-view capture (oracle-valued unique persistent tokens):")
        for policy, points in sorted(capacity["strategy_summary"].items()):
            if not isinstance(points, list):
                continue
            compact_points = []
            for point in points:
                if not isinstance(point, Mapping):
                    continue
                compact_points.append(
                    f"{point.get('capacity_bytes')} B: {point.get('active_capture_ratio', 0.0):.2%}"
                )
            if compact_points:
                lines.append(f"    - `{policy}`: {', '.join(compact_points)}")
    scope = payload.get("scope_strategy_summary")
    if isinstance(scope, list):
        lines.append("  - Scope selection (pre-target rank; post-hoc oracle evaluation):")
        for point in scope:
            if not isinstance(point, Mapping):
                continue
            lines.append(
                f"    - `{point.get('name')}`: selected `{point.get('selected_prefix_value')}` / candidate oracle `{point.get('candidate_oracle_prefix_value')}`; zero-accept selections `{point.get('zero_accept_positions')}`"
            )
    return lines


def write_compact_report(path: Path, *, raw_result: Path, title: str) -> None:
    """Create a compact report only from an already-written raw result."""
    require_string(title, "report.title")
    if not raw_result.is_file():
        raise ContractError("a compact report requires an existing raw result")
    receipt = json.loads(raw_result.read_text(encoding="utf-8"))
    required = {"schema_version", "evidence_class", "command", "git_revision", "input_fingerprints", "status", "payload"}
    if set(receipt) != required or receipt["schema_version"] != "pcss.result/v1":
        raise ContractError("raw result does not have the expected PCSS receipt shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"- Evidence class: `{receipt['evidence_class']}`",
        f"- Status: `{receipt['status']}`",
        f"- Git revision: `{receipt['git_revision']}`",
        f"- Exact command: `{receipt['command']}`",
        f"- Raw result: `{raw_result.name}` (`sha256:{file_sha256(raw_result)}`)",
        "- Input fingerprints:",
    ]
    lines.extend(f"  - `{label}`: `sha256:{fingerprint}`" for label, fingerprint in sorted(receipt["input_fingerprints"].items()))
    payload = receipt["payload"]
    if isinstance(payload, Mapping):
        lines.extend(_offline_replay_summary_lines(payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
