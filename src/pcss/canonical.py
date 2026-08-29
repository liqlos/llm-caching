"""Canonical JSON and fingerprints shared by all offline artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


class ContractError(ValueError):
    """Raised when an input violates a deliberately fail-closed contract."""


def canonical_json(value: Any) -> str:
    """Return the one UTF-8-safe JSON spelling used for fingerprints and JSONL."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={sorted(missing)}")
        if unknown:
            parts.append(f"unknown={sorted(unknown)}")
        raise ContractError(f"{field} has invalid keys: {', '.join(parts)}")


def require_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ContractError(f"{field} must be a{' non-empty' if nonempty else ''} string")
    return value


def require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} must be an integer >= {minimum}")
    return value
