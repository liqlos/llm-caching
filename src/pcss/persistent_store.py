"""Small crash-recoverable persistent store for conservative PCSS candidates.

The store deliberately keeps its on-disk format boring: each learned candidate
is written to an immutable, checksummed segment, while a separately checksummed
manifest is atomically replaced.  A server can treat every exception from this
module as an abstention; no stored draft is ever required for correctness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence


FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
SEGMENT_PREFIX = "segment-"
SEGMENT_NAME_RE = re.compile(r"^segment-(?:[0-9]+-[0-9a-f]{32}|compact-[0-9a-f]{32})\.json$")


class PersistentStoreError(RuntimeError):
    """Base class for a store failure which should make the provider abstain."""


class IncompatibleStoreError(PersistentStoreError):
    """The existing store belongs to a different model or privacy namespace."""


class CorruptStoreError(PersistentStoreError):
    """A manifest or segment cannot be authenticated."""


class ReadOnlyStoreError(PersistentStoreError):
    """A mutating operation was attempted on a read-only store."""


@dataclass(frozen=True)
class StoreIdentity:
    """All properties which must match before disk data may be reused."""

    namespace: str
    model: str
    tokenizer: str
    vocabulary: str
    chat_template: str
    thinking_policy: str
    repository_scope: str = "private"

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"identity.{name} must be a non-empty string")


@dataclass(frozen=True)
class LookupResult:
    """A candidate learned only from an already committed completion."""

    record_id: str
    context_length: int
    completion: tuple[int, ...]
    utility: float


@dataclass(frozen=True)
class StoreStats:
    segment_count: int
    record_count: int
    bytes_on_disk: int
    evictions: int
    corruptions: int


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tokens(value: Iterable[int], field: str, *, allow_empty: bool) -> tuple[int, ...]:
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in result):
        raise ValueError(f"{field} must contain non-negative integer token ids")
    return result


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise CorruptStoreError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise CorruptStoreError(f"{field} must be a finite number")
    return float(value)


class PersistentSegmentStore:
    """Immutable-segment store with exact suffix lookup and deterministic eviction.

    The hot index contains only segment metadata and record locations, rather
    than eagerly materialising a corpus-level token trie.  Segment contents are
    read and verified only for candidates whose context is a suffix of a query.
    """

    def __init__(
        self,
        path: str | Path,
        identity: StoreIdentity,
        *,
        max_bytes: int = 200 * 1024**3,
        read_only: bool = False,
        clock_ns: callable | None = None,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.path = Path(path)
        self.identity = identity
        self.max_bytes = max_bytes
        self.read_only = read_only
        self._clock_ns = clock_ns or time.time_ns
        self._lock = RLock()
        self._manifest: dict[str, Any] = {}
        self._segments: dict[str, dict[str, Any]] = {}
        self._evictions = 0
        self._corruptions = 0
        self._open()

    def _require_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyStoreError("store is read-only")

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "identity": asdict(self.identity),
            "segments": [],
            "evictions": 0,
        }

    @staticmethod
    def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
        return {"payload": payload, "sha256": _digest(payload)}

    @staticmethod
    def _unwrap(value: Any, kind: str) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"payload", "sha256"}:
            raise CorruptStoreError(f"invalid {kind} envelope")
        if not isinstance(value["payload"], dict) or not isinstance(value["sha256"], str):
            raise CorruptStoreError(f"invalid {kind} envelope")
        if _digest(value["payload"]) != value["sha256"]:
            raise CorruptStoreError(f"{kind} checksum mismatch")
        return value["payload"]

    def _atomic_json(self, destination: Path, value: dict[str, Any]) -> None:
        self._require_writable()
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        encoded = _canonical_bytes(value)
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _read_json(self, path: Path, kind: str) -> dict[str, Any]:
        try:
            return self._unwrap(json.loads(path.read_text(encoding="utf-8")), kind)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorruptStoreError(f"cannot read {kind} {path.name}") from error

    def _quarantine(self, path: Path) -> None:
        self._corruptions += 1
        if self.read_only or not path.exists() or not path.is_file():
            return
        target = path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex}")
        os.replace(path, target)

    def _validate_segment(self, name: str, expected_sha256: str | None = None) -> dict[str, Any]:
        if not isinstance(name, str) or not SEGMENT_NAME_RE.fullmatch(name):
            raise CorruptStoreError("invalid segment name")
        path = self.path / name
        payload = self._read_json(path, "segment")
        if payload.get("format_version") != FORMAT_VERSION or payload.get("identity") != asdict(self.identity):
            raise CorruptStoreError(f"segment {name} has incompatible format or identity")
        if set(payload) != {"format_version", "identity", "records"} or not isinstance(payload.get("records"), list):
            raise CorruptStoreError(f"segment {name} has no records")
        digest = _digest(payload)
        if expected_sha256 is not None and digest != expected_sha256:
            raise CorruptStoreError(f"segment {name} manifest checksum mismatch")
        for record in payload["records"]:
            if not isinstance(record, dict) or set(record) != {"id", "context", "completion", "utility", "recency_ns"}:
                raise CorruptStoreError(f"segment {name} contains an invalid record")
            try:
                _tokens(record["context"], "record.context", allow_empty=False)
                _tokens(record["completion"], "record.completion", allow_empty=False)
            except (TypeError, ValueError) as error:
                raise CorruptStoreError(f"segment {name} contains invalid tokens") from error
            if not isinstance(record["id"], str) or not re.fullmatch(r"[0-9a-f]{32}", record["id"]):
                raise CorruptStoreError(f"segment {name} contains an invalid record")
            _finite_number(record["utility"], "record.utility")
            _integer(record["recency_ns"], "record.recency_ns", minimum=0)
        return payload

    def _open(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._require_writable()
                self.path.mkdir(parents=True)
            if not self.path.is_dir():
                raise PersistentStoreError("store path is not a directory")
            if not self.manifest_path.exists():
                if self.read_only:
                    raise PersistentStoreError("read-only store has no manifest")
                self._manifest = self._empty_manifest()
                self._atomic_json(self.manifest_path, self._envelope(self._manifest))
            else:
                try:
                    self._manifest = self._read_json(self.manifest_path, "manifest")
                except CorruptStoreError:
                    self._quarantine(self.manifest_path)
                    raise
            if self._manifest.get("format_version") != FORMAT_VERSION:
                raise IncompatibleStoreError("unsupported store format")
            if self._manifest.get("identity") != asdict(self.identity):
                raise IncompatibleStoreError("store identity does not match this server")
            if not isinstance(self._manifest.get("segments"), list):
                raise CorruptStoreError("manifest segments must be a list")
            if set(self._manifest) != {"format_version", "identity", "segments", "evictions"}:
                raise CorruptStoreError("invalid manifest schema")
            self._evictions = _integer(self._manifest["evictions"], "manifest evictions")
            self._load_active_segments()
            self._recover_orphans()

    def _load_active_segments(self) -> None:
        self._segments.clear()
        changed = False
        active: list[dict[str, Any]] = []
        for item in self._manifest["segments"]:
            if not isinstance(item, dict) or set(item) != {"name", "sha256", "bytes"}:
                raise CorruptStoreError("invalid manifest segment entry")
            name = item["name"]
            if not isinstance(name, str) or not SEGMENT_NAME_RE.fullmatch(name):
                raise CorruptStoreError("invalid manifest segment name")
            if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise CorruptStoreError("invalid manifest segment checksum")
            recorded_bytes = _integer(item["bytes"], "manifest segment bytes", minimum=1)
            try:
                payload = self._validate_segment(name, item["sha256"])
                if (self.path / name).stat().st_size != recorded_bytes:
                    raise CorruptStoreError(f"segment {name} byte count mismatch")
            except CorruptStoreError:
                self._quarantine(self.path / str(name))
                changed = True
                continue
            self._segments[name] = payload
            active.append(item)
        if changed:
            if self.read_only:
                return
            self._manifest["segments"] = active
            self._write_manifest()

    def _recover_orphans(self) -> None:
        active = set(self._segments)
        for item in self.path.iterdir():
            if ".corrupt-" in item.name:
                continue
            if item.name.startswith(".") and item.name.endswith(".tmp"):
                if not self.read_only:
                    self._quarantine(item)
            elif item.name.startswith(SEGMENT_PREFIX) and item.name not in active and not self.read_only:
                # A segment written before the manifest commit is not durable;
                # never resurrect it because that could learn an aborted request.
                self._quarantine(item)

    def _write_manifest(self) -> None:
        self._manifest["evictions"] = self._evictions
        self._atomic_json(self.manifest_path, self._envelope(self._manifest))

    def append(self, context: Sequence[int], completion: Sequence[int], *, utility: float = 0.0) -> str:
        """Durably learn a committed context-to-completion mapping."""
        context_tokens = _tokens(context, "context", allow_empty=False)
        completion_tokens = _tokens(completion, "completion", allow_empty=False)
        if not isinstance(utility, (int, float)) or isinstance(utility, bool) or not math.isfinite(utility):
            raise ValueError("utility must be a finite number")
        with self._lock:
            self._require_writable()
            record = {
                "id": uuid.uuid4().hex,
                "context": list(context_tokens),
                "completion": list(completion_tokens),
                "utility": float(utility),
                "recency_ns": int(self._clock_ns()),
            }
            payload = {"format_version": FORMAT_VERSION, "identity": asdict(self.identity), "records": [record]}
            name = f"{SEGMENT_PREFIX}{record['recency_ns']}-{record['id']}.json"
            self._atomic_json(self.path / name, self._envelope(payload))
            entry = {"name": name, "sha256": _digest(payload), "bytes": (self.path / name).stat().st_size}
            self._segments[name] = payload
            self._manifest["segments"].append(entry)
            removed = self._evict_to_capacity()
            self._write_manifest()
            self._remove_committed_orphans(removed)
            return record["id"]

    learn = append

    def _remove_committed_orphans(self, names: Iterable[str]) -> None:
        """Delete files only after a manifest no longer references them."""
        for name in names:
            (self.path / name).unlink(missing_ok=True)

    def _evict_to_capacity(self) -> tuple[str, ...]:
        total = sum(int(entry["bytes"]) for entry in self._manifest["segments"])
        removed: list[str] = []
        while self._manifest["segments"] and total > self.max_bytes:
            candidates: list[tuple[float, int, str, dict[str, Any]]] = []
            for entry in self._manifest["segments"]:
                payload = self._segments[entry["name"]]
                record = payload["records"][0]
                candidates.append((float(record["utility"]), int(record["recency_ns"]), entry["name"], entry))
            _, _, name, victim = min(candidates)
            self._manifest["segments"].remove(victim)
            total -= int(victim["bytes"])
            self._segments.pop(name, None)
            removed.append(name)
            self._evictions += 1
        return tuple(removed)

    def lookup(self, visible_suffix: Sequence[int], *, max_tokens: int | None = None) -> LookupResult | None:
        """Return the strongest exact-context candidate, or ``None`` to abstain."""
        suffix = _tokens(visible_suffix, "visible_suffix", allow_empty=True)
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("max_tokens must be a positive integer")
        with self._lock:
            matches: list[dict[str, Any]] = []
            for payload in self._segments.values():
                for record in payload["records"]:
                    context = tuple(record["context"])
                    if len(context) <= len(suffix) and suffix[-len(context) :] == context if context else True:
                        matches.append(record)
            if not matches:
                return None
            best = max(matches, key=lambda item: (len(item["context"]), float(item["utility"]), int(item["recency_ns"]), item["id"]))
            completion = tuple(best["completion"][:max_tokens]) if max_tokens else tuple(best["completion"])
            return LookupResult(best["id"], len(best["context"]), completion, float(best["utility"]))

    def compact(self) -> bool:
        """Rewrite active records into one immutable segment and atomically install it."""
        with self._lock:
            self._require_writable()
            if len(self._segments) < 2:
                return False
            # Preserve every record field, particularly utility and recency, so
            # lookup ranking remains identical before and after compaction.
            records = [record for payload in self._segments.values() for record in payload["records"]]
            records.sort(key=lambda item: item["id"])
            payload = {"format_version": FORMAT_VERSION, "identity": asdict(self.identity), "records": records}
            name = f"{SEGMENT_PREFIX}compact-{uuid.uuid4().hex}.json"
            self._atomic_json(self.path / name, self._envelope(payload))
            entry = {"name": name, "sha256": _digest(payload), "bytes": (self.path / name).stat().st_size}
            old_names = tuple(self._segments)
            self._segments = {name: payload}
            self._manifest["segments"] = [entry]
            removed = self._evict_to_capacity()
            self._write_manifest()
            self._remove_committed_orphans((*old_names, *removed))
            return True

    def stats(self) -> StoreStats:
        with self._lock:
            return StoreStats(
                segment_count=len(self._segments),
                record_count=sum(len(payload["records"]) for payload in self._segments.values()),
                bytes_on_disk=sum(int(entry["bytes"]) for entry in self._manifest["segments"]),
                evictions=self._evictions,
                corruptions=self._corruptions,
            )
