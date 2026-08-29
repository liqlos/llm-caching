"""Small immutable token-archive artifacts for host-independent replay.

This module deliberately models a read-only archive, not a live PCSS storage
engine.  A builder emits one sealed artifact; readers verify every component
before serving bounded fixed-key reference lookups and cursor reads.
"""

from __future__ import annotations

import hashlib
import json
import struct
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import ContractError, canonical_json, require_int, require_string


ARCHIVE_SCHEMA_VERSION = "pcss.immutable-archive/v1"
RECORDS_SCHEMA_VERSION = "pcss.immutable-archive-records/v1"
TOKEN_WIDTH_BITS = 32
TOKEN_BYTE_ORDER = "little"
REFERENCE_KEY_TOKENS = 4
REFERENCE_KEY_BYTES = 16
REFERENCE_POSTING_LIMIT_DEFAULT = 32

_TOKEN_STRUCT = struct.Struct("<I")
_REFERENCE_STRUCT = struct.Struct(f"<{REFERENCE_KEY_BYTES}sII")
_TOKENS_FILE = "tokens.u32"
_RECORDS_FILE = "records.json"
_INDEX_FILE = "reference-index.bin"
_MANIFEST_FILE = "archive-manifest.json"
_SEAL_FILE = "archive-manifest.sha256"
_ARTIFACT_FILES = frozenset(
    {_TOKENS_FILE, _RECORDS_FILE, _INDEX_FILE, _MANIFEST_FILE, _SEAL_FILE}
)


class ArchiveContractError(ContractError):
    """Raised when an immutable archive input or artifact is invalid."""


def _raw_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_u32(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ArchiveContractError(f"{field} must be an unsigned 32-bit integer")
    return value


def _normalise_tokens(tokens: Iterable[int], field: str) -> tuple[int, ...]:
    if isinstance(tokens, (str, bytes, bytearray)):
        raise ArchiveContractError(f"{field} must be an iterable of unsigned 32-bit integers")
    try:
        normalized = tuple(tokens)
    except TypeError as exc:
        raise ArchiveContractError(
            f"{field} must be an iterable of unsigned 32-bit integers"
        ) from exc
    return tuple(_require_u32(token, f"{field}[{index}]") for index, token in enumerate(normalized))


def _normalise_metadata(metadata: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ArchiveContractError(f"{field} must be an object")
    try:
        normalized = json.loads(canonical_json(dict(metadata)))
    except (TypeError, ValueError) as exc:
        raise ArchiveContractError(f"{field} must be canonical JSON data") from exc
    if not isinstance(normalized, dict):  # Defensive: JSON objects are expected above.
        raise ArchiveContractError(f"{field} must be an object")
    return normalized


def _pack_tokens(tokens: tuple[int, ...]) -> bytes:
    if not tokens:
        return b""
    return struct.pack(f"<{len(tokens)}I", *tokens)


def _reference_key(tokens: tuple[int, ...]) -> bytes:
    """Return the fixed-width lookup key; callers still compare exact tokens."""
    return hashlib.sha256(_pack_tokens(tokens)).digest()[:REFERENCE_KEY_BYTES]


@dataclass(frozen=True)
class TokenRecord:
    """One logical immutable token stream and its JSON-safe metadata."""

    record_id: str
    tokens: tuple[int, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", require_string(self.record_id, "record_id"))
        object.__setattr__(self, "tokens", _normalise_tokens(self.tokens, "tokens"))
        object.__setattr__(self, "metadata", _normalise_metadata(self.metadata, "metadata"))


@dataclass(frozen=True)
class ArchiveRecord:
    """Descriptor for a stream stored in the common u32 token file."""

    record_id: str
    start_token: int
    token_count: int
    metadata: Mapping[str, Any]
    token_sha256: str

    @property
    def end_token(self) -> int:
        return self.start_token + self.token_count


@dataclass(frozen=True)
class ArchiveReference:
    """A reference into a record; no token payload is copied into the index."""

    record_id: str
    token_offset: int


@dataclass(frozen=True)
class _ReferenceEntry:
    key: bytes
    record_index: int
    token_offset: int


class ArchiveCursor:
    """Sequential bounded reads within exactly one immutable record."""

    def __init__(self, archive: "ImmutableTokenArchive", record: ArchiveRecord, position: int) -> None:
        self._archive = archive
        self._record = record
        self._position = position

    @property
    def record_id(self) -> str:
        return self._record.record_id

    @property
    def position(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return self._record.token_count - self._position

    def seek(self, position: int) -> None:
        position = require_int(position, "cursor.position", minimum=0)
        if position > self._record.token_count:
            raise ArchiveContractError("cursor.position exceeds record length")
        self._position = position

    def read(self, count: int) -> tuple[int, ...]:
        count = require_int(count, "cursor.count", minimum=0)
        actual_count = min(count, self.remaining)
        tokens = self._archive.read_tokens(self._record.record_id, self._position, actual_count)
        self._position += actual_count
        return tokens


class ImmutableTokenArchive:
    """Verified reader for a single sealed archive artifact.

    The reader never mutates the artifact.  Its reference index is deliberately
    simple: fixed-width hash keys are sorted on disk and every candidate is
    compared with the requested u32 suffix before it is returned.
    """

    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        records: tuple[ArchiveRecord, ...],
        entries: tuple[_ReferenceEntry, ...],
    ) -> None:
        self._root = root
        self._manifest = dict(manifest)
        self._records = records
        self._record_by_id = {record.record_id: record for record in records}
        self._record_indexes = {record.record_id: index for index, record in enumerate(records)}
        self._entries = entries
        self._keys = tuple(entry.key for entry in entries)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def records(self) -> tuple[ArchiveRecord, ...]:
        return self._records

    @property
    def token_count(self) -> int:
        return self._manifest["token_count"]

    @property
    def reference_count(self) -> int:
        return self._manifest["reference_count"]

    @property
    def posting_limit(self) -> int:
        return self._manifest["posting_limit"]

    def manifest(self) -> dict[str, Any]:
        """Return a detached JSON-compatible copy of the sealed manifest."""
        return json.loads(canonical_json(self._manifest))

    def byte_accounting(self) -> dict[str, int]:
        """Return exact on-disk bytes, including the manifest and its seal."""
        files = {
            "tokens_u32": self._root / _TOKENS_FILE,
            "records_json": self._root / _RECORDS_FILE,
            "reference_index": self._root / _INDEX_FILE,
            "manifest": self._root / _MANIFEST_FILE,
            "manifest_seal": self._root / _SEAL_FILE,
        }
        accounting = {name: path.stat().st_size for name, path in files.items()}
        accounting["total"] = sum(accounting.values())
        return accounting

    def open_cursor(self, record_id: str, *, position: int = 0) -> ArchiveCursor:
        record_id = require_string(record_id, "record_id")
        if record_id not in self._record_by_id:
            raise ArchiveContractError(f"unknown record_id: {record_id}")
        cursor = ArchiveCursor(self, self._record_by_id[record_id], 0)
        cursor.seek(position)
        return cursor

    def read_tokens(self, record_id: str, start: int, count: int) -> tuple[int, ...]:
        record_id = require_string(record_id, "record_id")
        start = require_int(start, "start", minimum=0)
        count = require_int(count, "count", minimum=0)
        try:
            record = self._record_by_id[record_id]
        except KeyError as exc:
            raise ArchiveContractError(f"unknown record_id: {record_id}") from exc
        if start > record.token_count or count > record.token_count - start:
            raise ArchiveContractError("token read exceeds record boundary")
        if count == 0:
            return ()
        byte_start = (record.start_token + start) * _TOKEN_STRUCT.size
        byte_count = count * _TOKEN_STRUCT.size
        with (self._root / _TOKENS_FILE).open("rb") as handle:
            handle.seek(byte_start)
            payload = handle.read(byte_count)
        if len(payload) != byte_count:
            raise ArchiveContractError("token stream ended before the requested cursor read")
        return tuple(struct.unpack(f"<{count}I", payload))

    def find_references(
        self, query_tokens: Iterable[int], *, limit: int | None = None
    ) -> tuple[ArchiveReference, ...]:
        """Find exact fixed-length suffix references with bounded postings."""
        query = _normalise_tokens(query_tokens, "query_tokens")
        if len(query) != REFERENCE_KEY_TOKENS:
            raise ArchiveContractError(
                f"query_tokens must contain exactly {REFERENCE_KEY_TOKENS} tokens"
            )
        if limit is not None:
            limit = require_int(limit, "limit", minimum=1)
        key = _reference_key(query)
        begin = bisect_left(self._keys, key)
        end = bisect_right(self._keys, key)
        if end - begin > self.posting_limit:
            raise ArchiveContractError("artifact violates the bounded posting contract")

        matches: list[ArchiveReference] = []
        for entry in self._entries[begin:end]:
            record = self._records[entry.record_index]
            candidate = self.read_tokens(record.record_id, entry.token_offset, REFERENCE_KEY_TOKENS)
            # A fixed-width fingerprint only narrows the range; this exact
            # comparison makes hash collisions fail closed rather than leak a proposal.
            if candidate != query:
                continue
            matches.append(ArchiveReference(record.record_id, entry.token_offset))
            if limit is not None and len(matches) >= limit:
                break
        return tuple(matches)

    lookup = find_references

    @classmethod
    def open(cls, root: str | Path) -> "ImmutableTokenArchive":
        root_path = Path(root)
        if not root_path.is_dir():
            raise ArchiveContractError("archive root must be an existing directory")
        actual_files = {path.name for path in root_path.iterdir()}
        if actual_files != _ARTIFACT_FILES:
            raise ArchiveContractError("archive root must contain exactly the sealed artifact files")

        manifest_path = root_path / _MANIFEST_FILE
        seal_path = root_path / _SEAL_FILE
        manifest_bytes = manifest_path.read_bytes()
        expected_seal = seal_path.read_text(encoding="ascii").strip()
        if len(expected_seal) != 64 or any(character not in "0123456789abcdef" for character in expected_seal):
            raise ArchiveContractError("manifest seal must be a lowercase SHA-256 digest")
        if _raw_sha256(manifest_bytes) != expected_seal:
            raise ArchiveContractError("manifest checksum mismatch")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArchiveContractError("manifest must be valid JSON") from exc
        _validate_manifest(manifest)

        for filename, description in manifest["files"].items():
            path = root_path / filename
            if path.stat().st_size != description["bytes"]:
                raise ArchiveContractError(f"byte size mismatch for {filename}")
            if _file_sha256(path) != description["sha256"]:
                raise ArchiveContractError(f"checksum mismatch for {filename}")

        records = _load_records(root_path / _RECORDS_FILE, manifest)
        entries = _load_index(root_path / _INDEX_FILE, records, manifest)
        _verify_token_stream(root_path / _TOKENS_FILE, records, manifest)
        _verify_index_exactness(root_path / _TOKENS_FILE, records, entries, manifest)
        return cls(root_path, manifest, records, entries)


def build_immutable_archive(
    root: str | Path,
    records: Iterable[TokenRecord],
    *,
    posting_limit: int = REFERENCE_POSTING_LIMIT_DEFAULT,
) -> ImmutableTokenArchive:
    """Build one immutable artifact directory; existing paths are never modified."""
    posting_limit = require_int(posting_limit, "posting_limit", minimum=1)
    root_path = Path(root)
    if root_path.exists():
        raise ArchiveContractError("archive output path already exists; immutable artifacts are never overwritten")
    try:
        normalized_records = tuple(records)
    except TypeError as exc:
        raise ArchiveContractError("records must be an iterable of TokenRecord objects") from exc
    for index, record in enumerate(normalized_records):
        if not isinstance(record, TokenRecord):
            raise ArchiveContractError(f"records[{index}] must be a TokenRecord")
    record_ids = [record.record_id for record in normalized_records]
    if len(set(record_ids)) != len(record_ids):
        raise ArchiveContractError("record_id values must be unique")
    if len(normalized_records) > 0xFFFFFFFF:
        raise ArchiveContractError("archive has too many records for its fixed reference format")

    root_path.mkdir()
    token_path = root_path / _TOKENS_FILE
    index_path = root_path / _INDEX_FILE
    records_path = root_path / _RECORDS_FILE

    descriptors: list[ArchiveRecord] = []
    all_entries: list[_ReferenceEntry] = []
    token_start = 0
    with token_path.open("wb") as token_handle:
        for record_index, record in enumerate(normalized_records):
            if token_start + len(record.tokens) > 0xFFFFFFFF:
                raise ArchiveContractError("archive token offsets exceed the fixed u32 reference format")
            payload = _pack_tokens(record.tokens)
            token_handle.write(payload)
            descriptors.append(
                ArchiveRecord(
                    record_id=record.record_id,
                    start_token=token_start,
                    token_count=len(record.tokens),
                    metadata=dict(record.metadata),
                    token_sha256=_raw_sha256(payload),
                )
            )
            for offset in range(len(record.tokens) - REFERENCE_KEY_TOKENS + 1):
                key = _reference_key(record.tokens[offset : offset + REFERENCE_KEY_TOKENS])
                all_entries.append(_ReferenceEntry(key, record_index, offset))
            token_start += len(record.tokens)

    all_entries.sort(key=lambda entry: (entry.key, entry.record_index, entry.token_offset))
    _validate_posting_bound(all_entries, posting_limit)
    with index_path.open("wb") as index_handle:
        for entry in all_entries:
            index_handle.write(_REFERENCE_STRUCT.pack(entry.key, entry.record_index, entry.token_offset))

    records_payload = {
        "schema_version": RECORDS_SCHEMA_VERSION,
        "records": [
            {
                "record_id": descriptor.record_id,
                "start_token": descriptor.start_token,
                "token_count": descriptor.token_count,
                "metadata": dict(descriptor.metadata),
                "token_sha256": descriptor.token_sha256,
            }
            for descriptor in descriptors
        ],
    }
    records_path.write_text(canonical_json(records_payload) + "\n", encoding="utf-8")

    files = {
        filename: {
            "bytes": (root_path / filename).stat().st_size,
            "sha256": _file_sha256(root_path / filename),
        }
        for filename in (_TOKENS_FILE, _RECORDS_FILE, _INDEX_FILE)
    }
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "token_width_bits": TOKEN_WIDTH_BITS,
        "token_byte_order": TOKEN_BYTE_ORDER,
        "reference_key_tokens": REFERENCE_KEY_TOKENS,
        "reference_key_bytes": REFERENCE_KEY_BYTES,
        "reference_entry_bytes": _REFERENCE_STRUCT.size,
        "posting_limit": posting_limit,
        "record_count": len(descriptors),
        "token_count": token_start,
        "reference_count": len(all_entries),
        "files": files,
    }
    manifest_path = root_path / _MANIFEST_FILE
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    (root_path / _SEAL_FILE).write_text(_raw_sha256(manifest_bytes) + "\n", encoding="ascii")
    return ImmutableTokenArchive.open(root_path)


def open_immutable_archive(root: str | Path) -> ImmutableTokenArchive:
    """Open and fully validate a previously built immutable archive."""
    return ImmutableTokenArchive.open(root)


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, Mapping):
        raise ArchiveContractError("manifest must be an object")
    expected = {
        "schema_version",
        "token_width_bits",
        "token_byte_order",
        "reference_key_tokens",
        "reference_key_bytes",
        "reference_entry_bytes",
        "posting_limit",
        "record_count",
        "token_count",
        "reference_count",
        "files",
    }
    if set(manifest) != expected:
        raise ArchiveContractError("manifest has missing or unknown fields")
    if manifest["schema_version"] != ARCHIVE_SCHEMA_VERSION:
        raise ArchiveContractError("unsupported archive schema version")
    if manifest["token_width_bits"] != TOKEN_WIDTH_BITS or manifest["token_byte_order"] != TOKEN_BYTE_ORDER:
        raise ArchiveContractError("unsupported token stream layout")
    if (
        manifest["reference_key_tokens"] != REFERENCE_KEY_TOKENS
        or manifest["reference_key_bytes"] != REFERENCE_KEY_BYTES
        or manifest["reference_entry_bytes"] != _REFERENCE_STRUCT.size
    ):
        raise ArchiveContractError("unsupported fixed reference index layout")
    require_int(manifest["posting_limit"], "manifest.posting_limit", minimum=1)
    for field in ("record_count", "token_count", "reference_count"):
        require_int(manifest[field], f"manifest.{field}", minimum=0)
    files = manifest["files"]
    if not isinstance(files, Mapping) or set(files) != {_TOKENS_FILE, _RECORDS_FILE, _INDEX_FILE}:
        raise ArchiveContractError("manifest.files must describe every content file exactly once")
    for filename, description in files.items():
        if not isinstance(description, Mapping) or set(description) != {"bytes", "sha256"}:
            raise ArchiveContractError(f"manifest.files.{filename} has invalid fields")
        require_int(description["bytes"], f"manifest.files.{filename}.bytes", minimum=0)
        digest = description["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArchiveContractError(f"manifest.files.{filename}.sha256 must be lowercase SHA-256")


def _load_records(path: Path, manifest: Mapping[str, Any]) -> tuple[ArchiveRecord, ...]:
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveContractError("records file must be valid JSON") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "records"}:
        raise ArchiveContractError("records file has invalid fields")
    if payload["schema_version"] != RECORDS_SCHEMA_VERSION or not isinstance(payload["records"], list):
        raise ArchiveContractError("unsupported records file")
    if len(payload["records"]) != manifest["record_count"]:
        raise ArchiveContractError("record count does not match manifest")

    records: list[ArchiveRecord] = []
    next_start = 0
    ids: set[str] = set()
    for index, item in enumerate(payload["records"]):
        if not isinstance(item, Mapping) or set(item) != {
            "record_id",
            "start_token",
            "token_count",
            "metadata",
            "token_sha256",
        }:
            raise ArchiveContractError(f"records[{index}] has invalid fields")
        record_id = require_string(item["record_id"], f"records[{index}].record_id")
        if record_id in ids:
            raise ArchiveContractError("records file contains duplicate record_id")
        ids.add(record_id)
        start_token = require_int(item["start_token"], f"records[{index}].start_token", minimum=0)
        token_count = require_int(item["token_count"], f"records[{index}].token_count", minimum=0)
        if start_token != next_start:
            raise ArchiveContractError("record token ranges must be contiguous and ordered")
        token_sha256 = item["token_sha256"]
        if not isinstance(token_sha256, str) or len(token_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in token_sha256
        ):
            raise ArchiveContractError(f"records[{index}].token_sha256 must be lowercase SHA-256")
        records.append(
            ArchiveRecord(
                record_id=record_id,
                start_token=start_token,
                token_count=token_count,
                metadata=_normalise_metadata(item["metadata"], f"records[{index}].metadata"),
                token_sha256=token_sha256,
            )
        )
        next_start += token_count
    if next_start != manifest["token_count"]:
        raise ArchiveContractError("record token lengths do not match manifest")
    return tuple(records)


def _load_index(
    path: Path, records: tuple[ArchiveRecord, ...], manifest: Mapping[str, Any]
) -> tuple[_ReferenceEntry, ...]:
    payload = path.read_bytes()
    expected_bytes = manifest["reference_count"] * _REFERENCE_STRUCT.size
    if len(payload) != expected_bytes:
        raise ArchiveContractError("reference index byte length does not match manifest")
    entries: list[_ReferenceEntry] = []
    previous: tuple[bytes, int, int] | None = None
    group_size = 0
    previous_key: bytes | None = None
    for offset in range(0, len(payload), _REFERENCE_STRUCT.size):
        key, record_index, token_offset = _REFERENCE_STRUCT.unpack_from(payload, offset)
        if record_index >= len(records):
            raise ArchiveContractError("reference index points outside the records table")
        record = records[record_index]
        if token_offset + REFERENCE_KEY_TOKENS > record.token_count:
            raise ArchiveContractError("reference index points outside a record")
        current = (key, record_index, token_offset)
        if previous is not None and current <= previous:
            raise ArchiveContractError("reference index must be strictly sorted")
        if key == previous_key:
            group_size += 1
        else:
            group_size = 1
            previous_key = key
        if group_size > manifest["posting_limit"]:
            raise ArchiveContractError("reference index exceeds its posting bound")
        entries.append(_ReferenceEntry(key, record_index, token_offset))
        previous = current
    return tuple(entries)


def _verify_token_stream(path: Path, records: tuple[ArchiveRecord, ...], manifest: Mapping[str, Any]) -> None:
    expected_bytes = manifest["token_count"] * _TOKEN_STRUCT.size
    if path.stat().st_size != expected_bytes:
        raise ArchiveContractError("token stream byte length does not match manifest")
    with path.open("rb") as handle:
        for record in records:
            payload = handle.read(record.token_count * _TOKEN_STRUCT.size)
            if _raw_sha256(payload) != record.token_sha256:
                raise ArchiveContractError(f"token checksum mismatch for record {record.record_id}")
        if handle.read(1):
            raise ArchiveContractError("token stream has trailing bytes")


def _verify_index_exactness(
    token_path: Path,
    records: tuple[ArchiveRecord, ...],
    entries: tuple[_ReferenceEntry, ...],
    manifest: Mapping[str, Any],
) -> None:
    expected: list[_ReferenceEntry] = []
    with token_path.open("rb") as handle:
        for record_index, record in enumerate(records):
            handle.seek(record.start_token * _TOKEN_STRUCT.size)
            payload = handle.read(record.token_count * _TOKEN_STRUCT.size)
            tokens = tuple(struct.unpack(f"<{record.token_count}I", payload)) if record.token_count else ()
            for token_offset in range(record.token_count - REFERENCE_KEY_TOKENS + 1):
                expected.append(
                    _ReferenceEntry(
                        _reference_key(tokens[token_offset : token_offset + REFERENCE_KEY_TOKENS]),
                        record_index,
                        token_offset,
                    )
                )
    expected.sort(key=lambda entry: (entry.key, entry.record_index, entry.token_offset))
    _validate_posting_bound(expected, manifest["posting_limit"])
    if tuple(expected) != entries:
        raise ArchiveContractError("reference index is not the deterministic index of the token streams")


def _validate_posting_bound(entries: list[_ReferenceEntry], posting_limit: int) -> None:
    current_key: bytes | None = None
    count = 0
    for entry in entries:
        if entry.key == current_key:
            count += 1
        else:
            current_key = entry.key
            count = 1
        if count > posting_limit:
            raise ArchiveContractError(
                f"reference key exceeds bounded posting limit {posting_limit}; archive build is rejected"
            )
