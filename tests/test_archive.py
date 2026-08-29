from __future__ import annotations

import json
from pathlib import Path

import pytest

import pcss.archive as archive_module
from pcss.archive import (
    ArchiveContractError,
    TokenRecord,
    build_immutable_archive,
    open_immutable_archive,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "archive" / "records-v1.json"


def fixture_records() -> tuple[TokenRecord, ...]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return tuple(
        TokenRecord(
            record_id=item["record_id"],
            tokens=tuple(item["tokens"]),
            metadata=item["metadata"],
        )
        for item in payload
    )


def test_build_verify_cursor_and_exact_references(tmp_path: Path) -> None:
    archive = build_immutable_archive(tmp_path / "fixture-archive", fixture_records())

    assert archive.token_count == 11
    assert archive.reference_count == 5
    assert archive.find_references((3, 4, 5, 6)) == (
        archive_module.ArchiveReference("alpha", 2),
        archive_module.ArchiveReference("beta", 0),
    )
    cursor = archive.open_cursor("alpha", position=1)
    assert cursor.read(3) == (2, 3, 4)
    assert cursor.read(100) == (5, 6)
    assert cursor.remaining == 0
    assert cursor.read(0) == ()
    with pytest.raises(ArchiveContractError, match="record boundary"):
        archive.read_tokens("alpha", 5, 2)

    reopened = open_immutable_archive(tmp_path / "fixture-archive")
    assert reopened.records[0].metadata == {"ordinal": 1, "source": "fixture"}
    accounting = reopened.byte_accounting()
    assert accounting["total"] == sum(value for key, value in accounting.items() if key != "total")


def test_open_rejects_corrupted_token_stream(tmp_path: Path) -> None:
    root = tmp_path / "corrupted"
    build_immutable_archive(root, fixture_records())
    token_path = root / "tokens.u32"
    bytes_before = bytearray(token_path.read_bytes())
    bytes_before[0] ^= 1
    token_path.write_bytes(bytes_before)

    with pytest.raises(ArchiveContractError, match="checksum mismatch"):
        open_immutable_archive(root)


def test_fixed_key_collision_still_requires_exact_token_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive_module, "_reference_key", lambda _: b"x" * 16)
    archive = build_immutable_archive(
        tmp_path / "collisions",
        (
            TokenRecord("first", (1, 2, 3, 4), {"source": "fixture"}),
            TokenRecord("second", (5, 6, 7, 8), {"source": "fixture"}),
        ),
        posting_limit=2,
    )

    assert archive.lookup((1, 2, 3, 4)) == (archive_module.ArchiveReference("first", 0),)
    assert archive.lookup((5, 6, 7, 8)) == (archive_module.ArchiveReference("second", 0),)


def test_builder_rejects_unbounded_postings_and_invalid_u32(tmp_path: Path) -> None:
    repeated = TokenRecord("repeat", (1, 2, 3, 4, 1, 2, 3, 4), {"source": "fixture"})
    with pytest.raises(ArchiveContractError, match="posting limit"):
        build_immutable_archive(tmp_path / "too-many-postings", (repeated,), posting_limit=1)
    with pytest.raises(ArchiveContractError, match="unsigned 32-bit"):
        TokenRecord("bad", (0x1_0000_0000,), {"source": "fixture"})


def test_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "once"
    build_immutable_archive(root, fixture_records())
    with pytest.raises(ArchiveContractError, match="already exists"):
        build_immutable_archive(root, fixture_records())
