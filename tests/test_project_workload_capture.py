import json
import hashlib

import pytest

from pcss.project_capture_proxy import PreForwardCaptureLog
from pcss.project_workload_capture import (
    AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN,
    CaptureContractError,
    _canonical_json,
    seal_capture,
    seal_paths,
    sidecar_hash_payload,
)


def event(index: int, episode: str, *, prompt: str | None = None) -> dict:
    value = {
        "schema_version": "pcss.project-workload-capture-event/v1",
        "event_id": f"00000000-0000-4000-8001-{index:012d}",
        "episode_id": episode,
        "capture_index": index,
        "captured_at": f"2026-08-31T00:{index:02d}:00Z",
        "origin": "ordinary_local_llama_server",
        "capture_provenance": "local_forward_proxy_pre_forward/v1",
        "capture_tool_version": "pcss-project-capture-proxy/1",
        "previous_event_sha256": None,
        "endpoint": "/v1/completions",
        "body": {"prompt": prompt or f"ordinary request {index}"},
    }
    value["event_sha256"] = hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return value


def chain(rows: list[dict]) -> list[dict]:
    previous = None
    for value in rows:
        value["previous_event_sha256"] = previous
        value["event_sha256"] = hashlib.sha256(
            (json.dumps({key: child for key, child in value.items() if key != "event_sha256"}, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        previous = value["event_sha256"]
    return rows


def current_source(tokens: list[int] | None = None) -> dict:
    visible_tokens = [7, 11, 13] if tokens is None else tokens
    return {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "private-source-compatibility",
        "source_id": "sha256:" + hashlib.sha256(_canonical_json(visible_tokens)).hexdigest(),
        "visible_token_ids": visible_tokens,
    }


def test_seal_selects_first_events_per_episode_in_global_order() -> None:
    events = [event(1, "00000000-0000-4000-8000-00000000000a"), event(2, "00000000-0000-4000-8000-00000000000b")]
    events.extend([event(3, "00000000-0000-4000-8000-00000000000a"), event(4, "00000000-0000-4000-8000-00000000000b"), event(5, "00000000-0000-4000-8000-00000000000a")])
    chain(events)
    workload, receipt = seal_capture(events, episodes=2, per_episode=2)
    assert [row["request_id"] for row in workload["requests"]] == [
        "00000000-0000-4000-8001-000000000001",
        "00000000-0000-4000-8001-000000000002",
        "00000000-0000-4000-8001-000000000003",
        "00000000-0000-4000-8001-000000000004",
    ]
    assert receipt["episodes"] == ["00000000-0000-4000-8000-00000000000a", "00000000-0000-4000-8000-00000000000b"]
    assert receipt["origin"] == "ordinary_local_llama_server"
    assert receipt["request_episode_map"] == {
        "00000000-0000-4000-8001-000000000001": "00000000-0000-4000-8000-00000000000a",
        "00000000-0000-4000-8001-000000000002": "00000000-0000-4000-8000-00000000000b",
        "00000000-0000-4000-8001-000000000003": "00000000-0000-4000-8000-00000000000a",
        "00000000-0000-4000-8001-000000000004": "00000000-0000-4000-8000-00000000000b",
    }
    assert set(receipt["request_episode_map"]) == {row["request_id"] for row in workload["requests"]}
    assert list(receipt["request_episode_map"]) == sorted(receipt["request_episode_map"])
    assert receipt["selection_rule"] == (
        "first visible pre-forward ordinary local llama-server requests "
        "from first captured episodes, preserved chronologically"
    )
    assert receipt["contains_responses"] is False


def test_seal_accepts_one_declared_agent_assisted_origin() -> None:
    events = [
        event(index, "00000000-0000-4000-8000-00000000000a")
        for index in (1, 2)
    ]
    for value in events:
        value["origin"] = AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN
    chain(events)

    _, receipt = seal_capture(events, episodes=1, per_episode=2)

    assert receipt["origin"] == AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN
    assert receipt["selection_rule"] == (
        "first visible pre-forward agent-assisted local llama-server requests "
        "from first captured episodes, preserved chronologically"
    )


def test_seal_rejects_a_mixed_origin_capture_chain() -> None:
    events = [
        event(1, "00000000-0000-4000-8000-00000000000a"),
        event(2, "00000000-0000-4000-8000-00000000000a"),
    ]
    events[1]["origin"] = AGENT_ASSISTED_LOCAL_LLAMA_SERVER_ORIGIN
    chain(events)
    with pytest.raises(CaptureContractError, match="exactly one declared origin"):
        seal_capture(events, episodes=1, per_episode=2)


def test_seal_rejects_response_and_future_fields() -> None:
    events = [event(1, "00000000-0000-4000-8000-00000000000a"), event(2, "00000000-0000-4000-8000-00000000000a")]
    events[1]["body"]["expected_output"] = "forbidden"
    chain(events)
    with pytest.raises(CaptureContractError, match="forbidden"):
        seal_capture(events, episodes=1, per_episode=2)


def test_seal_rejects_request_scoped_current_source_material() -> None:
    events = [
        event(1, "00000000-0000-4000-8000-00000000000a"),
        event(2, "00000000-0000-4000-8000-00000000000a"),
    ]
    events[0]["body"]["pcss_current_source"] = {
        "schema_version": "pcss.current-source/v1",
        "compatibility_fingerprint": "private-source-compatibility",
        "source_id": "sha256:" + "a" * 64,
        "visible_token_ids": [7, 11, 13],
    }
    chain(events)

    with pytest.raises(CaptureContractError, match="request-scoped current-source"):
        seal_capture(events, episodes=1, per_episode=2)


def test_seal_requires_complete_chronological_episodes() -> None:
    with pytest.raises(CaptureContractError, match="strictly chronological"):
        seal_capture(chain([event(2, "00000000-0000-4000-8000-00000000000a"), event(1, "00000000-0000-4000-8000-00000000000a")]), episodes=1, per_episode=2)


def test_paths_are_immutable_and_never_overwritten(tmp_path) -> None:
    source = tmp_path / "capture.jsonl"
    rows = [event(index, "00000000-0000-4000-8000-00000000000a") for index in (1, 2)]
    chain(rows)
    source.write_text("\n".join(json.dumps(value) for value in rows))
    workload = tmp_path / "workload.json"
    receipt = tmp_path / "receipt.json"
    sealed = seal_paths(source, workload, receipt, episodes=1, per_episode=2)
    assert sealed["selected_event_count"] == 2
    assert workload.stat().st_mode & 0o777 == 0o444
    assert receipt.stat().st_mode & 0o777 == 0o444
    assert json.loads(receipt.read_text())["request_episode_map"] == sealed["request_episode_map"]
    with pytest.raises(CaptureContractError, match="overwrite"):
        seal_paths(source, workload, receipt, episodes=1, per_episode=2)


def test_source_sidecar_is_required_and_binds_the_selected_private_workload(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    source = current_source()
    log.append("/v1/completions", {"prompt": "ordinary", "pcss_current_source": source})
    log.close()
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]

    with pytest.raises(CaptureContractError, match="requires a private source sidecar"):
        seal_capture(ledger_rows, episodes=1, per_episode=1)

    workload_path = tmp_path / "private-workload.json"
    receipt_path = tmp_path / "receipt.json"
    sidecar_path.chmod(0o644)
    with pytest.raises(CaptureContractError, match="group/world accessible"):
        seal_paths(
            ledger_path,
            workload_path,
            receipt_path,
            source_sidecar_path=sidecar_path,
            episodes=1,
            per_episode=1,
        )
    sidecar_path.chmod(0o600)
    receipt = seal_paths(
        ledger_path,
        workload_path,
        receipt_path,
        source_sidecar_path=sidecar_path,
        episodes=1,
        per_episode=1,
    )

    assert workload_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert json.loads(workload_path.read_text())["requests"][0]["body"]["pcss_current_source"] == source
    receipt_text = receipt_path.read_text()
    assert receipt["source_attachment_count"] == 1
    assert source["source_id"] not in receipt_text
    assert source["compatibility_fingerprint"] not in receipt_text
    assert "visible_token_ids" not in receipt_text


def test_source_sidecar_must_bind_its_ledger_event_exactly(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    log.append("/v1/completions", {"prompt": "ordinary", "pcss_current_source": current_source()})
    log.close()
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    sidecar_rows = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    sidecar_rows[0]["ledger_event_sha256"] = "0" * 64
    sidecar_rows[0]["sidecar_event_sha256"] = hashlib.sha256(
        _canonical_json(sidecar_hash_payload(sidecar_rows[0]))
    ).hexdigest()

    with pytest.raises(CaptureContractError, match="does not bind its public ledger event"):
        seal_capture(
            ledger_rows,
            source_sidecar_events=sidecar_rows,
            episodes=1,
            per_episode=1,
        )


def test_v2_capture_without_a_selected_source_attachment_fails_closed(tmp_path) -> None:
    ledger_path = tmp_path / "capture.jsonl"
    sidecar_path = tmp_path / "source-sidecar.jsonl"
    log = PreForwardCaptureLog(ledger_path, current_source_sidecar=sidecar_path)
    log.append("/v1/completions", {"prompt": "ordinary without source"})
    log.close()

    with pytest.raises(CaptureContractError, match="v2 capture selection"):
        seal_capture(
            [json.loads(line) for line in ledger_path.read_text().splitlines()],
            source_sidecar_events=[],
            episodes=1,
            per_episode=1,
        )
