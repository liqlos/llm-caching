from pcss.exact_prompt_policy_economics import _derive


def _inputs() -> dict:
    return {
        "manifest": "manifest", "config": "config", "bridge_binary": "bridge",
        "model_artifact": "model", "llama_cpp_library": "library", "archive_manifest": "archive",
    }


def _hit() -> dict:
    return {
        "status": "EXACT_PROMPT_ARCHIVE_HARNESS_POSITIVE", "evidence_class": "LOCAL_MEASURED",
        "input_fingerprints": _inputs(),
        "payload": {"cases": [{"request_id": "hit", "response": {"exact_output_and_state": True, "measurements": [{"serial_before_ns": 200, "serial_after_ns": 200}]}, "archive_measurements": [{"lookup_inclusive_candidate_ns": 100}]}]},
    }


def _edited() -> dict:
    return {
        "status": "CONTROLLED_LIVE_DECODE_INCONCLUSIVE", "evidence_class": "LOCAL_MEASURED",
        "input_fingerprints": _inputs(),
        "payload": {"cases": [{"request_id": "miss", "response": {"exact_output_and_state": True, "measurements": [{"serial_before_ns": 200, "serial_after_ns": 200}]}}]},
    }


def _miss() -> dict:
    return {
        "status": "EXACT_PROMPT_LOOKUP_PROFILE_VALID", "evidence_class": "LOCAL_MEASURED",
        "input_fingerprints": _inputs(),
        "payload": {"expect_hit": False, "net_lookup_and_cursor_ci_ns_per_call": {"upper": 10.0}},
    }


def test_mixed_policy_adds_one_miss_lookup_and_retains_hit_measurement() -> None:
    analysis = _derive(_hit(), _edited(), _miss())

    assert analysis["exact_hit_measurement_rows"] == 1
    assert analysis["edited_miss_measurement_rows"] == 1
    assert analysis["miss_lookup_upper_ns_per_request"] == 10.0
    assert analysis["decision"] == "MIXED_EXACT_PROMPT_POLICY_INCONCLUSIVE"
