from pcss.exact_prompt_economics import _derive
from pcss.canonical import ContractError
import pytest


def _live() -> dict:
    return {
        "status": "CONTROLLED_LIVE_DECODE_POSITIVE",
        "evidence_class": "LOCAL_MEASURED",
        "input_fingerprints": {"manifest": "manifest", "trace": "trace"},
        "payload": {
            "cases": [{
                "request_id": "request",
                "response": {"exact_output_and_state": True, "measurements": [{
                    "serial_before_ns": 200, "serial_after_ns": 200, "candidate_decode_ns": 100,
                    "verification_rounds": 2,
                }]},
            }],
        },
    }


def _lookup() -> dict:
    return {
        "status": "EXACT_PROMPT_LOOKUP_PROFILE_VALID",
        "evidence_class": "LOCAL_MEASURED",
        "input_fingerprints": {"manifest": "manifest", "trace": "trace", "archive_manifest": "archive"},
        "payload": {"net_lookup_and_cursor_ci_ns_per_call": {"upper": 10.0}},
    }


def _build() -> dict:
    return {
        "status": "EXACT_PROMPT_ARCHIVE_BUILT",
        "evidence_class": "SIMULATOR",
        "payload": {"archive_manifest_sha256": "archive"},
    }


def test_economics_adds_a_conservative_lookup_for_every_round() -> None:
    analysis = _derive(_live(), _lookup(), _build())

    assert analysis["lookup_upper_ns_per_verification_round"] == 10.0
    assert analysis["added_lookup_ns_across_all_measurements"] == 20.0
    assert analysis["decision"] == "COMPOSED_EXACT_PROMPT_DECODE_POSITIVE"
    assert analysis["lookup_inclusive_paired_serial_over_candidate_ratio"]["estimate"] > 1.0


def test_economics_rejects_mismatched_trace_receipts() -> None:
    lookup = _lookup()
    lookup["input_fingerprints"]["trace"] = "other"

    with pytest.raises(ContractError, match="disagree on trace"):
        _derive(_live(), lookup, _build())
