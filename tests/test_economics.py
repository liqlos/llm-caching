from __future__ import annotations

import pytest

from pcss.canonical import ContractError
from pcss.economics import _derive


def _profile() -> dict:
    return {
        "evidence_class": "LOCAL_MEASURED",
        "status": "PROFILE_VALID",
        "git_revision": "a" * 40,
        "payload": {
            "host_identity": {"fingerprint": "host"},
            "raw_profile": {
                "profiles": [
                    {
                        "k": 2,
                        "measurements": [
                            {"serial_before_ns": 200, "serial_after_ns": 202, "batch_verify_ns": 100},
                            {"serial_before_ns": 204, "serial_after_ns": 206, "batch_verify_ns": 101},
                        ],
                    }
                ]
            },
        },
    }


def _verifier() -> dict:
    return {
        "evidence_class": "LOCAL_MEASURED",
        "status": "VERIFIED",
        "git_revision": "a" * 40,
        "payload": {"execution_class": "TARGET_BACKEND"},
    }


def _provider() -> dict:
    return {
        "evidence_class": "LOCAL_MEASURED",
        "status": "PROVIDER_PROFILE_VALID",
        "payload": {
            "kind": "provider_abi_microprofile",
            "host_identity": {"fingerprint": "host"},
            "profiles": [
                {"mode": "always-miss", "net_overhead_ci_ns_per_call": {"upper": 1.0}},
            ],
        },
    }


def test_economics_preserves_batch_roofline_but_kills_serial_canonical_commit() -> None:
    result = _derive(_profile(), _verifier(), _provider())

    assert result["decision"] == "NO_GO_CURRENT_SERIAL_CANONICAL_COMMIT"
    row = result["rows"][0]
    assert row["batch_only_perfect_draft_ratio_ci"]["lower"] > 1.9
    assert row["current_serial_canonical_commit_ratio_ci"]["upper"] < 1.0
    assert "profitable_k" in result["not_claimed"]


def test_economics_rejects_unverified_or_cross_revision_inputs() -> None:
    verifier = _verifier()
    verifier["status"] = "FAILED"
    with pytest.raises(ContractError, match="not verified"):
        _derive(_profile(), verifier, _provider())

    verifier = _verifier()
    verifier["git_revision"] = "b" * 40
    with pytest.raises(ContractError, match="different revisions"):
        _derive(_profile(), verifier, _provider())
