from __future__ import annotations

import pytest

from pcss.canonical import ContractError
from pcss.llamacpp_integrated_workload import _PlanBoundRequest, run_integrated_workload


def test_integrated_workload_has_no_static_benchmark_fallback(tmp_path) -> None:
    """No invalid input may reach a target runner or silently use static timing."""
    with pytest.raises(ContractError, match="unable to load integrated run plan"):
        run_integrated_workload(
            plan_path=tmp_path / "plan.json",
            trace_path=tmp_path / "trace.jsonl",
            archive_path=tmp_path / "archive",
            manifest_path=tmp_path / "manifest.json",
            config_path=tmp_path / "config.json",
        )


def test_plan_bound_request_requires_begin_feedback_and_end() -> None:
    class Cursor:
        def read(self, count):
            assert count == 2
            return (7, 8)

    class Archive:
        def open_cursor(self, record_id, *, position):
            assert (record_id, position) == ("source", 3)
            return Cursor()

    request = _PlanBoundRequest(Archive(), "source", 3, 2)
    with pytest.raises(ContractError, match="before begin"):
        request.propose_or_abstain()
    request.begin_request()
    assert request.propose_or_abstain() == (7, 8)
    with pytest.raises(ContractError, match="before feedback"):
        request.end_request()
    request.feedback((7,))
    request.end_request()
