from __future__ import annotations

import pytest

from pcss.canonical import ContractError
from pcss.provider_profiler import _profile_mode


@pytest.mark.parametrize("mode", ["no-op", "always-miss", "shadow-hit"])
def test_provider_profile_uses_whole_block_clusters(mode: str) -> None:
    result = _profile_mode(mode, blocks=3, calls_per_block=5, warmup_calls=1)

    assert result["mode"] == mode
    assert len(result["raw_blocks"]) == 3
    assert result["net_overhead_ci_ns_per_call"]["block_clusters"] == 3
    assert result["provider_mean_ns_per_call"] > 0


def test_provider_profile_bounds_repetitions() -> None:
    with pytest.raises(ContractError):
        _profile_mode("no-op", blocks=1, calls_per_block=1, warmup_calls=0)
    with pytest.raises(ContractError, match="unknown"):
        _profile_mode("unsupported", blocks=2, calls_per_block=1, warmup_calls=0)
