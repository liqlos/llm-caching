from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from pcss.mlx_adapter import post_chat_completion, wait_ready

_LIVE_RESPONSE_WITH_LOGPROBS = {
    "id": "chatcmpl-5083c47f-b1d3-45df-9c93-478227629980",
    "system_fingerprint": "0.32.0-0.32.2-macOS-27.0-arm64-arm-64bit-Mach-O-applegpu_g16s",
    "object": "chat.completion",
    "model": "mlx-community/Qwen3-8B-4bit",
    "created": 1788291258,
    "choices": [
        {
            "index": 0,
            "finish_reason": "length",
            "logprobs": {
                "content": [
                    {"id": 151667, "logprob": 0.0},
                    {"id": 198, "logprob": 0.0},
                    {"id": 32313, "logprob": 0.0},
                ]
            },
            "message": {"role": "assistant", "content": None, "reasoning": "..."},
        }
    ],
    "usage": {
        "prompt_tokens": 14,
        "completion_tokens": 3,
        "total_tokens": 17,
        "prompt_tokens_details": {"cached_tokens": 13},
    },
}


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def test_post_chat_completion_captures_paired_benchmark_fields() -> None:
    with patch("pcss.mlx_adapter.urllib.request.urlopen", return_value=_FakeResponse(_LIVE_RESPONSE_WITH_LOGPROBS)):
        result = post_chat_completion(
            "http://127.0.0.1:8712",
            [{"role": "user", "content": "Say the word banana once."}],
            model="mlx-community/Qwen3-8B-4bit",
            max_tokens=3,
        )

    assert result.prompt_tokens == 14
    assert result.completion_tokens == 3
    assert result.cached_tokens == 13
    assert result.token_ids == (151667, 198, 32313)
    assert result.finish_reason == "length"
    assert result.system_fingerprint == "0.32.0-0.32.2-macOS-27.0-arm64-arm-64bit-Mach-O-applegpu_g16s"
    assert result.wall_s >= 0.0


def test_post_chat_completion_without_logprobs_yields_empty_token_ids() -> None:
    payload = json.loads(json.dumps(_LIVE_RESPONSE_WITH_LOGPROBS))
    del payload["choices"][0]["logprobs"]
    with patch("pcss.mlx_adapter.urllib.request.urlopen", return_value=_FakeResponse(payload)):
        result = post_chat_completion(
            "http://127.0.0.1:8712",
            [{"role": "user", "content": "hi"}],
            logprobs=False,
        )
    assert result.token_ids == ()
    assert result.cached_tokens == 13


def test_wait_ready_returns_once_models_endpoint_responds() -> None:
    with patch("pcss.mlx_adapter.urllib.request.urlopen", return_value=_FakeResponse({"object": "list", "data": []})):
        wait_ready("http://127.0.0.1:8712", timeout_s=1.0)


def test_wait_ready_raises_when_server_never_comes_up() -> None:
    import urllib.error

    with patch("pcss.mlx_adapter.urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        try:
            wait_ready("http://127.0.0.1:8712", timeout_s=0.5, poll_interval_s=0.1)
        except RuntimeError as error:
            assert "not ready" in str(error)
        else:
            raise AssertionError("expected RuntimeError")
