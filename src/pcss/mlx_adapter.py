"""HTTP adapter for ``mlx_lm.server``'s OpenAI-compatible chat endpoint.

Captures the fields the MLX line's paired-benchmark method needs (see
``NOW.md`` Phase M0 and ``docs/MLX_CACHE.md``): the reused-prefix length
(``usage.prompt_tokens_details.cached_tokens``, mlx-lm's equivalent of
llama-server's ``cache_n``), ``prompt_tokens``, wall-clock, and the
generated token IDs. Token IDs are not echoed on the message by default;
pass ``logprobs=True`` to request them, since mlx_lm.server exposes them as
``choices[0].logprobs.content[*].id``.

Standard library only, matching the rest of the corpus/replay tooling under
``artifacts/ssd-redesign``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    wall_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    token_ids: tuple[int, ...]
    finish_reason: str | None
    system_fingerprint: str | None
    raw: Mapping[str, Any]


def wait_ready(base_url: str, *, timeout_s: float = 120.0, poll_interval_s: float = 0.3) -> None:
    """Block until ``base_url``'s ``/v1/models`` responds 200, or raise."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError) as error:
            last_error = error
        time.sleep(poll_interval_s)
    raise RuntimeError(f"mlx_lm.server not ready within {timeout_s}s: {last_error}")


def post_chat_completion(
    base_url: str,
    messages: Sequence[Mapping[str, str]],
    *,
    model: str = "default",
    max_tokens: int = 48,
    temperature: float = 0.0,
    seed: int | None = 0,
    logprobs: bool = True,
    extra_body: Mapping[str, Any] | None = None,
    timeout_s: float = 300.0,
) -> ChatTurnResult:
    """POST one ``/v1/chat/completions`` turn and capture the paired-benchmark fields."""
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "logprobs": logprobs,
    }
    if seed is not None:
        body["seed"] = seed
    if extra_body:
        body.update(extra_body)

    encoded = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic_ns()
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw_bytes = response.read()
    wall_s = (time.monotonic_ns() - started) / 1e9
    doc = json.loads(raw_bytes)

    usage = doc.get("usage") or {}
    prompt_tokens_details = usage.get("prompt_tokens_details") or {}
    choices = doc.get("choices") or [{}]
    choice = choices[0] if choices else {}
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []
    token_ids = tuple(entry["id"] for entry in logprobs_content if "id" in entry)

    return ChatTurnResult(
        wall_s=wall_s,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=prompt_tokens_details.get("cached_tokens"),
        token_ids=token_ids,
        finish_reason=choice.get("finish_reason"),
        system_fingerprint=doc.get("system_fingerprint"),
        raw=doc,
    )
