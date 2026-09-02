#!/usr/bin/env python3
"""Level-2 iteration-speed loop (NOW.md "Iteration speed rules"): tiny-
model stress test for the MLX disk prompt-cache tier.

Reproduces the Phase M1 scale-bug class -- unbounded eager load-on-start
stall, background-writer-thread MLX-stream crash, load-then-evict file
race -- in well under two minutes: a tiny model (near-instant load and
prefill) with a deliberately small RAM budget, eviction on every insert, a
disk quota small enough to force disk-side eviction too, and several real
process restarts.

Run before every push that touches native/mlx-patches/ or
src/pcss/mlx_adapter.py. MUST run on the M4 host (starts mlx_lm.server;
see NOW.md's execution-host rule) but takes ~1-2 minutes, not hours.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pcss.mlx_adapter import post_chat_completion, wait_ready  # noqa: E402

VENV_PYTHON = REPO_ROOT / ".venv-mlx" / "bin" / "python"
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
HOST = "127.0.0.1"
PORT = 18799
N_DISTINCT_PROMPTS = 12
N_RESTARTS = 3
DISK_BUDGET_BYTES = 150_000_000  # small enough to force disk eviction, big
                                  # enough that the most-recent entries fit


def synthetic_prompt(seed: int, n_words: int) -> str:
    rng = random.Random(seed)
    return " ".join(f"tok{rng.randrange(50000)}" for _ in range(n_words))


def start_server(disk_dir: Path, log_handle) -> subprocess.Popen:
    command = [
        str(VENV_PYTHON), "-m", "mlx_lm.server",
        "--model", MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--prompt-cache-size", "1",  # evicts on every insert
        "--prompt-cache-dir", str(disk_dir),
        "--prompt-cache-disk-bytes", str(DISK_BUDGET_BYTES),
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    wait_ready(f"http://{HOST}:{PORT}", timeout_s=60.0)
    return process


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()  # SIGTERM: exercises the flush-on-shutdown path
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="mlx-stress-"))
    disk_dir = work_dir / "disk-cache"
    disk_dir.mkdir()
    log_path = work_dir / "server.log"
    base_url = f"http://{HOST}:{PORT}"

    print(f"[mlx-stress] work_dir={work_dir}", flush=True)
    log_handle = log_path.open("w")
    process = start_server(disk_dir, log_handle)
    errors: list[str] = []
    restart_every = max(1, N_DISTINCT_PROMPTS // N_RESTARTS)
    try:
        for i in range(N_DISTINCT_PROMPTS):
            if i > 0 and i % restart_every == 0:
                stop_server(process)
                process = start_server(disk_dir, log_handle)
                print(f"[mlx-stress] restarted before prompt {i}", flush=True)
            prompt = synthetic_prompt(seed=i, n_words=200 + i * 40)
            try:
                result = post_chat_completion(
                    base_url, [{"role": "user", "content": prompt}],
                    model=MODEL, max_tokens=5, temperature=0.0, seed=0,
                    logprobs=False, timeout_s=30.0,
                )
                print(
                    f"[mlx-stress] i={i} prompt_tokens={result.prompt_tokens} "
                    f"cached_tokens={result.cached_tokens} wall_s={result.wall_s:.2f}",
                    flush=True,
                )
            except Exception as e:
                errors.append(f"request {i} failed: {e!r}")

        # Replay the most recently processed prompts (now evicted from RAM,
        # size=1, but still likely to survive on disk -- LRU evicts the
        # oldest first, and with a tight budget the earliest, smallest
        # entries are long gone by the end of the loop) to exercise disk
        # lookup on RAM miss. At least one should come back mostly cached
        # from disk -- not just the trivial shared chat-template prefix --
        # or the disk tier isn't actually being exercised.
        best_replay_ratio = 0.0
        replay_indices = range(max(0, N_DISTINCT_PROMPTS - 3), N_DISTINCT_PROMPTS)
        for i in replay_indices:
            prompt = synthetic_prompt(seed=i, n_words=200 + i * 40)
            try:
                result = post_chat_completion(
                    base_url, [{"role": "user", "content": prompt}],
                    model=MODEL, max_tokens=5, temperature=0.0, seed=0,
                    logprobs=False, timeout_s=30.0,
                )
                ratio = (result.cached_tokens or 0) / max(1, result.prompt_tokens or 1)
                best_replay_ratio = max(best_replay_ratio, ratio)
                print(
                    f"[mlx-stress] replay i={i} prompt_tokens={result.prompt_tokens} "
                    f"cached_tokens={result.cached_tokens} wall_s={result.wall_s:.2f}",
                    flush=True,
                )
            except Exception as e:
                errors.append(f"replay {i} failed: {e!r}")
        if best_replay_ratio < 0.5:
            errors.append(
                f"no replay recovered a real disk hit (best cached/prompt "
                f"ratio {best_replay_ratio:.2f}) -- disk lookup on RAM miss "
                "may not be working, not just a config/sizing issue"
            )
    finally:
        stop_server(process)
        log_handle.close()

    log_text = log_path.read_text()
    if "Traceback" in log_text or "RuntimeError" in log_text:
        errors.append(f"server log contains a traceback/RuntimeError -- see {log_path}")

    index_path = disk_dir / "index.json"
    if index_path.exists():
        try:
            idx = json.loads(index_path.read_text())
            total_bytes = sum(r["n_bytes"] for r in idx)
            print(f"[mlx-stress] disk index: {len(idx)} entries, {total_bytes / 1e6:.1f} MB", flush=True)
            if total_bytes > DISK_BUDGET_BYTES * 1.25:
                errors.append(
                    f"disk store {total_bytes / 1e6:.1f} MB exceeds its "
                    f"{DISK_BUDGET_BYTES / 1e6:.0f} MB budget with no real headroom"
                )
            if len(idx) == 0:
                errors.append(
                    "disk index has 0 entries -- demote-on-eviction isn't "
                    "persisting anything (or every entry is larger than the "
                    "disk budget; check prompt sizes vs DISK_BUDGET_BYTES)"
                )
        except (json.JSONDecodeError, KeyError) as e:
            errors.append(f"disk index unreadable/corrupt: {e!r}")
    else:
        errors.append("disk index.json missing after the run")

    if errors:
        print(f"[mlx-stress] FAIL (artifacts kept at {work_dir})")
        for e in errors:
            print(" -", e)
        return 1

    shutil.rmtree(work_dir, ignore_errors=True)
    print("[mlx-stress] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
