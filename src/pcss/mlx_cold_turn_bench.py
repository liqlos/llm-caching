#!/usr/bin/env python3
"""Level-3 iteration-speed loop (NOW.md "Iteration speed rules"): cold-
turn TTFT microbench against a disk-store fixture, using the diagnostic
``/debug/simulate_restart`` endpoint instead of real process restarts.

Costs one request plus one near-instant simulated restart per restart
point (~24 for the full pre-registered schedule) instead of 24 real
process restarts (~15-20s model reload each). Diagnostic config -- never
gate evidence on its own (NOW.md); for screening profit-ladder levers and
restore-path work. The pre-registered Gate G1-MLX receipt (real restarts,
real model, all three arms) is the actual evidence.

Prerequisite: a disk-store *fixture* -- the ``--prompt-cache-dir``
directory from a completed, clean ``ram_disk`` gate run against the same
frozen workload. Cloned per invocation via ``cp -c`` (APFS clonefile:
copy-on-write, effectively instant) so iterating never mutates the
fixture or needs to rebuild it.

The frozen workload's "prompts" are already chat-shaped (system preamble
+ growing transcript baked into one string per turn, see
PHASE0_WORKLOAD_DEFINITION_MLX_V1.md), so a restart-point turn's prompt is
self-contained -- there is no need to replay every intervening turn to
reach it; only the restart-boundary turns are sent here.

MUST run on the M4 host (starts mlx_lm.server; NOW.md's execution-host
rule), but the whole run is minutes, not hours.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pcss.mlx_adapter import post_chat_completion, wait_ready  # noqa: E402

VENV_PYTHON = REPO_ROOT / ".venv-mlx" / "bin" / "python"
MODEL = "mlx-community/Qwen3-8B-4bit"
HOST = "127.0.0.1"
PORT = 18790
MID_SESSION_RESTART_AFTER_TURN = 9
MID_SESSION_MIN_TURNS = 15


def default_restart_points(sessions: list[list[str]]) -> set[tuple[int, int]]:
    """Mirrors PHASE_M1_GATE_G1_MLX_DEFINITION.md's mechanical schedule."""
    points: set[tuple[int, int]] = set()
    for session_index in range(1, len(sessions)):
        points.add((session_index, 0))
    for session_index, prompts in enumerate(sessions):
        if len(prompts) > MID_SESSION_MIN_TURNS:
            points.add((session_index, MID_SESSION_RESTART_AFTER_TURN + 1))
    return points


def clone_fixture(fixture_dir: Path, work_dir: Path) -> Path:
    """APFS clonefile copy: near-instant, copy-on-write."""
    clone_dir = work_dir / "disk-cache-clone"
    subprocess.run(["cp", "-c", "-R", str(fixture_dir), str(clone_dir)], check=True)
    return clone_dir


def start_server(disk_dir: Path, disk_bytes: int, log_handle) -> subprocess.Popen:
    command = [
        str(VENV_PYTHON), "-m", "mlx_lm.server",
        "--model", MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--prompt-cache-bytes", "4294967296",
        "--prompt-cache-dir", str(disk_dir),
        "--prompt-cache-disk-bytes", str(disk_bytes),
    ]
    process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    wait_ready(f"http://{HOST}:{PORT}", timeout_s=120.0)
    return process


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def simulate_restart(base_url: str) -> None:
    req = urllib.request.Request(f"{base_url}/debug/simulate_restart", data=b"{}", method="POST")
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        resp.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True, help="Completed ram_disk run's --prompt-cache-dir")
    parser.add_argument("--workload", required=True, help="corpus_v1_workload.json path")
    parser.add_argument("--disk-bytes", type=int, default=64_424_509_440)
    parser.add_argument("--restart-points", default=None, help='JSON file: [[session, turn], ...]; default = the pre-registered G1-MLX schedule')
    parser.add_argument("--out", default=None, help="Write results JSON here")
    args = parser.parse_args()

    workload = json.loads(Path(args.workload).read_text())
    sessions = [s["prompts"] for s in workload["sessions"]]

    if args.restart_points:
        restart_points = {tuple(p) for p in json.loads(Path(args.restart_points).read_text())}
    else:
        restart_points = default_restart_points(sessions)

    work_dir = Path(tempfile.mkdtemp(prefix="mlx-cold-turn-"))
    disk_dir = clone_fixture(Path(args.fixture_dir), work_dir)
    log_path = work_dir / "server.log"
    base_url = f"http://{HOST}:{PORT}"

    print(f"[cold-turn] fixture cloned to {disk_dir}, {len(restart_points)} restart points", flush=True)
    log_handle = log_path.open("w")
    process = start_server(disk_dir, args.disk_bytes, log_handle)
    results: list[dict] = []
    try:
        for session_index, prompts in enumerate(sessions):
            for turn_index, prompt in enumerate(prompts):
                if (session_index, turn_index) not in restart_points:
                    continue
                simulate_restart(base_url)
                result = post_chat_completion(
                    base_url, [{"role": "user", "content": prompt}],
                    model=MODEL, max_tokens=1, temperature=0.0, seed=0,
                    logprobs=False, timeout_s=120.0,
                )
                row = {
                    "session": session_index,
                    "turn": turn_index,
                    "prompt_tokens": result.prompt_tokens,
                    "cached_tokens": result.cached_tokens,
                    "ttft_s": result.wall_s,
                }
                results.append(row)
                print(
                    f"[cold-turn] s={session_index} t={turn_index} "
                    f"prompt_tokens={result.prompt_tokens} cached_tokens={result.cached_tokens} "
                    f"ttft_s={result.wall_s:.3f}",
                    flush=True,
                )
    finally:
        stop_server(process)
        log_handle.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
