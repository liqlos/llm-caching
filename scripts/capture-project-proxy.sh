#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CAPTURE.jsonl --origin {ordinary_local_llama_server|agent_assisted_local_llama_server} [--current-source-sidecar SIDECAR.jsonl] [extra proxy arguments]" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PCSS_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

exec "$python_bin" -m pcss.project_capture_proxy \
  --capture-log "$1" "${@:2}"
