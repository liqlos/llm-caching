#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 MANIFEST.json WORKLOAD.json MODEL.gguf OUTPUT_DIRECTORY [CAPTURE_RECEIPT.json]" >&2
  exit 2
}

[[ $# -eq 4 || $# -eq 5 ]] || usage

root="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$1"
workload="$2"
model="$3"
output="$4"
capture_receipt="${5:-}"
server="${PCSS_SERVER:-$root/build/llama.cpp/bin/llama-server}"
python_bin="${PCSS_PYTHON:-$root/.venv/bin/python}"

[[ -f "$manifest" ]] || { echo "manifest is not a file: $manifest" >&2; exit 2; }
[[ -f "$workload" ]] || { echo "workload is not a file: $workload" >&2; exit 2; }
[[ -f "$model" ]] || { echo "model is not a file: $model" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite output: $output" >&2; exit 2; }
[[ -x "$server" ]] || { echo "server is not executable: $server" >&2; exit 2; }
if [[ -n "$capture_receipt" ]]; then
  [[ -f "$capture_receipt" ]] || { echo "capture receipt is not a file: $capture_receipt" >&2; exit 2; }
fi
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

arguments=(
  -m pcss.live_server_benchmark
  --manifest "$manifest"
  --workload "$workload"
  --model "$model"
  --server "$server"
  --output "$output"
)
if [[ -n "$capture_receipt" ]]; then
  arguments+=(--capture-receipt "$capture_receipt")
fi

exec "$python_bin" "${arguments[@]}"
