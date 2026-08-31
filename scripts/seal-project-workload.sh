#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 && $# -ne 4 ]]; then
  echo "usage: $0 CAPTURE.jsonl WORKLOAD.json RECEIPT.json [CURRENT-SOURCE-SIDECAR.jsonl]" >&2
  exit 2
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PCSS_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

if [[ $# -eq 4 ]]; then
  exec "$python_bin" -m pcss.project_workload_capture \
    --input "$1" --workload "$2" --receipt "$3" --source-sidecar "$4"
fi

exec "$python_bin" -m pcss.project_workload_capture \
  --input "$1" --workload "$2" --receipt "$3"
