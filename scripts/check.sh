#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
python_bin="${PCSS_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi
ruff_bin="${PCSS_RUFF:-ruff}"
"$ruff_bin" check src tests
"$python_bin" -m pytest -q
