#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
server="${PCSS_SERVER:?set PCSS_SERVER to the llama-server executable}"
model="${PCSS_MODEL:?set PCSS_MODEL to the GGUF model}"
model_sha256="${PCSS_MODEL_SHA256:?set PCSS_MODEL_SHA256 to the expected model SHA-256}"
patch_sha256="${PCSS_PATCH_SHA256:?set PCSS_PATCH_SHA256 to the applied native patch SHA-256}"
output="${PCSS_RESULT:?set PCSS_RESULT to a new JSON result path}"
python_bin="${PCSS_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

exec env PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m pcss.live_server_e2e \
  --server "$server" \
  --model "$model" \
  --model-sha256 "$model_sha256" \
  --patch-sha256 "$patch_sha256" \
  --output "$output" \
  "$@"
