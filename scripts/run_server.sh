#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
model="${PCSS_MODEL:?set PCSS_MODEL to an existing GGUF model path}"
store="${PCSS_STORE:-$root/.pcss-store}"
port="${PCSS_PORT:-8080}"

test -f "$model" || { echo "PCSS_MODEL is not a readable GGUF: $model" >&2; exit 2; }
model_sha256="$("$root/scripts/verify_model.sh" "$model")"
printf 'PCSS model SHA-256: %s\n' "$model_sha256" >&2
if [[ $# -gt 0 && -z "${PCSS_FINGERPRINT:-}" ]]; then
  echo "set PCSS_FINGERPRINT when passing extra server options so template/thinking changes cannot reuse a store" >&2
  exit 2
fi
fingerprint="${PCSS_FINGERPRINT:-sha256:$model_sha256|template=server-default|thinking=server-default}"
namespace="${PCSS_NAMESPACE:-local-uid-$(id -u)}"
mkdir -p "$store"

server="$root/build/llama.cpp/bin/llama-server"
test -x "$server" || bash "$root/scripts/build.sh"

exec "$server" \
  --model "$model" \
  --port "$port" \
  --spec-type pcss-cache \
  --pcss-store "$store" \
  --pcss-fingerprint "$fingerprint" \
  --pcss-namespace "$namespace" \
  --pcss-max-disk-gib "${PCSS_MAX_DISK_GIB:-200}" \
  --pcss-ram-mib "${PCSS_RAM_MIB:-256}" \
  --metrics \
  "$@"
