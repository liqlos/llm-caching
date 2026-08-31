#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
store="${PCSS_STORE:?set PCSS_STORE to the exact PCSS store directory}"
namespace="${PCSS_NAMESPACE:-local-uid-$(id -u)}"
tool="$root/build/llama.cpp/bin/pcss-store-tool"

if [[ -n "${PCSS_FINGERPRINT:-}" ]]; then
  fingerprint="$PCSS_FINGERPRINT"
else
  model="${PCSS_MODEL:?set PCSS_MODEL or the exact PCSS_FINGERPRINT used by the server}"
  model_sha256="$("$root/scripts/verify_model.sh" "$model")"
  fingerprint="sha256:$model_sha256|template=server-default|thinking=server-default"
fi

test -d "$store" || { echo "PCSS store directory does not exist: $store" >&2; exit 2; }
test -x "$tool" || bash "$root/scripts/build.sh"

exec "$tool" compact \
  --store "$store" \
  --namespace "$namespace" \
  --fingerprint "$fingerprint" \
  --max-disk-gib "${PCSS_MAX_DISK_GIB:-200}" \
  --ram-mib "${PCSS_RAM_MIB:-256}"
