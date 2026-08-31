#!/usr/bin/env bash
set -euo pipefail

store="${PCSS_STORE:?set PCSS_STORE to the exact PCSS store directory}"
test -d "$store" || { echo "PCSS store directory does not exist: $store" >&2; exit 2; }

printf 'store=%s\n' "$(cd "$store" && pwd -P)"
for name in records.pcss manifest.pcss records.pcss.prev manifest.pcss.prev; do
  path="$store/$name"
  if [[ -f "$path" ]]; then
    bytes="$(wc -c < "$path" | tr -d ' ')"
    if command -v sha256sum >/dev/null 2>&1; then
      digest="$(sha256sum "$path" | awk '{print $1}')"
    else
      digest="$(shasum -a 256 "$path" | awk '{print $1}')"
    fi
    printf '%s bytes=%s sha256=%s\n' "$name" "$bytes" "$digest"
  fi
done
