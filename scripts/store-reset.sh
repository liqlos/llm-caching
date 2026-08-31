#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd -P)"
store="${PCSS_STORE:?set PCSS_STORE to the exact PCSS store directory}"
confirmation="${PCSS_CONFIRM_RESET:-}"

test "$confirmation" = "move-store-to-backup" || {
  echo "set PCSS_CONFIRM_RESET=move-store-to-backup to authorize the recoverable reset" >&2
  exit 2
}
test -d "$store" || { echo "PCSS store directory does not exist: $store" >&2; exit 2; }

store_abs="$(cd "$store" && pwd -P)"
if [[ "$store_abs" = "/" || "$store_abs" = "$root" ]]; then
  echo "refusing broad PCSS reset target: $store_abs" >&2
  exit 3
fi

backup="${store_abs}.reset.$(date -u +%Y%m%dT%H%M%SZ).$$"
mv "$store_abs" "$backup"
printf 'moved=%s\nbackup=%s\nrecoverable=true\n' "$store_abs" "$backup"
