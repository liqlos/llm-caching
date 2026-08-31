#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: scripts/verify_model.sh /absolute/model.gguf}"
expected="${PCSS_MODEL_SHA256:-}"

test -f "$model" || { echo "PCSS model is not a readable file: $model" >&2; exit 2; }

if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$model" | awk '{print $1}')"
else
  actual="$(shasum -a 256 "$model" | awk '{print $1}')"
fi

if [[ -n "$expected" && "$actual" != "$expected" ]]; then
  echo "PCSS model SHA-256 mismatch: expected $expected, got $actual" >&2
  exit 3
fi

printf '%s\n' "$actual"
