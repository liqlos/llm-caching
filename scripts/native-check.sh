#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bash "$root/scripts/build.sh"
"$root/build/llama.cpp/bin/test-pcss-store"
test -x "$root/build/llama.cpp/bin/pcss-store-tool"
"$root/build/llama.cpp/bin/llama-server" --help | rg --quiet -- '--spec-type.*pcss-cache'
"$root/build/llama.cpp/bin/llama-server" --help | rg --quiet -- '--pcss-store'
