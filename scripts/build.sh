#!/usr/bin/env bash
set -euo pipefail

# Build the pinned llama.cpp submodule.  Never substitute a Homebrew install:
# the submodule SHA is the product's native ABI and server source.
root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="$root/third_party/llama.cpp"
build_root="${PCSS_BUILD_DIR:-$root/build/llama.cpp}"
patch_root="$root/native/patches"

test -f "$source_root/CMakeLists.txt" || {
  echo "missing pinned llama.cpp submodule; run: git submodule update --init --recursive" >&2
  exit 2
}

# A clean pinned submodule receives the reviewed local delta exactly once.
# Refuse a different dirty upstream tree rather than silently mixing changes.
if git -C "$source_root" diff --no-ext-diff --quiet; then
  for patch in "$patch_root"/*.patch; do
    git -C "$source_root" apply "$patch"
  done
else
  # A dirty source tree is accepted only when it is exactly explained by the
  # maintained patch series; never mix an arbitrary upstream edit into a build.
  for patch in "$patch_root"/*.patch; do
    git -C "$source_root" apply --reverse --check "$patch"
  done
fi

cmake -S "$source_root" -B "$build_root" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$build_root" --parallel "${PCSS_BUILD_JOBS:-2}"
