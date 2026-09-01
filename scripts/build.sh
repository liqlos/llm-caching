#!/usr/bin/env bash
set -euo pipefail

# Build the pinned llama.cpp submodule.  Never substitute a Homebrew install:
# the submodule SHA is the product's native ABI and server source.
root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="$root/third_party/llama.cpp"
build_root="${PCSS_BUILD_DIR:-$root/build/llama.cpp}"
patch_root="$root/native/patches"
pin_revision="$(sed -n 's/^revision=//p' "$root/third_party/llama.cpp.PIN")"

test -f "$source_root/CMakeLists.txt" || {
  echo "missing pinned llama.cpp submodule; run: git submodule update --init --recursive" >&2
  exit 2
}

# The source tree must be exactly PIN + the maintained patch series, whether
# that state comes from applying the series onto a clean PIN checkout or from
# the local fork history that already contains it.  Anything else is refused
# rather than silently mixed into a build.
expected_tree() {
  local wt
  wt="$(mktemp -d)"
  git -C "$source_root" worktree add -q --detach "$wt" "$pin_revision"
  for patch in "$patch_root"/*.patch; do
    git -C "$wt" apply --index "$patch"
  done
  git -C "$wt" write-tree
  git -C "$source_root" worktree remove --force "$wt"
}

actual_tree() {
  local idx
  idx="$(mktemp)"
  rm -f "$idx"
  GIT_INDEX_FILE="$idx" git -C "$source_root" read-tree HEAD
  GIT_INDEX_FILE="$idx" git -C "$source_root" add -A
  GIT_INDEX_FILE="$idx" git -C "$source_root" write-tree
  rm -f "$idx"
}

if [ "$(git -C "$source_root" rev-parse HEAD)" = "$pin_revision" ] \
  && git -C "$source_root" diff --no-ext-diff --quiet \
  && [ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]; then
  for patch in "$patch_root"/*.patch; do
    git -C "$source_root" apply "$patch"
  done
else
  expected="$(expected_tree)"
  actual="$(actual_tree)"
  if [ "$expected" != "$actual" ]; then
    echo "third_party/llama.cpp is not PIN $pin_revision plus native/patches (tree $actual, expected $expected)" >&2
    exit 2
  fi
  echo "patch series already present in $(git -C "$source_root" rev-parse --short HEAD)"
fi

cmake -S "$source_root" -B "$build_root" \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$build_root" --parallel "${PCSS_BUILD_JOBS:-2}"
