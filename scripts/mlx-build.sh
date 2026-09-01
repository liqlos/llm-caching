#!/usr/bin/env bash
set -euo pipefail

# Install the pinned mlx-lm submodule editable into the venv. Never
# substitute a PyPI install: the submodule SHA plus native/mlx-patches is
# the product's mlx-lm source (see third_party/mlx-lm.PIN and NOW.md).
root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="$root/third_party/mlx-lm"
patch_root="$root/native/mlx-patches"
pin_revision="$(sed -n 's/^revision=//p' "$root/third_party/mlx-lm.PIN")"
venv_root="${PCSS_MLX_VENV:-$root/.venv-mlx}"

test -f "$source_root/pyproject.toml" || {
  echo "missing pinned mlx-lm submodule; run: git submodule update --init --recursive" >&2
  exit 2
}

# The source tree must be exactly PIN + the maintained patch series, whether
# that state comes from applying the series onto a clean PIN checkout or from
# the local fork history that already contains it. Anything else is refused
# rather than silently mixed into a build.
expected_tree() {
  local wt
  wt="$(mktemp -d)"
  git -C "$source_root" worktree add -q --detach "$wt" "$pin_revision"
  for patch in "$patch_root"/*.patch; do
    [ -e "$patch" ] || continue
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
    [ -e "$patch" ] || continue
    git -C "$source_root" apply "$patch"
  done
else
  expected="$(expected_tree)"
  actual="$(actual_tree)"
  if [ "$expected" != "$actual" ]; then
    echo "third_party/mlx-lm is not PIN $pin_revision plus native/mlx-patches (tree $actual, expected $expected)" >&2
    exit 2
  fi
  echo "patch series already present in $(git -C "$source_root" rev-parse --short HEAD)"
fi

test -d "$venv_root" || python3 -m venv "$venv_root"
# shellcheck disable=SC1091
source "$venv_root/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$source_root"

installed_root="$(python -c 'import mlx_lm, pathlib; print(pathlib.Path(mlx_lm.__file__).resolve().parent.parent)')"
if [ "$installed_root" != "$source_root" ]; then
  echo "mlx_lm resolved to $installed_root, expected the editable install at $source_root (check for a stray PyPI install shadowing it)" >&2
  exit 2
fi

echo "mlx-lm editable-installed from $source_root @ $(git -C "$source_root" rev-parse --short HEAD) into $venv_root"
