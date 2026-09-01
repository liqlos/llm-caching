# Native RAM + SSD L2 prompt/KV cache

This repository carries an active native llama-server prompt/KV cache with a
RAM tier and an optional persistent SSD L2 tier. SSD L2 is disabled by default.
It is enabled only when a nonzero capacity, an absolute store path, and a
caller-attested compatibility key are supplied together.

The sole current-state authority for behavior, receipts, qualification claims,
and limitations is [docs/SSD_L2_CACHE.md](docs/SSD_L2_CACHE.md). The plan,
gates, and execution rules are in [NOW.md](NOW.md).

## Layout

- `third_party/llama.cpp` — the product. A submodule tracking the private
  fork `liqlos/llama.cpp` (`main`), which is upstream llama.cpp at the pinned
  revision plus the RAM+SSD prompt-cache commits. The patched `llama-server`
  is what gets measured and shipped.
- `native/patches/` — the same commits as a patch series, so the build is
  reproducible from the bare upstream pin (`scripts/build.sh` verifies this).
- `src/pcss/`, `tests/`, `scripts/`, `qualification/`, `artifacts/` — the
  laboratory around the product: benchmark harnesses, owner-trace corpora,
  receipts, and historical PCSS evidence. No product code lives here.

## Runtime controls

```sh
llama-server \
  --cache-ram <MiB> \
  --cache-ssd <MiB> \
  --cache-ssd-path /absolute/cache/root \
  --cache-ssd-key sha256:<64-lowercase-hex> \
  ...
```

`--cache-ssd 0` is the default. A nonzero SSD capacity requires both the path
and key. The implementation validates key syntax but does not derive the
compatibility identity independently.

## Build and checks

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
bash scripts/check.sh
bash scripts/native-check.sh
```

For a local pinned llama-server build, set `PCSS_MODEL` to an existing GGUF and
run `bash scripts/run_server.sh`. The scripts do not download a model or fall
back to a system llama.cpp installation.

## Historical records

- [PCSS_PRODUCT.md](PCSS_PRODUCT.md): historical PCSS product status.
- [M4_RUNBOOK.md](M4_RUNBOOK.md): historical PCSS M4 frontier snapshot.
- [native/README.md](native/README.md): historical native PCSS integration.
- [Sequence-datastore review](artifacts/ssd-redesign/20260831T130127Z-84982163c750/REVIEW.md): run-specific final decision.
- [Continuous journal](artifacts/ssd-redesign/20260831T130127Z-84982163c750/continuous-journal/JOURNAL.md): append-only experimental history.

