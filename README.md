# Persistent prompt/KV cache for local LLM serving on Apple Silicon

This repository builds and measures a multi-level (RAM + persistent SSD)
prompt/KV cache for the owner's own Macs. The current line is built on
**MLX / mlx-lm**, whose server already ships an in-memory prompt-cache tier;
this project adds the persistent disk tier under it and measures it on the
owner's real agent-session traffic.

- Plan, gates, and execution rules: [NOW.md](NOW.md)
- Current MLX-line state and claim boundary: [docs/MLX_CACHE.md](docs/MLX_CACHE.md)
- Frozen llama.cpp line (receipts and transferable findings):
  [docs/SSD_L2_CACHE.md](docs/SSD_L2_CACHE.md)

## Layout

- `third_party/mlx-lm` — the product (created in Phase M0): a submodule
  tracking the private fork `liqlos/mlx-lm`, upstream at the pinned release
  plus the persistent-cache commits. `native/mlx-patches/` mirrors those
  commits as a patch series.
- `third_party/llama.cpp`, `native/patches/` — the frozen llama.cpp line:
  private fork `liqlos/llama.cpp` with the RAM+SSD cache commits and the
  mirrored series (`scripts/build.sh` verifies checkout == pin + series).
- `src/pcss/`, `tests/`, `scripts/`, `qualification/`, `artifacts/` — the
  laboratory: benchmark harnesses, owner-trace corpora, receipts, and
  historical PCSS evidence. No product code lives here.

## Build and checks (laboratory)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
bash scripts/check.sh
```

Live runs and benchmarks happen only on the dedicated M4 host (see the
execution host rule in `NOW.md`).

## Historical records

- [PCSS_PRODUCT.md](PCSS_PRODUCT.md): historical PCSS product status.
- [M4_RUNBOOK.md](M4_RUNBOOK.md): historical PCSS M4 frontier snapshot.
- [native/README.md](native/README.md): historical native PCSS integration.
- [Sequence-datastore review](artifacts/ssd-redesign/20260831T130127Z-84982163c750/REVIEW.md): run-specific final decision.
- [Continuous journal](artifacts/ssd-redesign/20260831T130127Z-84982163c750/continuous-journal/JOURNAL.md): append-only experimental history.
