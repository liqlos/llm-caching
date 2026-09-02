# MLX persistent prompt/KV cache — current state

This is the sole current-state authority for the MLX line: what exists
upstream, what this project has built on top, what has been measured, and
the claim boundary. The plan and execution rules live in `NOW.md`.

## Upstream baseline (`ml-explore/mlx-lm` `main`, read on 2026-09-01)

Upstream is active: commits land daily on `main` (six on 2026-09-01), but
PyPI releases are cut only every month or two (last: v0.31.3, 2026-04-22).
The description below is of `main`; the PyPI package can lag it by months,
which is why the project pins a `main` SHA and installs from the fork.

`mlx_lm.server` already has an in-memory RAM prompt-cache tier:

- `LRUPromptCache` (`mlx_lm/models/cache.py`): a `PromptTrie` keyed by
  `(model, token sequence)`; `fetch_nearest_cache` returns an exact entry, a
  longer entry trimmed down to the common prefix (`trim_prompt_cache`), or the
  longest shorter prefix; `insert_cache` stores an entry, drops now-redundant
  prefixes of trimmable caches, and evicts by per-type LRU
  (`assistant` < `user` < `system` priority) under `--prompt-cache-size`
  (count, default 10) and `--prompt-cache-bytes`.
- The server inserts at the end of the `system` and `user` segments and at
  the end of every generation (`assistant`). There is no trigger problem:
  every completed request lands in the cache by design.
- Responses expose the reused prefix length as
  `usage.prompt_tokens_details.cached_tokens` (OpenAI-compatible). This is
  the measurement hook, equivalent to llama-server's `cache_n`.
- `save_prompt_cache` / `load_prompt_cache` serialize a cache to
  `.safetensors` with per-layer metadata; `KVCache.to_quantized` produces a
  4/8-bit `QuantizedKVCache`. Both are library functions; the server does not
  use them.
- **There is no disk tier and no persistence across restart.** Eviction
  discards the entry; a restart starts empty.
- **`--prompt-cache-bytes` was dead code on `main` at the pinned SHA**
  (`e93eaf7f491a`): `run()` constructed `LRUPromptCache` with only the
  count-based `max_size` (`--prompt-cache-size`, default 10), never wiring
  `max_bytes` through. The byte quota was silently unenforced — on the M4
  host, 10 sequences reached 12.7 GB with `--prompt-cache-bytes 4294967296`
  set, more than 3x over budget, and the server process was later found
  dead mid-run with no traceback (consistent with, though not confirmed by
  a jetsam log line as, an OOM kill). Fixed in the fork:
  `native/mlx-patches/0001-fix-wire-prompt-cache-bytes-into-LRUPromptCache-s-ma.patch`,
  PIN revision unchanged. All measurements below are post-fix.

## What this project adds

Nothing yet beyond the `--prompt-cache-bytes` fix above (Phase M0, Gate M0
closed — see receipt below). Phase M1 (persistent disk tier) has not
started.

## Gate M0 — closed, baseline recorded (2026-09-02)

**Setup:** `mlx_lm.server` (fork `liqlos/mlx-lm`, commit `5927f81`, PIN
`e93eaf7f491a` + 1 patch), `mlx` 0.32.2, `mlx-lm` 0.32.0, model
`mlx-community/Qwen3-8B-4bit`, M4 Pro host (`ai-server@mac2-llm.local`).
Workload: `PHASE0_WORKLOAD_DEFINITION_MLX_V1.md`, corpus v1 — all 15
`oculusprojects` sessions meeting `MIN_USER_TURNS=5` (of 18 quiet
candidates; owner-trace data, never leaves the owner's machines), 286
turns per arm. Two arms, each a single continuous server process (no
restart — restart is Phase M1's question): cache disabled
(`--prompt-cache-size 0`) vs upstream RAM cache with `--prompt-cache-bytes`
4 GiB. Receipt: `artifacts/mlx/20260901T193508Z-phase-m0-baseline/result_m0_baseline.json`.

**Results:**

- Within-process cache hit rate (RAM arm): 99.65% (285/286 turns) — every
  turn but the very first cold one in the whole run hit the cache, as
  expected for exact chronological prefix replay.
- Paired per-turn wall-clock, no-cache over RAM (n=286): mean ratio
  **6.90x**, median **5.61x**, bootstrap 95% CI [6.36, 7.44] — i.e. the
  existing RAM cache alone already makes cached turns roughly 5-7x faster
  than recomputing from scratch on this workload.
- Ratio scales with reused-prefix size, as expected: 1.4x at 1-499 cached
  tokens, 2.3x at 500-1999, 6.2x at 2000-7999, 11.3x at 8000+ tokens
  (mean-of-bucket; see receipt for medians and n per bucket).
- Peak RSS: 4655 MB (no-cache) vs 4725 MB (RAM, post-fix) — both sane on
  the 24 GiB host; confirms the `--prompt-cache-bytes` fix holds the RAM
  arm inside its 4 GiB budget rather than the pre-fix 12.7 GB+ runaway.
- Post-restart first-turn `cached_tokens`: not applicable here by design
  (both arms are continuous processes) — this is exactly the gap Phase M1
  / Gate G1-MLX measures.

**Verdict:** Gate M0 passed — adapter validated end-to-end, receipt
recorded, no product decision made. The bar Phase M1's disk tier must
clear at Gate G1-MLX is this RAM-only baseline, not the no-cache arm: the
existing RAM cache already captures most of the same-session win, so the
disk tier's only unclaimed territory is the post-restart cold-start gap
this baseline deliberately did not measure.

## Claim boundary

Every claim above is bound to: M4 host `ai-server@mac2-llm.local`, `mlx`
0.32.2, `mlx-lm` 0.32.0 (fork commit `5927f81`), model
`mlx-community/Qwen3-8B-4bit`, corpus v1 manifest
(`artifacts/mlx/20260901T193508Z-phase-m0-baseline/corpus-v1/corpus_v1_manifest.json`),
receipt `result_m0_baseline.json` in the same run root. No claim beyond
Gate M0 exists yet.

## Relationship to the llama.cpp line

`docs/SSD_L2_CACHE.md` is the frozen authority for the llama.cpp line. Its
findings that transfer to MLX regardless of engine:

- save triggers must cover single-conversation traffic (upstream mlx-lm
  already does this by inserting on every completion);
- writing a growing conversation's full state on every turn is quadratic
  write amplification — demote to disk on eviction and on shutdown, not per
  completion;
- restore cost scales with reused-prefix size, so the disk tier must be
  measured with a realistic RAM tier in front of it, never as SSD-only;
- a store running at its quota ceiling puts compaction inside the measured
  window; size quotas with headroom;
- binary "hit / no hit" on a handful of restart boundaries is noise; measure
  continuous per-turn paired ratios bucketed by cached-token count;
- on hit turns with reused prefix under ~4,000 tokens the disk tier was
  1.05x-1.3x faster on real owner traffic even under a degenerate config —
  the benefit is real where the prefix is moderate; the open question is
  the large-prefix regime.
