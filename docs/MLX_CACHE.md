# MLX persistent prompt/KV cache — current state

This is the sole current-state authority for the MLX line: what exists
upstream, what this project has built on top, what has been measured, and
the claim boundary. The plan and execution rules live in `NOW.md`.

## Upstream baseline (mlx-lm v0.31.3, 2026-04-22; verified against `main` on 2026-09-01)

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

## What this project adds

Nothing measured yet. The MLX line starts at `NOW.md` Phase M0.

## Claim boundary

No MLX claims exist. Every future claim is bound to the recorded M4 host,
model, mlx / mlx-lm versions, system fingerprint, workload manifest, and
receipt hashes, exactly as the llama.cpp line was.

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
