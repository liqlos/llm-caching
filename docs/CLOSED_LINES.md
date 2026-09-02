# Closed lines

This file is the only record in the working tree of what this project tried
and closed. Code, patches, and receipts live in git history (commit ranges
below) and in the untracked `artifacts/` directories on the owner's
machines. None of this is current instruction.

## Prefix / prefill KV caching — OUT OF SCOPE (closed 2026-09-02)

Two lines built a persistent RAM+SSD prompt/KV prefix cache and measured it
on the owner's real Claude Code traffic:

- **llama.cpp SSD L2** (2026-08-31 → 2026-09-01, commits `8498216`..`3f253e2`):
  native C++ SSD tier in `llama-server`, caller-attested compatibility key.
  Synthetic M4 receipts: 4.99x exact reuse, 1.58x partial. On owner traces
  the write path never fired for single-slot traffic (trigger gap, patched),
  then a forced single-tier config measured a provisional NO_GO. Frozen
  before the planned multi-level run.
- **mlx-lm disk tier** (2026-09-01 → 2026-09-02, commits `59edbae`..`84b82b7`):
  Python disk tier under mlx-lm's `LRUPromptCache` (flush/load,
  demote-on-eviction, disk lookup on miss, cost-model admission, 4-bit
  quantized entries, write-behind). Gate GO on owner traces: post-restart
  cold turns 1.1x (small prefix) to 8.9x (8k+ tokens), warm turns at parity,
  13-20% less total wall-clock than RAM-only at 1 restart per 12 turns.
  Also found and fixed a real upstream bug: `--prompt-cache-bytes` was
  unwired on mlx-lm `main` (OOM). That fix remains a valid upstream PR.

**Why closed:** [oMLX](https://github.com/jundot/omlx) (Apache-2.0, 21k
stars, daily commits) already ships a block-based paged KV cache with prefix
sharing and copy-on-write, a hot RAM tier and a cold SSD tier that survives
restarts, TurboQuant KV compression, continuous batching, and an
Anthropic-compatible API — built explicitly for Claude Code-style local
coding. Everything both lines built is a subset of it. **This project does
not build prefix caching. When prefix caching is needed, run oMLX.**

Transferable findings (engine-independent, keep in mind for any tier that
loads state from disk): a per-entry fixed load cost makes small recoveries
lose; decide by a cost model, not a constant; write full state per turn is
quadratic amplification — write on eviction or at idle; never measure at
the store's quota ceiling; report per-turn paired ratios bucketed by
recovered size, not a single aggregate.

The private forks `liqlos/llama.cpp` and `liqlos/mlx-lm` still hold the
code as backups. They are not referenced by this repository.

## Sequence datastore on llama.cpp (PCSS) — NO_GO for that design (2026-08-31)

Retrieval-based speculative decoding: a datastore of previously seen token
sequences proposes drafts, the model verifies them in one pass. Implemented
as a custom C++ bridge to llama.cpp (`native/pcss_llama_bridge.cpp`,
removed 2026-09-02; see commit `027067b` and earlier). Verdict
`NO_GO_CURRENT_REDESIGN` in
`artifacts/ssd-redesign/20260831T130127Z-84982163c750/REVIEW.md`:

- production K8/K16 verification was not serial-canonical (output
  mismatches) — a defect of that bridge, not of speculative verification,
  which is exact by construction;
- K4 exact-reuse was 8% slower than baseline in that integration;
- no datastore candidate fit the 256 MiB storage projection on the
  calibration traces;
- naturalistic workload showed zero persistent hits with that design.

The *idea* is not closed. It is reopened on MLX under a new hypothesis in
`NOW.md`, because both failure causes were specific to that implementation:
mlx-lm's speculative path verifies drafts natively and exactly, and a
suffix-tree datastore over text tokens costs ~10 bytes/token, not KV state.

## Historical PCSS documents

`PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, and `qualification/` describe the 2026-08
PCSS line on llama.cpp. Their kill gates, thresholds, capture protocols and
holdout rules bind only that line. The Python lab modules named `llamacpp_*`,
`exact_prompt_*`, `live_server_*`, `verifier*`, `provider*` under `src/pcss/`
belong to it and are kept as prior art only.
