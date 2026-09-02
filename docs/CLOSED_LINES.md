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

## Persistent cross-session datastore speculation on MLX — CLOSED 2026-09-03

Line D, reopened from the PCSS verdict above under the hypothesis that a
new implementation (MLX, exact-by-construction verification, an in-RAM
suffix/n-gram index rather than KV state) would fix both of PCSS's failure
causes. It did — the datastore never diverged from what the underlying
speculative engine already does — but the thesis still failed on its own
central measurement.

**Gate D0 (throughput): NO_GO.** The datastore's margin over the
per-request lookup mlx-dspark already ships (a persistent global index vs.
drafting from the current session's own history) measured **1.014**
against a pre-registered 1.20 threshold, on 319 real `Qwen3-8B-8bit`
greedy turns replayed against a 1.46M-token index built from the owner's
own coding-agent transcripts (6 projects, 44 sessions). Robust to every
variant tried: 1.014-1.071 across three verify-cost curves; flat under a
6x larger index (0.007 drift); no profitable project subset (one project
was below parity); unmoved by fixing several real bugs in the draft
algorithm (a matcher that stayed silent half the time, a tie-break that
favored recency over match quality) that each mattered more than the
cross-session margin itself. Two independent implementations (an n-gram
hash and a suffix array, built without sharing code or results) converged
on the same ratio within 5%. Root cause: the owner's sessions are long
(median 76,694 tokens) and a per-request drafter self-saturates on the
current session's own history long before a persistent store can add
much — of 43,659 accepted draft tokens in the suffix-array replay, other
sessions and projects contributed 4.2%. Receipt:
`artifacts/mlx/20260902T152845Z-phase-d0-marathon/d0b/GATE_D0_RECEIPT.md`
(local-only; referenced here by path, not content).

**Exactness: PASS**, once measured against the guarantee the underlying
engine (mlx-dspark) actually offers — greedy up to floating-point ties —
rather than the stronger "bit-identical" this project's own docs
mistakenly claimed until this closure. No divergence between the
datastore-on and datastore-off arms ever exceeded ~2 bf16 ulps, across 24
real `Qwen3-8B-8bit` turns in two kernel modes.

**One real, general lever surfaced and was declined.** Verify cost at long
context runs ~34% above a bandwidth-bound floor at 16k tokens (each extra
verify row re-reads nearly the full KV cache) — a property of Apple
Silicon's attention kernels, not of the datastore, applying to any
drafter and any model on the host. The owner decided not to expand this
project's scope to pursue it; it may be picked up by a future,
separately-scoped project.

**Why closed:** the central technical thesis — that persistence across
sessions and projects beats per-request lookup — is false on the owner's
actual workload (long agentic coding sessions), independent of engine
correctness or algorithm quality. `GlobalDatastore` stays in the fork
(`liqlos/mlx-dspark`) as a working, correctness-checked, opt-in feature;
no further engineering or measurement is planned against it here.

## Historical PCSS documents

`PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, and `qualification/` describe the 2026-08
PCSS line on llama.cpp. Their kill gates, thresholds, capture protocols and
holdout rules bind only that line. The Python lab modules named `llamacpp_*`,
`exact_prompt_*`, `live_server_*`, `verifier*`, `provider*` under `src/pcss/`
belong to it and are kept as prior art only.
