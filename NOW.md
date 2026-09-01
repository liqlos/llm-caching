# Current handoff — MLX line

`docs/MLX_CACHE.md` is the sole current-state authority for behavior,
receipts, and recorded verdicts of the MLX line. `NOW.md` is the sole
authority for the plan, gates, and execution rules. If they disagree on
gate status, the plan here wins and the doc gets reconciled.

## Strategic decision (2026-09-01)

The product is a multi-level (RAM + persistent SSD) prompt/KV cache for the
owner's own Apple Silicon machines, built on **MLX / mlx-lm**, not on
llama.cpp. Reasons: the owner runs Macs; MLX is the native, usually faster
engine there; mlx-lm already ships the RAM tier (`LRUPromptCache`) and the
serialization primitives, so the missing piece — a persistent disk tier — is
a Python increment on top of existing infrastructure rather than a C++
subsystem; and mlx-lm's contribution bar is realistic for incremental PRs.

The llama.cpp line is **frozen**, not deleted: its fork, patches, receipts,
and `docs/SSD_L2_CACHE.md` stay as evidence and as a possible upstream
trigger-fix contribution. Do not continue it unless the owner reopens it.

## Reading order and what does NOT bind

- Read `NOW.md`, then `docs/MLX_CACHE.md`. Nothing else is required.
- `docs/SSD_L2_CACHE.md` is the frozen llama.cpp-line authority; read it
  for transferable findings (listed in `docs/MLX_CACHE.md`), not for
  instructions.
- `PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `native/README.md`, and everything
  under `qualification/` are the historical PCSS / sequence-datastore line.
  Their kill gates, thresholds, proxy-capture and owner-holdout rules bind
  ONLY that line and must not be imported.
- Upstream `AGENTS.md`/`CONTRIBUTING.md` files inside any `third_party/`
  fork describe rules for upstream PRs; private forks are exempt. Commits to
  a fork submodule inside this repository are ordinary in-scope work and
  need no per-action approval: commit in the submodule, push its branch to
  the fork, regenerate the mirrored patch series, commit the gitlink here.
- This repository has no separate "commit policy": in-scope commits and
  pushes that advance the plan are allowed without asking.

## Goal invariant (does not change)

Deliver honest, receipt-backed evidence of whether a persistent SSD tier
under mlx-lm's RAM prompt cache produces real value on the owner's
representative traffic — and, if it does, maximize that value through
receipt-gated iteration. The evidence bar never drops: raw receipts,
explicit claim boundaries, paired controls where a cost is claimed, no
synthetic-to-representative generalization, no post-hoc workload selection.
A clean, receipt-backed negative result satisfies the goal; an unsupported
positive claim does not.

## What carries over from the llama.cpp line (reuse, do not rebuild)

- **Owner-trace corpus pipeline**: the v2/v2b builders and the frozen
  workload definitions under
  `artifacts/ssd-redesign/20260901T163602Z-phase1b-trigger-patch/`
  (`corpus-v2b/`, `PHASE0_WORKLOAD_DEFINITION_V2B.md`, `phase1b2_run.py`,
  `phase1b2_analyze.py`). Requests are already chat-shaped (system preamble
  + growing transcript); they map onto mlx-lm's `/v1/chat/completions`.
- **Paired benchmark method**: baseline vs treatment arms, same request
  bodies, per-turn wall-clock, bootstrap intervals, receipts with model /
  engine / host hashes. Only the HTTP adapter changes: the reused-prefix
  metric is `usage.prompt_tokens_details.cached_tokens` instead of
  `cache_n`.
- **Findings** listed in `docs/MLX_CACHE.md` (trigger coverage, write
  amplification, restore-cost scaling, quota headroom, continuous metrics).
- **Owner-trace data rules** (below) and the **execution host rule**
  (below) — unchanged.

## Repository layout for the MLX line

- `third_party/mlx-lm` — submodule tracking the owner's private fork
  `liqlos/mlx-lm` (`main` = upstream release tag + our commits). Created in
  Phase M0. Same fork workflow as `third_party/llama.cpp`; mirrored patch
  series under `native/mlx-patches/`.
- `src/pcss/` — the laboratory (Python). New MLX-specific harness code goes
  here next to the existing modules; do not fork the lab.
- `artifacts/mlx/<run-id>/` — run roots for the MLX line.
- `third_party/llama.cpp`, `native/patches/`, `docs/SSD_L2_CACHE.md` —
  frozen llama.cpp line.

## Ordered plan (gated)

### Phase M0 — Stand up the MLX lab (no product code yet)

1. Create the private fork `liqlos/mlx-lm` from `ml-explore/mlx-lm` at the
   latest release tag (`v0.31.3` as of 2026-04-22; check for newer), add it
   as `third_party/mlx-lm`, pin the tag in `third_party/mlx-lm.PIN`, and
   set up `native/mlx-patches/` + a `scripts/mlx-build.sh` that installs the
   fork editable into the venv and verifies checkout == PIN + series
   (mirror `scripts/build.sh`).
2. Verify the toolchain on the M4 host: `mlx`, `mlx-lm` versions,
   `mlx-community/Qwen3-8B-4bit` present (it is), `mlx_lm.server` starts,
   `cached_tokens` appears in responses.
3. Write the mlx-lm HTTP adapter for the lab: OpenAI chat endpoint, capture
   `cached_tokens`, `prompt_tokens`, wall-clock, generated token IDs.
4. **Baseline receipt (upstream as-is, RAM tier only)** on corpus v2b
   extended to all 18 `oculusprojects` sessions, pre-registered as
   `PHASE0_WORKLOAD_DEFINITION_MLX_V1.md` before running, on the M4 host:
   arms = cache disabled (`--prompt-cache-size 0` or equivalent) vs upstream
   RAM cache with a realistic `--prompt-cache-bytes` (start at 4 GiB on the
   24 GiB host with the 8B 4-bit model). Record per-turn paired ratios
   bucketed by `cached_tokens`, within-process hit profile, post-restart
   first-turn `cached_tokens` (expected 0 — that is the gap), RAM usage.
   This is the reference every later arm is compared against.

Gate M0: adapter works, receipts validate, baseline recorded in
`docs/MLX_CACHE.md`. No product decision here.

### Phase M1 — Persistent disk tier MVP

Implement in the fork, as small increments that each stand alone (they are
the future upstream PR series):

1. **Flush on shutdown / load on start**: serialize every RAM entry via
   `save_prompt_cache` into a store directory with an index
   (token sequence → file, model key, mlx / mlx-lm version, system
   fingerprint, checksum); rebuild the RAM cache from the index on start.
   Compatibility key derived from those fields, fail-closed on mismatch.
2. **Demote on eviction**: when `LRUPromptCache` evicts under its
   count/bytes limits, write the entry to the disk tier asynchronously
   instead of discarding it. Disk tier has its own byte quota with LRU and
   headroom (never measure at the ceiling).
3. **Disk lookup on RAM miss**: extend `fetch_nearest_cache` so that a RAM
   miss (or a shorter RAM prefix than the disk has) consults the disk
   index with the same exact / longer-trim / shorter semantics, loads via
   `load_prompt_cache`, and promotes the entry back to RAM.
4. CLI: `--prompt-cache-dir`, `--prompt-cache-disk-bytes`; disabled by
   default; all three semantics logged.

Gate G1-MLX (pre-register metrics before the run, on the M4 host, same
frozen workload as M0): arms = cache disabled / RAM-only / RAM+disk, long-
lived server with real restarts between sessions. GO if the RAM+disk arm
beats RAM-only on post-restart cold turns and is not net slower elsewhere,
on continuous per-turn paired ratios. If disk adds nothing over RAM-only
with a working write path and adequate restart boundaries: earned NO_GO;
record it, keep the feature disabled by default, and stop the line.

### Phase M2 — Profit ladder (only after GO at G1-MLX)

One lever at a time: hypothesis → paired experiment → receipt →
keep-or-discard, journaling continuously. Suggested order:

1. **Quantized disk entries**: store demoted caches as `QuantizedKVCache`
   (4-bit) — ~4x less disk and read; measure restore cost and output
   parity separately.
2. **Segment-level entries**: persist `system` and `user` segment caches
   (mlx-lm already distinguishes them) so a shared preamble is one small
   hot entry rather than part of every conversation's tail.
3. **Prefetch**: start loading the disk candidate while the request is
   still being tokenized / queued; overlap disk read with prefill of the
   remainder.
4. **Tier budgets**: RAM/disk size sweep — hit rate and latency vs budget
   curve, not a single point.
5. **Admission / recency policy** across restarts; multi-project stores.
6. **Restore path**: lazy per-layer load, mmap-friendly layout, only if
   the cost decomposition says restore dominates.

Stop condition (loop-until-dry): the ladder ends when two consecutive
levers each fail to produce a receipt-backed improvement, or when a lever's
cost provably exceeds its benefit. Record the cumulative paired result in
`docs/MLX_CACHE.md`.

### Phase M3 — Upstream (owner-driven, after M1/M2 receipts)

Open an issue on `ml-explore/mlx-lm` describing the persistence gap with
receipts; then the M1 increments as separate PRs in order. The owner writes
the issue and PR text and must be able to defend every line; agents prepare
material only.

## Owner-trace corpus (authorized data source)

The owner has explicitly authorized using their real local agent-session
histories as workload data: Claude Code session transcripts (under
`~/.claude/projects/`, JSONL) and Codex session logs (under `~/.codex/`).

- **Privacy boundary:** this data never leaves the owner's own machines.
  Never upload it, publish it, or embed excerpts in documents — receipts
  reference corpora by manifest hash, not by content.
- **Pre-registration, not post-hoc:** a trace-derived workload is a
  versioned, frozen definition with corpus manifest and selection criteria
  written BEFORE any measured run. Selecting or trimming traces after
  seeing results is prohibited.
- Heterogeneous traffic is expected and is the point.

## Execution host rule (binding)

All live server runs, benchmarks, and experiments run on the dedicated
Apple M4 Pro host (24 GiB), `ai-server@mac2-llm.local`. The local M1 Pro is
NOT a measurement host — memory-constrained and shared with unrelated
development — use it only for code edits, unit tests, lint, and data
processing. A measurement on any other hardware is invalid as gate
evidence. Every receipt records the host, `mlx` and `mlx-lm` versions, and
the server's `system_fingerprint`.

## Anti-stall rules (binding on the executing agent)

- **Two-strike loop-breaker.** The same experiment design may fail or
  stall twice; a third run needs a written new hypothesis. A code change to
  the mechanism under test, or a tier-configuration change that alters
  which mechanism is measured, is a new hypothesis; a bare rerun or a
  different sample of the same corpus is not.
- **Measure the product, not a proxy.** Configs chosen to "guarantee the
  tier is exercised" are diagnostics, never gate evidence. Gate runs use a
  configuration a real deployment would use on the recorded host.
- **Approach freedom within the evidence bar.** Substitute designs,
  harnesses, or measurement approaches freely with a recorded rationale;
  the plan binds the questions and the gate order, not the mechanism of
  answering.
- **Negative results are progress.** A receipt-backed NO_GO or blocker is a
  completed deliverable; a confirmed positive is not a finish line while
  the ladder has untested levers.
- **Timebox before pivot.** A planned item that consumes a full session
  without a receipt or a concrete blocker gets journaled honestly and
  replaced by the cheapest design that still answers the same question.
- **No silent scope changes.** Skipping a gate, entering M2 without GO, or
  redefining the workload to manufacture a positive is prohibited.
- **Continuous journaling.** Append progress to the run journal as you go
  (per item: hypothesis, receipt path, verdict) so any interruption leaves a
  resumable state.

## Explicitly parked (do not start)

- Rewriting the store or server in another language (e.g. Rust): parked
  until the profit ladder shows where the time goes.
- The sequence datastore (cross-prompt, non-prefix reuse): remains
  `NO_GO_CURRENT_REDESIGN`. Reopening requires a written new hypothesis
  addressing serial-canonical K>2 correctness AND the storage-projection
  failure.
- The llama.cpp line: frozen (see Strategic decision).
