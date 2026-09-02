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
   **current `main` SHA** (not the last PyPI release: upstream tags
   releases every month or two while `main` moves daily — 66 commits since
   May 2026, six on 2026-09-01 alone; the `LRUPromptCache` described in
   `docs/MLX_CACHE.md` is `main` code). Add it as `third_party/mlx-lm`,
   record that SHA plus the upstream URL in `third_party/mlx-lm.PIN`, and
   set up `native/mlx-patches/` + a `scripts/mlx-build.sh` that installs the
   fork editable into the venv (`pip install -e third_party/mlx-lm`) and
   verifies checkout == PIN + series (mirror `scripts/build.sh`). Bumping
   the PIN to a newer upstream SHA is allowed any time our commits rebase
   cleanly; record the bump in the run journal.
2. Verify the toolchain on the M4 host: `mlx` at its latest release
   (v0.32.x), `mlx-lm` installed **from the fork, not PyPI** (the PyPI
   0.31.3 there is four months behind `main`), `mlx-community/Qwen3-8B-4bit`
   present (it is), `mlx_lm.server` starts, `cached_tokens` appears in
   responses.
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
keep-or-discard, journaling continuously.

Done so far: lever "admission threshold" (kept, +1.5 pp), lever "remove
eager restore" (kept, ~2 s per restart). Diagnosis from Gate G1-MLX: the
only losing case is a cold turn that recovers a few hundred preamble
tokens, because the disk path has a **fixed cost per entry** (whole-state
safetensors file, full-file sha256, monolithic read) that a small saving
cannot cover; large recoveries win 3-6x. The ladder below is ordered to
attack that fixed cost at its source, cheapest first.

**Ladder (revised 2026-09-02, replaces the original order):**

1. **Cost-model admission instead of a constant.** The server keeps two
   online estimates — prefill tokens/s and disk read MB/s (EWMA over
   observed requests/loads) — and consults disk only when
   `entry_bytes / read_speed < recoverable_tokens / prefill_speed`
   (times a safety factor). Replaces the fixed 1500-token threshold;
   adaptive to host, model, quantization, and disk. Small change.
2. **Warm-start the small stuff only.** On start, eagerly load only
   `system`-segment entries (mlx-lm already tags them; they are tiny),
   never conversation states. Closes the fresh-session-after-restart case
   cheaply; the removed bulk restore was wrong because it loaded 2 GB
   entries, not because eager loading is wrong.
3. **Shrink the fixed cost.** Quantized KV on disk (`to_quantized`,
   4-bit, ~4x less to read — measure output parity separately), mmap-based
   load instead of a full read, chunked checksums or
   write-temp-then-rename instead of a full-file sha256 on every load.
   Expected to make the currently-losing cold turns profitable without
   any threshold.
4. **Block-level content-addressed store** (the structural upgrade; start
   only after 3 shows where the monolithic path tops out). Store KV in
   256-token blocks keyed by the hash of the token prefix (mlx-lm's
   `KVCache.step` is already 256): a shared preamble is stored once for
   all sessions; a growing conversation appends blocks instead of
   rewriting its whole state (kills the write amplification that grew the
   store to 45 GB); partial hits read exactly the matching blocks. This is
   the vLLM / SGLang prefix-cache layout applied to the disk tier. It is
   a rewrite of `DiskPromptStore`, weeks of agent work — justified by
   receipts from 3, not assumed.
5. **Write-behind at idle, not at eviction.** After each completed
   request, when the queue is empty, flush new blocks/entries to disk in
   the background so disk is always a superset of RAM and eviction is a
   free drop. Natural on 4; a stopgap on monolithic files.
6. **Layer-pipelined restore.** The forward pass consumes KV layer by
   layer, so load layer *i+1* while layer *i* computes the uncached
   tokens: hides most of the disk read inside prefill. Requires touching
   the prefill loop; do after 4, where per-layer/per-block layout makes it
   natural.
7. **Tier budgets** (RAM/disk sweep, curve not point) and **multi-project
   stores / recency across restarts** — after the structural levers,
   because their optimum moves with each of them.

Metric discipline for every lever: report cold-turn TTFT bucketed by
recoverable tokens, plus the "elsewhere" ratio; the full-workload
percentage is reported but is not the target — it is dominated by the
261 warm turns the disk tier cannot and should not affect. The product's
real numbers are cold-turn gain and zero warm-turn cost.

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

## Iteration speed rules (binding — a full 3-arm run is ~2.5 h, most of it waste)

The default development loop must take minutes, not tens of minutes.
Multi-hour full-workload runs are for final receipts only, run in the
background while the next item is being worked on. Use this ladder of
loops, cheapest first, and go up a level only when the level below passes:

1. **Unit tests (seconds).** Every mechanism change ships with a test in
   `third_party/mlx-lm/tests/`; synthetic caches, no model.
2. **Tiny-model stress (1-2 min, M4).** Correctness and robustness of the
   disk tier — file races, thread streams, budget bounds, quota eviction,
   restart recovery — do not depend on model quality. Use a small MLX
   model (e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`: ~1 s load,
   instant prefill) with synthetic long prompts, a small RAM budget, many
   restarts, and eviction on every insert. All three Phase M1 scale bugs
   are reachable here in under two minutes. Build this harness once
   (`src/pcss/mlx_stress.py` or similar) and run it before every push.
3. **Cold-turn microbench on a store fixture (2-5 min, real model).**
   Build the disk store for the frozen workload once and keep it as a
   fixture; clone it per iteration (APFS `cp -c` is instant). Add a
   diagnostic-only server endpoint / flag that drops the RAM tier and
   reopens the disk index in-process ("simulated restart") so the 24
   post-restart cold turns cost 24 requests, not 24 x 20 s model reloads.
   Measure TTFT (streaming, `max_tokens=1`), not 48 decode tokens: the
   cache only affects prefill, decode is noise for this question. This is
   the loop for ladder levers and restore-path work. Diagnostic config —
   never gate evidence on its own.
4. **Screen on the real model, real restarts, subset (~10 min).**
   Pre-registered subset (e.g. 4 sessions, 8 restart points), `ram_disk`
   only, against stored reference arms. Levers that fail here are
   discarded without a full run.
5. **Full receipt (hours, background, rare).** Only to confirm a GO or to
   record the cumulative ladder result; never to discover bugs.

- **Do not re-run control arms whose code path did not change.** `no_cache`
  and `ram` are measured once per (workload, PIN, non-disk patch set) and
  reused as the reference for every `ram_disk` iteration; record the
  reference run id in the receipt. Pairing across runs is acceptable for
  iteration (temperature 0, fixed seed, same host); only the final GO for a
  phase is confirmed with a fresh simultaneous 3-arm run.
- **Stress before scale.** Before any multi-hour gate run of a changed
  mechanism, run a ~10-minute stress scenario built to hit the failure
  classes seen so far: a pre-populated large disk store (tens of GB, 10+
  entries) against a deliberately small RAM budget (e.g. 512 MiB), several
  restarts, eviction on every insert. All three Phase M1 scale bugs were
  reachable this way; finding them at turn 616 of 858 cost two hours each.
- **Mini-gate for ladder levers.** Profit-ladder levers are screened on a
  pre-registered subset (e.g. 4 sessions, 8 restart points, ~20 min)
  against the reference arms; only levers that pass the screen get the
  full workload. The subset is fixed once, before screening any lever.
- **Keep restart overhead out of the measured path.** Model reload
  (~15-20 s x 72 restarts) is already outside `wall_s`; do not add restarts
  beyond the pre-registered schedule.
- **Never wait idle on a run.** While a run executes, do the next item that
  does not depend on it: cost decomposition of prior data, the next lever's
  code and tests, doc reconciliation.

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
