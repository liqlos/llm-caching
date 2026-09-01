# SSD L2 current handoff

`docs/SSD_L2_CACHE.md` is the sole current-state authority for behavior,
receipts, and recorded verdicts; `NOW.md` is the sole authority for the
plan, gates, and execution rules. If the two disagree on gate status, the
plan here wins and the doc gets reconciled.

## Reading order and what does NOT bind

- Read `NOW.md`, then `docs/SSD_L2_CACHE.md`. Nothing else is required.
- `PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `native/README.md`, and everything
  under `qualification/` are the historical PCSS / sequence-datastore line.
  Their kill gates, acceptance thresholds (e.g. `LCB95 >= 1.05`), proxy
  capture and owner-holdout rules, and "do not retry" clauses bind ONLY that
  historical line. They are not evidence standards for SSD L2 work and must
  not be imported into it.
- `third_party/llama.cpp` is a private pinned fork. Its `AGENTS.md` /
  `CLAUDE.md` describe rules for contributing to upstream llama.cpp; by its
  own text private forks are exempt. Commits to the submodule inside this
  repository are ordinary in-scope work under this repository's rules and
  need no per-action approval. The upstream rules apply only when an actual
  upstream PR is being prepared, and that step is always owner-driven.
- This repository has no separate "commit policy" beyond the owner's
  standing instructions: in-scope commits that advance the plan are allowed.
- **Native source reproducibility.** The submodule commits on top of the
  upstream PIN (`third_party/llama.cpp.PIN`) exist only in the local
  checkouts (M1 and M4); `.gitmodules` points at upstream, so a fresh clone
  cannot fetch them. The reproducible route is the PIN revision plus
  `native/patches/*.patch`, which `scripts/build.sh` applies (and skips when
  the checkout already contains them). Whenever the submodule gains a
  commit, regenerate the series with
  `git -C third_party/llama.cpp format-patch <PIN>..HEAD` and verify it
  applies cleanly on the PIN before committing.

## Gate status: `G1b_NO_GO_UNDER_FORCED_DISK_CONFIG` → Phase 1c required

Phase 1b (`artifacts/ssd-redesign/20260901T163602Z-phase1b-trigger-patch/RUN_SUMMARY.md`)
delivered three things that stand: the single-slot save trigger patch
(uncommitted in `third_party/llama.cpp`, awaiting owner approval), a working
owner-trace corpus pipeline (v2 cross-project, v2b single-project), and the
**first real-traffic profit ever measured in this project**: on hit turns
with reused prefix under ~4,000 tokens the SSD arm was 1.05x–1.3x faster
than baseline in both corpora independently.

Its `NO_GO_SSD_L2_PRODUCT` verdict is NOT accepted as final, for a reason
the run itself flagged: it was executed with `--cache-ram 1` (1 MiB), i.e. a
degenerate single-tier SSD-only cache with no RAM tier. Under that config
every turn pays a full disk round-trip, every completion save is demoted to
disk immediately (gigabytes written; the v2b store sat at its 4096 MiB quota
ceiling, so compaction ran inside the measured window), and the measured
1.7x–1.8x slowdown at 8k–15k reused tokens is the cost of that write/restore
path, not of multi-level caching. The product goal is a multi-level
RAM+SSD cache; that configuration has not been measured on owner traffic.
The gate's pre-registered thresholds were therefore evaluated on the wrong
mechanism, and the two-strike rule wrongly blocked the correction — fixed
below (a tier-configuration change that alters the mechanism under test is a
new hypothesis).

Also not accepted as evidence of absent demand: the binary
first-turn-after-restart hit metric (0/3, then 1/3) on three restart
boundaries is too small a sample to decide anything, and a missed
post-restart partial hit on a byte-identical preamble points at checkpoint
placement (granularity), not at absent reuse.

### Phase 1c — Measure the actual product: multi-level RAM+SSD on owner traffic

First action: commit the trigger patch in `third_party/llama.cpp` and the
pending `NOW.md` / `docs/SSD_L2_CACHE.md` changes (see "Reading order" — no
per-action approval is needed for submodule commits in this repo).

1. **Decompose the hit-turn cost first, from data already collected.**
   Server logs carry `prompt cache update took … ms` and eviction/disk
   lines; split `ssd_s` on hit turns into restore vs save/demote/compaction.
   Expected finding: save-on-completion of a growing conversation rewrites
   the full state every turn (quadratic bytes), and with RAM=1 MiB each
   write goes straight to disk. Record the decomposition as a receipt before
   changing anything.
2. **Fix the write amplification if step 1 confirms it.** Candidates:
   save-on-completion only when the slot goes idle for N seconds (not per
   turn), or on prompt switch/shutdown only; incremental checkpoint saves;
   never let the store run at its quota ceiling during measurement
   (size the quota with headroom so compaction is not inside the window).
3. **Run the multi-level config on M4:** `--cache-ram` sized realistically
   for the host (start at 4–8 GiB on the 24 GiB M4 with the 8B model),
   `--cache-ssd` with headroom above the working set, on corpus v2b
   extended to all available sessions of that project (18 exist — use more
   restart boundaries than three). Pre-register the definition as v2c
   before running. Baseline arm: cache fully disabled. Optional third arm:
   RAM-only (`--cache-ssd 0`) — this isolates what SSD adds over the RAM
   tier, which is the actual product question.
4. **Metrics for Gate G1c (pre-register, continuous not binary):**
   per-turn paired ratio on every turn, bucketed by `cache_n`; post-restart
   first-turn `cache_n` as a fraction of shared-preamble length; total
   session wall-clock per arm. GO if the SSD arm beats the RAM-only arm on
   cold-start turns and is not net slower than RAM-only elsewhere.
   If SSD adds nothing over RAM-only on this host, that is the earned
   NO_GO — and it is also the finding that the win is in the RAM tier and
   the ladder should optimize that instead.

### Historical: Phase 1 (pre-trigger-patch) root cause

The Phase 1 run (`artifacts/ssd-redesign/20260901T152546Z-4c0f7eab23ef/RUN_SUMMARY.md`)
was executed honestly and its measurements stand: 0 bytes ever reached the
SSD store for solo-developer `--parallel 1` traffic. But its verdict was
premature. Independent code review found that the run conflated two
different facts:

- **What it proved:** the current implementation cannot *write* to the
  persistent tier for this traffic shape. There are two `prompt_save` call
  sites, not one: the idle-sibling-slot loop
  (`tools/server/server-context.cpp` ~2591, multi-slot only) and the
  prompt-switch path in `get_available_slot` (~1817, fires on LRU selection
  or LCP `f_keep < 0.5`). With one growing conversation per server process,
  neither fires. Nothing flushes on request completion or shutdown, and the
  SSD tier only receives RAM-demoted entries
  (`server-task.cpp`, `server_prompt_cache::update`, ~2100).
- **What it did not prove:** that representative traffic lacks persistent
  reuse. Its own data shows reuse potential (byte-identical preambles,
  83% within-process LCP continuity). The cache had nothing to serve
  because nothing was ever saved — a trigger gap, not absent demand.

Therefore the decisive question of Gate G1 remains **open**, and the block
is a small, well-localized patch away from testable.

## Proven scope and results

- Native RAM plus optional SSD L2 prompt/KV caching is implemented.
- SSD L2 remains disabled by default.
- The caller-attested synthetic M4 exact-reuse ratio is `4.9897x`.
- The caller-attested synthetic M4 partial-reuse ratio is `1.5825x`.
- Exact and partial qualification outputs and generated token IDs matched.
- Novel input establishes only no detected latency difference.
- Pressure proves bounded capacity, eviction, and restart for the four-prefix probe.
- Pressure does not prove output or generated-token equality.
- The compatibility key remains caller-attested, not caller-derived.
- Learning-write overhead: closed by the Phase 1 run (no reproducible net
  cost beyond noise once the tier has nothing to write).
- Solo `--parallel 1` traffic: automatic save-to-persistent-tier never
  engages with the current triggers (see Gate G1 status above).

## Claim and execution boundary

- Do not generalize synthetic results to representative owner traffic.
- Do not claim universal speedup, production economics, or default enablement.
- Do not repeat unchanged qualification, novel, learning, or pressure experiments.

## Goal invariant (does not change)

Deliver honest, receipt-backed evidence of whether persistent SSD L2 caching
produces real value on representative owner traffic — and, if it does,
maximize that value through receipt-gated iteration. The evidence bar never
drops: raw receipts, explicit claim boundaries, paired controls where a cost
is claimed, no synthetic-to-representative generalization. A clean,
receipt-backed negative result satisfies the goal; an unsupported positive
claim does not.

## Ordered next plan (gated)

## Owner-trace corpus (authorized data source)

The owner has explicitly authorized using their real local agent-session
histories as workload data for this project: Claude Code session transcripts
(under `~/.claude/projects/`, JSONL) and Codex session logs (under
`~/.codex/`). Rules:

- **Privacy boundary:** this data never leaves the owner's own machines
  (local M1 for processing is fine; the M4 host is the owner's). Never
  upload it, publish it, or embed excerpts in documents — receipts reference
  the corpus by manifest hash, not by content.
- **Processing:** transcripts must be converted into replayable request
  sequences (system preamble + growing transcript + tool results), preserving
  the real prefix-repetition structure. The Phase 1b corpus pipeline
  (`artifacts/ssd-redesign/20260901T163602Z-phase1b-trigger-patch/`,
  corpus v2/v2b builders) is the current implementation — extend it. The
  older PCSS capture harness (`src/pcss/project_workload_capture.py`,
  `PROJECT_NATURALISTIC_CAPTURE.md`) is historical prior art only; its
  proxy-receipt and holdout rules do not apply here.
- **Pre-registration, not post-hoc:** a trace-derived workload is a new,
  versioned Phase 0 definition (`PHASE0_WORKLOAD_DEFINITION v2`), frozen
  with corpus manifest and selection criteria BEFORE any measured run.
  Selecting or trimming traces after seeing results is prohibited — that is
  the "redefining the workload to manufacture a positive" failure mode.
- Heterogeneous traffic is expected and is the point: the corpus should mix
  projects and session shapes. Getting profit despite heterogeneity is what
  the profit ladder's granularity/policy levers are for.

### Phase 1b — Close the trigger gap, then rerun the decisive experiment

1. Implement a save trigger that works for single-slot sequential traffic.
   Candidate designs (pick by evidence, combining is allowed):
   - save-on-completion: flush the slot's state into the prompt cache when
     a completion request finishes (with debouncing so a growing
     conversation does not rewrite the store on every turn);
   - save-on-shutdown: flush live slot state on graceful server exit;
   - periodic idle flush.
   Both existing call sites and the demotion path are located above — this
   is a localized patch to the already-patched server, and it is also the
   concrete upstream-contribution artifact.
2. Rerun the Phase 1 representative experiment against the patched trigger,
   using either the existing frozen Phase 0 definition or a pre-registered
   owner-trace v2 definition (preferred if the corpus pipeline is ready;
   see "Owner-trace corpus" above). This is a legitimate new hypothesis
   under the two-strike rule: the code changed, the design did not.

### Gate G1b — superseded by Phase 1c / Gate G1c (see top of file)

Phase 1b ran and is complete; its verdict was reached under a forced
single-tier config and is superseded. Gate G1c (defined in Phase 1c above)
is the live gate:

- SSD arm beats RAM-only on cold-start turns and is not net slower
  elsewhere: `GO_PROFIT_LADDER` — proceed to Phase 3.
- SSD adds nothing over a realistic RAM tier on this host, with write
  amplification fixed and adequate restart boundaries: `NO_GO_SSD_L2_PRODUCT`
  earned — package the trigger patch and receipts for upstream; then decide
  whether the RAM tier itself is the product to optimize.

### Phase 3 — Profit ladder (only after `GO_PROFIT_LADDER`)

Do not stop at the first confirmed win. Iterate levers one at a time, each
as hypothesis → paired experiment → receipt → keep-or-discard, journaling
continuously. Suggested ladder order (reorder freely with rationale):

1. Trigger tuning: debounce interval, checkpoint granularity/frequency for
   partial-prefix reuse depth — including dynamic/adaptive granularity
   (checkpoint density driven by observed reuse patterns rather than a
   fixed interval), which is the main lever for heterogeneous traffic.
2. Tier budgets: RAM/SSD size sweep on the representative workload
   (hit rate and latency vs budget curve, not a single point).
3. Admission and recency policy: what to demote, what to keep hot,
   restart-persistent recency.
4. Write-path cost: async writes, compaction thresholds, store layout.
5. Workload extensions: multi-project stores, multiple compatibility
   namespaces, warm-start after reboot.
6. Hardening from the old plan as it becomes load-bearing: caller-derived
   fail-closed compatibility identity; larger-cardinality pressure with raw
   output and token receipts.

**Stop condition (loop-until-dry):** the ladder ends when two consecutive
levers each fail to produce a receipt-backed improvement on the
representative workload, or when a lever's cost provably exceeds its
benefit. Record the final cumulative paired result in
`docs/SSD_L2_CACHE.md`.

## Execution host rule (binding)

All live server runs, benchmarks, qualification, pressure, and profit-ladder
experiments run on the dedicated Apple M4 Pro host (24 GiB), reachable as
`ai-server@mac2-llm.local` — the same host as the canonical receipts in
`docs/SSD_L2_CACHE.md`. The local M1 Pro machine is NOT a measurement host:
it is memory-constrained and shared with unrelated development. Use it only
for code edits, unit tests, and lint. A measurement produced on any other
hardware is invalid as gate evidence (the first Phase 1 run made exactly
this mistake and its latency numbers are host-confounded). Every receipt
must record the hardware it ran on.

## Anti-stall rules (binding on the executing agent)

- **Two-strike loop-breaker.** If the same experiment design fails or stalls
  twice without a new hypothesis, do not run it a third time. Either change
  the design under a written new hypothesis, or record the blocker in the
  run journal and move to the next runnable item. A code change to the
  mechanism under test counts as a new hypothesis, and so does a tier
  configuration change that alters which mechanism is being measured
  (single-tier forced-disk vs multi-level RAM+SSD are different mechanisms);
  a bare rerun or a different sample of the same corpus does not.
- **Measure the product, not a proxy.** Experiment configs chosen to
  "guarantee the tier is exercised" (e.g. `--cache-ram 1`) are diagnostics,
  never gate evidence. Gate runs use the configuration a real deployment
  would use on the recorded host.
- **Approach freedom within the evidence bar.** The agent is authorized to
  substitute an alternative experimental design, harness, or measurement
  approach for any planned item without asking, provided the Goal invariant
  holds and the substitution is recorded with its rationale. The plan binds
  the questions and the gate order, not the specific mechanism of answering.
- **Negative results are progress.** A receipt-backed NO-GO or blocker is a
  completed deliverable. Do not keep iterating in search of a positive
  result — but a confirmed positive result is also not a finish line while
  the profit ladder has untested levers (see stop condition).
- **Timebox before pivot.** If a planned item consumes a full session
  without producing either a receipt or a concrete blocker, stop, journal
  the state honestly, and pivot to the cheapest design that can still
  answer the same question.
- **No silent scope changes.** Reordering within a phase and substituting
  designs are allowed; skipping a gate, entering Phase 3 without
  `GO_PROFIT_LADDER`, or redefining the Phase 0 workload to manufacture a
  positive is not.
- **Continuous journaling.** Long runs must append progress to the run
  journal as they go (per lever: hypothesis, receipt path, verdict), so any
  interruption leaves a resumable state.

## Explicitly parked (do not start)

A performance rewrite of the store/server path (e.g. in Rust) is parked
until the profit ladder has proven where the time actually goes on
representative traffic. Optimizing before that evidence exists is premature.

The sequence datastore remains `NO_GO_CURRENT_REDESIGN`. Reopening it
requires a written new hypothesis that addresses serial-canonical K>2
correctness AND the storage-projection failure — absent that, do not touch
it. Historical PCSS documents do not define or authorize next work.
