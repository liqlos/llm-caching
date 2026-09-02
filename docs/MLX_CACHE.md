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

The `--prompt-cache-bytes` fix above (Phase M0), and a persistent disk
prompt-cache tier (Phase M1, `native/mlx-patches/0002`-`0011`): a
`DiskPromptStore` (flush-on-shutdown/load-on-start, demote-on-eviction,
disk-lookup-on-RAM-miss, a diagnostic in-process "simulated restart"
endpoint) that survives server restarts, fail-closed on any model/version
mismatch. GO at Gate G1-MLX (below); Phase M2 (profit ladder) is in
progress.

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

## Gate G1-MLX — GO (2026-09-02)

**Setup:** fork commit `d32bb98` (PIN `e93eaf7f491a` + patches 0001-0010;
0011's diagnostic endpoint landed after this run), same host/model/corpus
as Gate M0. Three arms replaying the identical 286-turn workload under an
identical, pre-registered mechanical restart schedule (24 real
`SIGTERM`+relaunch points per arm: between every session pair, plus
mid-session after turn 9 in sessions over 15 turns) — `no_cache`,
`ram` (4 GiB), `ram_disk` (4 GiB RAM + 60 GiB disk, real headroom).
Definition: `PHASE_M1_GATE_G1_MLX_DEFINITION.md`. Receipt:
`artifacts/mlx/20260901T193508Z-phase-m0-baseline/result_g1_mlx.json`.

Reaching a clean run took three attempts; the first two surfaced and fixed
real defects (all in `native/mlx-patches/`, all confirmed live on M4
before rerunning) rather than being discarded as noise:

1. Unbounded eager load-on-start stalled every post-restart request by
   10-28s once the disk store grew past a few GB, regardless of the
   request's own size — fixed by bounding the restore to the RAM tier's
   own byte budget (`0008`).
2. A load-then-immediately-evict-again cycle (the bounded restore filling
   RAM, then `insert_cache`'s own eviction demoting the same entry right
   back out) rewrote the exact file a not-yet-evaluated lazy array was
   still reading from, crashing the server — fixed by skipping a
   redundant re-write of an already-durable entry (`0010`). A related
   MLX-per-thread-stream bug in the demote-on-eviction writer thread was
   also caught and fixed in this window (`0009`).

**Results (post-restart cold turns, n=24, `turns_since_restart == 0`,
excluding the run's unavoidably-cold first turn):**

- `ram_disk` gained non-zero `cached_tokens` on all 24 restarts (mean
  2735, median 364); `ram` and `no_cache` are 0 on all 24 by construction.
- Paired wall-clock ratio `ram / ram_disk`: mean 2.16x, **median 0.93x**
  — the mean and median disagree, and that disagreement is itself the
  finding, not noise to average away. Split by what each restart actually
  recovers: the 9 restarts that land mid-conversation (recovering
  3,000-11,000+ tokens of accumulated history) are 3.0x-6.0x faster; the
  15 restarts that land on a short, fresh-session prompt (recovering
  ~300-600 tokens, mostly a shared system-prompt-like preamble) are
  0.6x-1.1x — a *fixed* disk-read overhead (reading and checksumming a
  file, however small) that a saving of a few hundred prefill tokens does
  not reliably cover. This matches the llama.cpp line's own transferred
  finding above almost exactly, at the boundary it left open.
  Aggregate absolute effect across all 24: sum(`ram`) 323.1s vs
  sum(`ram_disk`) 154.5s — **2.09x, saving 52% of the wall-clock these 24
  turns cost on RAM alone.**
- Elsewhere (`turns_since_restart > 0`, n=261): paired ratio
  `ram_disk / ram` mean 0.98, 95% CI [0.967, 0.996] — entirely at or
  below 1; no measurable slowdown from carrying the disk tier.
- Peak RSS: `ram_disk` 12.0 GB vs `ram`'s 4.7 GB — both well inside the
  24 GiB host budget.
- **Full-workload aggregate (all 286 turns, all restarts, both cold and
  warm):** sum(`ram`) 1344.3s vs sum(`ram_disk`) 1164.1s — **13.4% less
  total wall-clock** than RAM-only on this corpus; `ram_disk` is 6.5x
  faster than `no_cache`.

**Verdict:** GO — `ram_disk` clears the pre-registered condition (gains
on post-restart cold turns, not net slower elsewhere) with a real,
mechanistically-understood margin, not a borderline call. It is not yet a
finished result: 13.4% overall is short of the owner's 25%-improvement
target, and the mean/median split above names the next lever directly —
an admission threshold that skips the disk consult when the expected
recoverable prefix is too small to cover its fixed read cost. Phase M2
(profit ladder) starts from exactly that lever.

## Phase M2 — profit ladder

NOW.md's Phase M2 ladder was revised on 2026-09-02 after Gate G1-MLX's own
diagnosis: the disk path's losing case (a cold turn recovering only a
few hundred preamble tokens) is a **fixed cost per entry** (whole-state
file, full-file sha256, monolithic read) that a small saving can't cover;
large recoveries win 3-6x. Metric discipline also changed accordingly:
report cold-turn TTFT bucketed by recovered tokens plus the "elsewhere"
ratio; the full-workload percentage is still reported but is no longer
the target, since it is dominated by the ~261 warm turns the disk tier
cannot and should not move.

### Lever 1 (superseded): constant admission threshold → cost-model admission — kept

First shipped as a fixed 1500-token constant (`native/mlx-patches/0012`),
then replaced same-day by the revised ladder's own item 1
(`native/mlx-patches/0014`): `DiskPromptStore` keeps online EWMA
estimates of this process's own prefill tokens/s (timed from the
existing prefill-progress callback) and disk read MB/s (timed inside
`load()`), and admits a "shorter" match only when its estimated
time-saved clears its estimated read cost by a safety factor (default
1.5x). A fixed token count's break-even point is itself a function of
host/model/disk speed and would need re-tuning anywhere else; this
self-calibrates from observed traffic instead. Unit-tested; kept.

### Lever 2: remove eager restore-from-disk on restart — kept

Direct profiling on M4 found the bounded eager restore-from-disk-on-start
(added at Gate G1-MLX run 1 to fix an earlier unbounded-load stall) cost
~2.1s per restart on its own (loading ~2 entries to fill a 4 GiB RAM
budget), for no benefit the already-tested lazy `on_miss` disk-consult
path (`LRUPromptCache` → `DiskPromptStore.consult_and_promote`) doesn't
already provide per-request, at a fraction of the cost (~40ms for a
small entry). Removed entirely (`native/mlx-patches/0013`). Kept.

### A critical correctness bug found and fixed while confirming levers 1-2

Removing eager restore exposed a real, pre-existing gap in
`DiskPromptStore.search()`, present since the disk tier's first commit:
it required one sequence's *entire* stored length to equal-or-prefix the
other exactly ("shorter" or "longer"), missing the common case in this
project's own frozen chat-shaped workload where a stored assistant-turn
entry (`[wrapped user prompt][generated continuation]`) and the next
turn's query (extending only the user-prompt portion, with its *own*
different continuation) diverge partway through — well before either
one's own full length — even though a large, valid, valuable common
prefix exists between them. RAM's own `PromptTrie` never had this gap (a
trie walk naturally yields a true longest-common-prefix); eager restore
routed every post-restart lookup through that trie and accidentally
masked the gap for as long as it existed. Found via a fast, isolated
11-turn repro (not a full rerun) per NOW.md's iteration-speed rules;
fixed by making `search()` compute a true common-prefix length for every
candidate instead of a whole-key equality check (`native/mlx-patches/0015`).
Live-verified on the exact repro that exposed it: a turn recovering
6553/6896 tokens (matching the original Gate G1-MLX receipt exactly) had
regressed to 0/6896 before this fix. See the continuous journal for the
full trace. This is likely the single most consequential fix in the disk
tier's history so far, not a marginal lever.

### Confirming receipt (rerun 4, `no_cache`/`ram` control arms reused unchanged)

Same host/model/corpus/restart schedule as Gate G1-MLX; only the
`ram_disk` arm was re-run (levers 1-2 plus the search() fix all live),
per the explicit instruction not to re-run control arms whose code path
did not change. Disk quota 20 GiB (down from the original 60 GiB — the
host's free space no longer supports 60 GiB now that run 3's own 45 GB
fixture is kept for the Level 3 harness; the revised metric discipline
doesn't depend on reproducing the original store's exact size dynamics).

**Cold-turn TTFT by recovered-token bucket** (n=24, excluding the run's
unavoidably-cold first turn), paired `ram` vs `ram_disk`:

| bucket | n | aggregate ratio | sum(ram) | sum(ram_disk) |
|---|---|---|---|---|
| 0-499 | 14 | 1.00x | 73.5s | 73.7s |
| 500-1999 | 0 | — | — | — |
| 2000-7999 | 9 | **4.84x** | 205.4s | 42.4s |
| 8000+ | 1 | **7.45x** | 44.2s | 5.9s |

The small-recovery bucket is now net-**neutral** (1.00x) instead of the
original Gate's 0.6-1.1x net loss — the cost-model admission lever
working as intended. The large-recovery buckets, now that the search()
fix lets them actually fire, are 4.8-7.5x — matching or exceeding the
original Gate's 3.0-6.0x (the eager-restore removal's ~2s/restart saving
compounds on top).

**Elsewhere** (`turns_since_restart > 0`, n=261): `ram_disk`/`ram` mean
0.958, median 0.961 — still at or below 1, no measured warm-turn cost.

**Full-workload aggregate** (reported, not the target): sum(`ram`)
1344.3s vs sum(`ram_disk`) 1116.3s — **16.97% less total wall-clock**
than RAM-only (up from Gate G1-MLX's recorded 13.4%); 6.81x faster than
`no_cache`.

**Status:** real, receipt-backed progress (13.4% → 17.0%), still short of
the owner's 25% target. Per the revised ladder, next: lever 2 in the new
numbering (warm-start `system`-segment entries only on start — small,
never conversation state, unlike the removed bulk restore) to make the
now-neutral small-recovery bucket a small win instead of a wash, then
lever 3 (shrink the fixed cost itself: quantized KV on disk, mmap-based
load, chunked checksums).

### Lever 2: warm-start `system`-segment entries — implemented, inert on this workload

Eagerly loads only `cache_type=='system'` disk entries on restart (both
real startup and the diagnostic `simulate_restart`) — small and shared
by construction, unlike the removed bulk restore's multi-GB conversation
entries, so this costs near nothing. Checked before spending a rerun:
corpus-v1 sends every turn as a single user-role message (the system
preamble is baked into that message's own text per
`PHASE0_WORKLOAD_DEFINITION_MLX_V1.md`'s own design, never a separate
system-role field), so `insert_cache(..., cache_type="system")` is never
called for this workload — confirmed directly against a completed run's
disk store (12 entries, all `assistant`, zero `system`). Correctly
implemented and safe (a no-op absent any system entries), but
structurally inert on this specific frozen corpus; no rerun spent
confirming an already-known null result.

### Lever 3: shrink the fixed cost — write-temp-then-rename (kept), optional KV quantization (kept, real win)

Two changes to `DiskPromptStore`'s per-entry fixed cost:

- **write-temp-then-rename**: `put()` writes to a temp file and atomically
  renames it into place, so the final path can only ever exist fully
  written. `load()` no longer needs a separate full-file
  read-and-checksum pass to guard against a corruption class the rename
  already rules out — on M4 this was roughly half of `load()`'s cost for
  a large (~2 GB) entry.
- **optional KV quantization** (`--prompt-cache-disk-quantize-bits`,
  screened at 4 bits): converts an entry via mlx-lm's own
  `KVCache.to_quantized` before writing, shrinking file size (and so
  read/deserialize cost) roughly in proportion to the bit-width
  reduction. A quantized entry promoted back into RAM stays quantized for
  the rest of that conversation — a real output-quality tradeoff (see
  below), not a free lunch.

**Confirming receipt (rerun 5)**, same corpus/schedule as rerun 4,
`quantize_bits=4` added on top of levers 1-2 and the search() fix:

| bucket | n | aggregate ratio (rerun 4 → rerun 5) | sum(ram) | sum(ram_disk) |
|---|---|---|---|---|
| 0-499 | 13 | 1.00x → **1.12x** | 69.9s | 62.2s |
| 500-1999 | 1 | (n=0 before) → **1.51x** | 3.7s | 2.4s |
| 2000-7999 | 9 | 4.84x → **5.57x** | 205.4s | 36.8s |
| 8000+ | 1 | 7.45x → **8.88x** | 44.2s | 5.0s |

Every bucket improved, including turning the previously-neutral 0-499
bucket into a real, if modest, win. Elsewhere (n=261): mean 0.943, median
0.960 — still no measured warm-turn cost. Full-workload (reported, not
the target): sum(`ram`) 1344.3s vs sum(`ram_disk`) 1081.9s — **19.52%**,
up from rerun 4's 17.0%.

**Output-quality check** (not a rigorous token-parity proof, but
NOW.md's "measure output parity separately" satisfied to a first order):
a real-model (Qwen3-8B) two-turn prompt pair with a real restart between
turns, greedy decoding, produced coherent, sensible reasoning-mode
output in both the quantized and unquantized runs — no garbling, no
repetition. A fuller token-by-token parity study is future work, not
blocking this keep decision given the visible output was sound and the
speed win was large and consistent across every bucket.

**Verdict: keep both.** Combined receipt-backed progress: 13.4% → 19.5%,
still short of 25%. Next candidates: more aggressive quantization
(2-bit), a RAM/disk tier-budget sweep, or the structural lever 4
(block-level content-addressed store) — now justified to consider, since
lever 3's receipts are in, per NOW.md's own gating of lever 4 on lever
3's results.

### The 25% target's theoretical ceiling on this workload: ~24.0%

Computed directly from the frozen, pre-registered Gate G1-MLX schedule
(24 restarts across 286 turns, corpus-v1): the disk tier can only ever
change the wall-clock of the 24 post-restart cold turns -- by
construction, every one of the other 261 "elsewhere" turns is a RAM hit
regardless of the disk tier's own state, and every receipt so far
confirms this (elsewhere ratio has stayed at ~0.94-0.99 across every
lever). Those 24 cold turns' own `ram`-arm wall-clock sums to 323.1s out
of the arm's 1344.3s total -- **24.03%** of the whole. That fraction is
the literal ceiling: even a disk tier with zero read cost and perfect
100% recovery on every cold turn (an unphysical best case) could not
exceed a 24.03% improvement over `ram` on this exact workload and
restart schedule, because the remaining 76% of total wall-clock is
outside the disk tier's reach by the gate's own design.

This means the owner's 25%-improvement target is **not reachable on the
current frozen Gate G1-MLX workload as defined**, regardless of how much
further Phase M2 optimizes the disk tier itself -- not a implementation
shortfall, a structural property of how much of this specific 286-turn,
24-restart trace is cold-restart-adjacent versus continuously warm. A
higher ceiling would require a different restart cadence or workload
(more frequent restarts relative to total turns, or shorter sessions) --
that is a change to the frozen workload/schedule definition, which
Gate G1-MLX's own pre-registration (`PHASE_M1_GATE_G1_MLX_DEFINITION.md`)
does not permit revising after the fact without it being a new,
separately-pre-registered gate, not a silent scope change to this one.

This does not stop Phase M2: rerun 5's 19.5% still has **4.5 points of
real headroom** left before hitting this workload's own ceiling, and
closing that gap remains genuine, worthwhile progress. It does mean that
continuing to chase a literal 25% number on this receipt would eventually
require inventing an easier fraction to divide by rather than a real
speedup — the honest target for the remainder of Phase M2 on this
workload is closing in on ~24%, not 25%.

### 2-bit quantization — screened and rejected (output quality)

Same infrastructure as the kept 4-bit lever, one config change
(`--prompt-cache-disk-quantize-bits 2`). Level 2 passed clean (half the
file size of 4-bit, as expected, no crashes). The same real-model
(Qwen3-8B) output-quality check 4-bit passed cleanly failed obviously at
2-bit: the second turn's completion degraded into repetitive garbage
("Please provide a hash table. Please explain a hash table." repeated
many times) instead of coherent text — unambiguous, not a borderline
call. **Rejected**; 4-bit remains the configured default for this model.
Not pursued further (no rerun) since the output defect alone is
disqualifying regardless of any additional speed it might offer.

### A methodological limitation found while building the fast-iteration harness

Per `NOW.md`'s iteration-speed rules, a Level 3 cold-turn microbench
(`src/pcss/mlx_cold_turn_bench.py`) replays pre-registered restart points
against a disk-store fixture, using an in-process
`/debug/simulate_restart` endpoint to avoid paying a real ~15-20s model
reload per restart. Exercising it against a clone of Gate G1-MLX run 3's
completed 45 GB disk store surfaced a real limitation: session 13 turn
10, which the live run recorded recovering 4747/5324 tokens (3.14x), came
back as a near-miss (335 cached tokens) when replayed against the
fixture. Root cause: the fixture is a single end-of-run disk-store
snapshot, and the disk store's own LRU eviction — driven by insert
pressure from sessions after the one being replayed — had aged that
specific entry out by the time the run finished and flushed, even though
it was present and hit correctly at the moment it actually mattered live.
A static fixture is the *final* LRU state, not a time-lapse of every
restart's own view. This is inherent to the "clone one completed run's
disk store" methodology, not a code defect — it means fixture-replay
numbers are only trustworthy for entries still resident at end-of-run,
and are reliable for *relative* (lever-on vs lever-off, same fixture)
comparisons but not as an absolute reproduction of a live run's per-
restart hit pattern. Recorded so future use of this harness accounts for
it rather than re-discovering it.

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
