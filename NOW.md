# Current handoff — Line D: CLOSED (persistent cross-session datastore speculation on Apple Silicon)

`NOW.md` is the sole authority for scope, plan, gates, and execution rules.
`docs/CLOSED_LINES.md` records what was tried and closed. Everything else
in the tree (`PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `qualification/`, the
`llamacpp_*` / `exact_prompt_*` / `live_server_*` lab modules) is historical
prior art and binds nothing.

## Addendum (2026-09-03, live speedup verified)

The owner asked, correctly, whether a real >=25% speedup had actually been
demonstrated live on varied prompts, rather than inferred from an offline
acceptance-rate replay. It had not; this addendum closes that gap.

**Answer: yes, real and reproduced twice, but not from anything this
project built.** A live, paired decode-tok/s benchmark
(`g1d/live_speedup_benchmark.py`, receipt at `g1d/LIVE_SPEEDUP_RECEIPT.md`)
on the real D0 target, 14 varied prompts across 6 projects (context
414-11,551 tokens): mlx-dspark's own shipped DFlash+lookup hybrid drafter
(default settings, no datastore) measured **geomean 1.501x (50% faster)**,
10/14 prompts at or above 1.25x. An independent first run (12 prompts, 2
mildly contaminated by an unrelated host process) agreed in order of
magnitude at 1.395x geomean.

**This project's own contribution -- per-request n-gram lookup and the
persistent datastore -- is a clean NET LOSS live**, geomean ~0.26x (about
4x slower than plain decoding) in both runs, zero of 26 paired
measurements beating baseline. The `ngram_min=1` policy the offline D0
algorithm sweep favored (+15.6% raw accepted-tokens-per-round with no cost
model) is a large net loss once the real, context-dependent verify cost is
paid on every mostly-rejected round. This is the clearest evidence in this
project's history that a raw-acceptance metric without a cost model can
rank a drafting policy backwards relative to live wall-clock.

**Consequence for the closure below: unchanged, reinforced.** The real,
working speedup this host already delivers comes entirely from
mlx-dspark's own shipped drafter, which required no engineering from this
project. Nothing here reopens Line D -- it sharpens why it is closed: the
datastore doesn't just fail to add enough marginal value (Gate D0), it
actively hurts live throughput under the policy that looked best offline.

## Status (2026-09-03, Line D closed)

**Line D is closed.** Gate D0 (throughput) was NO_GO, the exactness gate
(correctness) passed cleanly, and the one remaining real lever found in
this work -- long-context verify cost, ~34% above a bandwidth-bound floor
at 16k -- is a general Apple-Silicon engine lever unrelated to the
datastore, and the owner decided not to expand this project's scope to
pursue it (2026-09-03; see decision record below). Nothing further is
planned on this line. The three sections immediately below are the
as-answered record, kept for reference; do not resume work from them
without a new scope decision.

## Decision record (2026-09-03)

1. **Gate D0 verdict: accepted as NO_GO, no restatement.** Criterion 1
   (datastore's margin over per-request lookup) measured 1.014 against a
   1.20 threshold, robust across every verify curve, every algorithmic
   variant swept, every project, and a 6x growth in index size. See
   `d0b/GATE_D0_RECEIPT.md`.
2. **Exactness standard: "greedy up to floating-point ties," not
   bit-identical.** Three independent agents, working read-only with no
   coordination, unanimously reached this from the fork's own source.
   Datastore neutrality (no divergence above ~2 bf16 ulps between
   datastore-on and datastore-off) gate: PASS. See the Goal invariant
   below and `d1/exactness_gate_v2_summary.json`.
3. **Long-context verify cost: real, large (34% at 16k), and out of
   scope.** Measured directly (not extrapolated) on the M4:
   `d0c/verify_curve_ctx.json`. Offered to the owner as a scope-expansion
   decision (new line: general Apple-Silicon verify-cost engineering,
   unrelated to the datastore) versus closing Line D outright. Owner chose
   **close the line, do not expand scope** -- this lever may be picked up
   by a future, separately-scoped project, but no further GPU time or
   engineering goes to it here.

## Status (2026-09-02, Gate D0 answered — superseded by the closure above)

**Gate D0 is NO_GO as pre-registered.** The receipt is
`artifacts/mlx/20260902T152845Z-phase-d0-marathon/d0b/GATE_D0_RECEIPT.md`
(+ `.json`, both local-only, not in git); it carries every number and the file
and key each one came from. Criterion 1, the global datastore's margin over the
per-request lookup mlx-dspark already ships, measured **1.014** against a 1.20
threshold (1.014-1.071 across all three verify curves). Criterion 2, the product
against no drafting, measured **2.005** against a 1.30 threshold. The gate needs
both, so it fails; the disagreement between the two is the finding, because
criterion 1 prices only the crossing of session boundaries while criterion 2
prices the product, and the product is delivered almost entirely by per-request
lookup. The pre-registered escape clause of a profitable subset of projects does
not apply: no project reaches 1.20 and one is below parity at 0.974.

Evidence is 319 real `Qwen3-8B-8bit` greedy turns replayed offline against a
1.46M-token index. The margin does not grow with the cache: over a 6x larger
index it drifts 0.007, so the remaining turns of the planned 600 cannot carry
it. It is also not an artefact of a weak matcher -- selection weight, match-length
cap, iterative refill, minimum match length, and the SuffixDecoding variants
(frequency scoring, adaptive length, tree drafts) were each swept, and while the
absolute level moves from 2.47 to 3.06 tokens per step, the margin stays inside
1.01-1.07 throughout. A second agent, given only the raw data and none of this
reasoning, reached the same ratio independently. The one axis that did matter is
the drafter's minimum match length: demanding a trigram before proposing left it
silent in half of all rounds, and dropping to a single-token match is worth
+15.6%.

Two caveats bind anything quoted from the receipt. Everything at 8k context and
beyond rides on an extrapolated per-row depth slope, not a measured curve -- the
host calibration is tagged `ctx512`. And the largest lever found anywhere in this
work is outside the datastore: each extra verify row costs close to a full KV
read, so the optimal draft width collapses from 8 to 3 as context grows, and a
bandwidth-bound multi-row attention would be worth more than the datastore ever
was, for every drafter and every model on the host.

In flight: `d1/exactness_gate_v2.py` on M4, checking on the real D0 target that
the datastore never changes output and reporting the speculative-vs-plain
identity rate, in both the stock and the server kernel classes. The D0b generator
is stopped to free the GPU and resumes with `--resume`; completing it only
lengthens the size curve. Earlier status claims now superseded: prefill *is* a
large part of the 4.4 tok/s figure (the 32k-char sliding tail window makes each
prompt a fresh string, so the prompt cache reuses almost nothing), and the memo's
roofline should be read as 29 tok/s at 512 context and 23 at 8k.

## Scope decision (owner, 2026-09-02)

**Not prefix / prefill KV caching** — oMLX ships it; see
`docs/CLOSED_LINES.md`.

**What this project builds:** decode-side acceleration by speculating from
a **persistent, cross-session datastore of previously generated and seen
token sequences** ("computed token sequences saved to SSD"), verified by
the target model. Output is correct up to floating-point ties (see Goal
invariant below; the earlier "bit-identical" framing was retracted
2026-09-03). Engine: MLX on the owner's Macs. Traffic: agentic coding.

## Research memos (read before designing or measuring anything)

- `docs/research/PRIOR_ART_DATASTORE_SPECULATION.md` — who has what;
  verdict: the niche is open.
- `docs/research/DATASTORE_DESIGN_STUDY.md` — index structures,
  persistence, draft policy, offline evaluation, logging schema.
- `docs/research/APPLE_SILICON_VERIFY_ECONOMICS.md` — why verify cost is
  what it is on M-series, per-target speedup ceilings, kernels to reuse.
- `docs/research/REPO_AUDIT_2026-09-02.md` — the post-pivot cleanup record.
- `docs/research/APPLE_SILICON_HEADROOM.md` — where the host's decode
  time actually goes, the per-row verify depth slope, and which levers
  are worth pulling; supersedes the roofline figures in
  `APPLE_SILICON_VERIFY_ECONOMICS.md`.
- `docs/research/DATASTORE_FUNDAMENTAL_PROFIT.md` — the oracle bound on
  occurrence selection and why count statistics carry little signal on
  this workload.

## Strategic position (research pass, 2026-09-02)

The field confirms the idea and has already built most of the stack. What
exists must be reused, not rebuilt; the one open gap is where this project
lives. Verified by the exhaustive prior-art search: **nobody, on any
platform, ships a drafting datastore that is (a) fed by completed responses
across requests, (b) serialized to disk and reloaded, and (c) composed
with a model drafter.** vLLM's SuffixDecoding global tree lives only for
the process (no serializer in `suffix_tree.h`); llama.cpp's 2024
`examples/lookup` n-gram cache is the only persisted one and is unused
(≤1.11x, and `llama-server` never saves it — `save_dynamic=false; // TODO`);
every Apple Silicon lookup drafter (mlx-dspark, Rapid-MLX 3.6k★,
mlx-serve, lookup-spec-metal, ngram-engine) is per-request only.

**Already built on Apple Silicon (do not rebuild):**

- **Model-based drafters, lossless, no training:** DFlash 2 / DSpark via
  `ARahim3/mlx-dspark` (MIT, active daily) — 3.63x on Qwen3.8-27B-8bit,
  2.78x Gemma-4-12B, 2.40x Ornith-9B on an M4 Pro; native MTP via
  oMLX / mlx-vlm / MTPLX — 1.2x–2.6x depending on model. These are the
  workload-agnostic baseline any datastore method must add to, not
  replace.
- **Per-request lookup drafting:** `mlx-dspark` already has prompt-lookup
  n-gram drafting (`src/mlx_dspark/lookup.py`, ~280 lines): trigram+
  suffix match against the *current* prompt + generation, hybridised with
  DFlash by default, match-scaled long drafts up to 32 tokens, measured
  4.5x on copy-heavy code editing (Gemma-12B, 75 tok/s); mlx-dspark's own
  contract for this is "greedy up to fp ties," not bit-identical (see Goal
  invariant). It is **per-request only**: the index dies with the request.
- **Serving:** mlx-dspark serves OpenAI-compatible with prefix caching and
  a Mac app; oMLX serves OpenAI + Anthropic with the SSD prefix cache and
  consumes mlx-vlm's speculative stack (drafter kinds `dflash`, `mtp`,
  `eagle3`; per-family exact verifiers; DDTree).
- **Not a destination:** mlx-lm. Its maintainers closed every n-gram /
  prompt-lookup PR in 2026 (#1297, #1508, #1557, #1498 — "beyond our
  capacity to review"); its own speculative path has an open exactness
  bug (#1423) and cannot rewind hybrid-model caches (#1446). Do not build
  on or target mlx-lm's speculative loop.

**The open gap — this project's niche:** a **global, persistent datastore
across sessions** (SuffixDecoding's *global suffix tree* over previous
outputs, NeurIPS 2025 spotlight, `arXiv:2411.04975`), on Apple Silicon,
plugged into the existing hybrid drafting stack. The paper's ablation is
the thesis: the global tree beats the per-request tree on most tasks, and
speedup keeps rising as more past outputs are cached (adaptive online;
~10.75 bytes/token; 2.5x mean on SWE-Bench, 5.3x on a multi-agent SQL
workflow, 1.7x over prompt-lookup). Its C++ suffix tree
(`snowflakedb/ArcticInference/csrc/suffix_decoding/`, nanobind,
Apache-2.0, no CUDA dependency in that directory) is reusable standalone.
Nobody has this on Mac: `fmsouza/lookup-spec-metal` (0 stars) is at
"offline acceptance, no decode loop" for per-request lookup only.

**Apple Silicon economics that shape every decision** (details and
sources in `APPLE_SILICON_VERIFY_ECONOMICS.md`):

- The linear verify curve is a **kernel-dispatch artifact, not hardware**:
  below MLX's `qmv` batch limit (13 rows on M3/M4/M5, 6 on M1/M2) the
  quantized matmul re-reads and re-dequantizes the whole weight matrix per
  input row. Physics says a 4-bit model on an M4 Pro should verify ~6
  tokens for the price of one and an 8-bit model ~12 before compute takes
  over; stock MLX 0.32.2 is 2.5–3.5x off that in exactly the 2..12 window.
  A reusable fix exists (`small_m_qmm.py` in mlx-dspark, MIT: widths 6–8
  cost about width 5) but is not bit-identical (1–2 ulps); mlx-vlm's
  per-family exact verifiers are bit-identical at linear cost. **This is a
  fork in the road for a project promising bit-identical output — see Goal
  invariant.**
- A second cliff is in attention: MLX's vector SDPA path needs
  `q_len ≤ 8` and `q_len × gqa ≤ 32`, so Qwen3.8-27B (gqa 6) falls off at
  width 6 (2.9x per call at 32k context), and verify cost grows with
  context depth. mlx-dspark's `sdpa_split.py` works around it. On 27B the
  practical draft width is ~5 at long context unless that workaround is
  adopted.
- ANE and the CPU matrix units are prefill tools only (fixed shapes,
  weight duplication, shared bus); they do not help verify at width 8–32.
- **Per-target ceilings on the 24 GiB M4 Pro** (model, A = accepted
  tokens per round): Qwen3-8B-8bit ~26 tok/s baseline, up to ~6.5x at
  A=12 and ~8.4x at A=24 — the best ratio; Qwen3.8-27B-4bit ~14 tok/s,
  ~4.5x ceiling, no benefit past width ~16 — the best quality; 8B-4bit
  saturates near width 8; 27B-8bit and Qwen3.8-Flash-Next do not fit;
  the 35B-A3B MoE is structurally poor (each extra draft token activates
  new experts; A=3 is a wash). **M1 Pro is linear from width 2 and only
  fits 8B/12B-4bit: it is not a speculation host.**
- Plain-language: the method is model-agnostic; the gain is accepted
  tokens per round ÷ round cost in decode steps, and the denominator
  changes with bits/param and step length. 8-bit weights widen the free
  verify window; short steps make fixed overheads dominate.
- Measurement doctrine: benchmark dependent chains with rotated weights
  per (chip × mlx version); independent-call microbenchmarks mispriced a
  kernel by 1.13x vs −26%.
- Cautions from the evidence: datastore drafting **regresses** on some
  models/workloads (Rapid-MLX: 0.77x on JSONL, 0.86–0.95x on GPT-OSS /
  Qwen3.6; thc1006: all-negative on an A3B MoE). A per-request acceptance
  gate that turns drafting off when it does not pay is mandatory, not a
  lever; and the gain must be shown on top of DFlash/MTP, never only
  against plain decoding.
- Frontier for later: AdaPLD (June 2026) adds semantic-similarity
  fallback when lexical retrieval fails (3.1x on code editing, +0.5%
  verify cost); SAM-Decoding (suffix automaton, O(1)/step).

## Why the 2026-08-31 NO_GO does not bind this line

That verdict (`NO_GO_CURRENT_REDESIGN`) had two causes specific to the
llama.cpp C++ bridge: non-canonical K>2 verification (a bridge defect —
exact verification is what mlx-dspark / mlx-vlm already do) and a
datastore that did not fit 256 MiB (a suffix tree over text tokens is
~10 B/token). New hypothesis, not a rerun.

## Goal invariant

Receipt-backed evidence that a persistent cross-session datastore adds
decode speed **on top of the best existing per-request lookup + model
drafter** on the owner's representative traffic, with output correct up to
**floating-point ties** (greedy: token identity except where the target's
own top-2 logits differ by at most ~2 bf16 ulps; sampling: distribution-
preserving). Evidence bar as before: raw receipts, pre-registration, paired
arms, no post-hoc workload selection. A clean negative satisfies the goal.

Exactness decision (revised 2026-09-03, three independent agents voted
unanimously on the substance after `exactness_gate_v2` ran on the real D0
target): the earlier claim of a "bit-identical verify path (singleton-
numerics exact verifiers)" was FALSE for this fork and is retracted. No
singleton-numerics exact verifier exists in mlx-dspark; that class of
verifier is mlx-vlm's, a different vehicle (see
`APPLE_SILICON_VERIFY_ECONOMICS.md`). mlx-dspark's own documented contract
(`lookup.py`, `generate.py`, `CHANGELOG.md`) has always been "greedy up to
floating-point ties," never bit-identity, because a width-W verify forward
(quantized matmul, M=W) takes a different kernel path from width-1 plain
decode (M=1) and non-associative bf16 reduction can flip an argmax exactly
at a near-tie. `exactness_gate_v2` on `Qwen3-8B-8bit`, 24 real turns x 200
tokens, found speculative-vs-plain identity of 14/24 turns (58%) in stock
kernels and 13-14/24 in the server's own kernels (small-M MMA, SDPA split);
every divergence's top-2 logit margin was 0.0, 0.125 or 0.25 nats -- 0/1/2
bf16 ulps, matching the fork's own measured ulp drift for batched quantized
matmul (`small_m_qmm.py`) and its documented ~0.5%/token near-tie flip
rate. The datastore introduces no new divergence class: where the
datastore-off and datastore-on arms both diverged from plain, they
diverged at the identical token index in half the cases and within 1
token in the rest, because turning the datastore on changes the verify
round's width, not the numerics. **The gate this project can actually own
is datastore neutrality, not bit-identity to plain decoding**: does the
datastore-on arm match the datastore-off arm at the same width? Any
divergence between those two, or any divergence at a margin above ~2 bf16
ulps, stops the line as a real defect. Receipts report the measured
turn-level identity rate to plain decoding as a fact, not a pass/fail
gate. The small-M and SDPA-split kernels stay opt-in as before, now for
speed rather than exactness, since they don't change this picture.

## Open questions for the owner (answer before D1, not before D0)

1. Which local model will actually serve agentic coding on the 24 GiB
   M4 Pro (and on the M1)? The answer sets the baseline drafter (DFlash 2
   exists for Qwen3.8-27B / Gemma-4 / Qwen3-8B; MTP for Qwen3.5+) and the
   headroom. D0 runs on `Qwen3-8B-8bit` by default.
2. Is the daily loop local-model-driven at all today? If the owner's real
   agent traffic goes to cloud Claude, this project is research/product
   for other Mac users, and the corpus is a proxy for their traffic too.

## Ordered plan (gated)

### Phase D0 — Offline opportunity study (no engine code)

The decisive question is what the **global** datastore adds over the
per-request lookup that already exists. Measure that before building.

1. **D0a — transcript proxy (minutes, local M1 is fine).** From the
   owner-trace corpus, assistant outputs in chronological order,
   tokenized with the target tokenizer. Simulate greedy speculation per
   position with three drafters: (a) per-request n-gram/suffix over
   prompt + output so far (= mlx-dspark's `NGramIndex` semantics:
   trigram+ match, latest occurrence, match-scaled length up to 32);
   (b) global suffix tree over all earlier sessions' outputs *and
   prompts*; (c) both, global preferred on longer match. Report mean
   accepted tokens per round and tokens/pass under the M-series cost
   model (round cost = 1 + width-dependent verify factor from
   mlx-dspark's measured curve), at widths 8 / 16 / 32, per project and
   overall; plus the paper's "structuredness" entropy on 100 outputs.
   Pre-register thresholds before running. Proxy caveat stated.
2. **D0b — real-model outputs (hours, M4, background).** Generate full
   responses with the D0 model (greedy) on corpus prompts in session
   order; global tree learned from sessions 1..k, measured on k+1; same
   metrics. Reuse ArcticInference's `simulator.py` (exact greedy
   acceptance simulation) rather than writing one. Also record the tree's
   measured bytes/token per project and a datastore-size curve (truncate
   training sessions) — the paper says gain keeps rising with history;
   verify it here. This is the number that counts.
3. **D0c — verify-cost curve (10 min, M4).** Measure the D0 model's
   verify cost vs width 1..32 on this host as a dependent chain with
   rotated weights (not independent calls), or read mlx-dspark's cached
   curve, so tokens/pass converts to wall-clock honestly.
4. **D0 model choice:** default `Qwen3-8B-8bit` (best ratio on this
   host); the owner's stated daily model `Qwen3.8-27B-4bit` is the second
   arm once the pipeline works — its long-context width cliff at 6 must be
   in the cost curve.

**Gate D0 (pre-registered):** global (c) must beat per-request (a) by
≥ 20% in projected wall-clock at the best width on D0b, and (c) must
project ≥ 1.3x over plain decoding. Below either → earned NO_GO for this
workload; record entropy so the result transfers. Between → proceed only
if a per-project split shows a clearly profitable subset.

### Phase D1 — Global datastore drafter inside the existing stack

Build **into `mlx-dspark`** (fork `liqlos/mlx-dspark`, MIT), not from
scratch and not on mlx-lm. Design per `DATASTORE_DESIGN_STUDY.md`:

1. **Datastore:** ArcticInference's suffix tree (C++, nanobind,
   Apache-2.0; ~4 µs/token update, ~12 µs lookup, native frequency
   scoring and per-sequence removal) **kept entirely in RAM** under a byte
   quota (1–4 GiB on the 24 GiB host; a single NVMe page fault costs a
   whole round). Persistence = an append-only CRC'd token log per
   (tokenizer, project) namespace, replayed on start (~40 s at 10^7
   tokens), FIFO/LRU eviction by sequence, idle compaction; snapshot only
   if replay is measured slow. Add the serializer the upstream tree lacks.
   Bytes/token is workload-dependent (~10 B/token on repetitive output,
   hundreds on high-entropy text) — D0a measures it per project and
   `max_depth` is the main knob. No block storage: that concern was a
   KV-cache leftover.
2. **Drafter:** extend `lookup.py`'s proposal step. Every source
   (per-request index, global tree, DFlash) is scored in *expected
   accepted tokens*; draft only when `E[accepted] + 1 > c(width)` under
   this host's measured verify curve; fill to the plateau; longest match /
   recency as cold-start tie-break; α = 2, `min_token_prob` ≈ 0.1, cap 32
   (cap ~5 on gqa-6 targets at long context until the SDPA workaround is
   adopted). Linear drafts first; tree speculation only if logged
   branching justifies it. Keep the hybrid with DFlash and the long-draft
   gate.
3. **Logging (from day one):** one row per round — source, match length,
   predicted score, drafted, accepted, verify width, verify/draft ms,
   gate state — plus per-request and per-process totals. Every tuning
   knob in D2 is a function of these columns.
4. **Exactness gate (hard, revised 2026-09-03 -- see Goal invariant):**
   datastore-on must match datastore-off at the same verify width, on
   every corpus prompt at every width; any divergence there, or any
   divergence at a margin above ~2 bf16 ulps, stops the line. Identity to
   plain greedy decoding is reported (both arms hit 58% at width 8 on the
   D0 target, `exactness_gate_v2`), not gated -- that ceiling belongs to
   mlx-dspark's own verify path, not to this project's datastore.
5. **Gate G1-D (pre-registered, M4, the D0 model):** paired decode
   tokens/s, arms = plain / mlx-dspark auto (DFlash+per-request lookup,
   as shipped) / + global datastore, on the frozen corpus in session
   order with the global tree learned online. GO at ≥ 1.2x over the
   as-shipped arm (CI excluding 1.0) with no prompt below 0.95x.
   Also report the cold-history case (first sessions, empty tree) so the
   online-adaptation story is honest.

### Phase D2 — Ladder (only after GO at G1-D)

1. Adaptive speculation length and frequency-scored candidate choice
   (paper's α·p), tuned against this host's verify curve.
2. Cross-project sharing vs isolation; eviction policy; tree size vs gain
   curve (the paper says gain keeps rising — verify it here).
3. Prompt-side ingestion: index tool outputs / file contents seen in
   prompts, not only generations (agents re-emit what they read).
4. Semantic fallback (AdaPLD-style embedding similarity) when lexical
   retrieval misses.
5. Sampling correctness for temperature > 0 (point-mass proposal rule as
   in mlx-dspark).
6. Port to oMLX / mlx-vlm as a drafter kind if the receipts justify the
   larger audience.

Stop condition: two consecutive levers without receipt-backed improvement.

### Phase D3 — Upstream / product (owner-driven)

PR to mlx-dspark (single maintainer, MIT — fork is the fallback); then
oMLX. Owner writes and defends the PRs.

## Owner-trace corpus (authorized data source)

Claude Code transcripts under `~/.claude/projects/` (JSONL) and Codex logs
under `~/.codex/`. Data never leaves the owner's machines; receipts
reference corpora by manifest hash. Workload definitions versioned and
frozen before measured runs; no post-hoc selection. Prior corpus builders
live in the untracked `artifacts/mlx/20260901T193508Z-phase-m0-baseline/`
run root on both machines — reuse the extraction; D0 needs full assistant
outputs and tool results, not only prompts.

## Execution host rule (binding)

Live model runs and benchmarks only on the Apple M4 Pro host (24 GiB),
`ai-server@mac2-llm.local`. The local M1 Pro is for code, unit tests,
lint, and data processing (D0a is fine there). Every receipt records host,
`mlx` / `mlx-dspark` versions, model hash, tokenizer fingerprint, and the
host's measured verify-cost curve.

## Iteration speed rules (binding)

1. Unit tests (seconds) for tree, lookup, persistence, rewind.
2. Offline acceptance replay (seconds–minutes) — the primary loop for
   drafter policy; no model needed.
3. Small-model decode loop (1–2 min, M4) for exactness and mechanics.
4. Subset screen on the real model (~10 min) for ladder levers.
5. Full receipt (background, rare) only to confirm a gate.
Never re-run a control arm whose code path did not change; never wait
idle on a run.

## Marathon execution protocol (binding for long autonomous sessions)

The owner is not available during a marathon. Optimise for wall-clock:
the M4 must never sit idle while the agent thinks, and the agent must
never wait on the M4.

1. **First 15 minutes — start everything slow, in the background, on
   M4:** `git pull`; create the private fork `liqlos/mlx-dspark` and
   install it editable in a venv; download `mlx-community/Qwen3-8B-8bit`,
   `Qwen3.8-27B-4bit` and the matching DFlash 2 / DSpark drafters; run
   mlx-dspark's own first-run cost calibration; kick off the D0c verify
   curve. All via `nohup … &` with logs in the run root. Only then start
   thinking about code.
2. **Overlap by default.** While D0b generation runs on M4 (hours), do
   D0a on the M1 (minutes), then the D1 datastore code and unit tests.
   While a screen runs, write the next lever. Check background jobs
   between tasks, not by waiting.
3. **Python first, C++ only when it pays.** D0a/D0b simulation uses a
   pure-Python suffix/n-gram index (or ArcticInference's `simulator.py`
   if its extension builds on macOS in under 15 minutes — do not fight
   the build). The C++ tree enters at D1 only if the Python index is
   measured too slow for the online loop.
4. **No owner questions.** Every open question gets a recorded default in
   the run journal and work continues: local daily model =
   `Qwen3.8-27B-4bit` (second arm), D0 model = `Qwen3-8B-8bit`, the owner
   runs local models for agentic coding, mlx-dspark is the vehicle. If a
   result would flip under a different answer, say so in the receipt.
5. **Corpus extraction once.** Rebuild the owner-trace corpus with full
   assistant outputs and tool results on the M1 (data stays on the
   owner's machines), version it, sync the sealed workload to M4, freeze
   it before any measured run.
6. **Timeboxes.** D0 verdict within the first session-day. A D1 MVP with
   the exactness gate passing within the next. Anything that blows its
   box gets journaled and replaced by the cheapest design that still
   answers the question.
7. **Deliverable of a marathon:** Gate D0 verdict with receipts; if GO,
   the D1 MVP through the exactness gate and a first G1-D screen; NOW.md
   and the run journal updated so the next session resumes without
   context. Commit and push at every milestone.

## Anti-stall rules (binding)

- Two-strike loop-breaker: a third run of the same design needs a written
  new hypothesis (a code or policy change is one; a bare rerun is not).
- Measure the product, not a proxy: D0a is a proxy and says so; gates use
  the model's own outputs and this host's verify curve.
- Approach freedom within the evidence bar, with recorded rationale.
- Negative results are progress; a positive is not a finish line while
  the ladder has levers.
- Timebox before pivot; no silent scope changes; continuous journaling in
  the run root.
- In-scope commits and pushes need no per-action approval.

## Explicitly parked

- Everything in this section, as of 2026-09-03: Line D itself is closed
  (see `docs/CLOSED_LINES.md`), so these are no longer "levers to reach
  for once D0 passes" -- there is no active line to reach for them from.
- Long-context verify-cost engineering (Metal attention kernel or
  equivalent): real, measured (~34% above a bandwidth-bound floor at 16k),
  general to any drafter/model on this host, and NOT datastore-specific.
  Owner declined to expand this project's scope to pursue it (2026-09-03).
  A future, separately-scoped project may pick this up.
- Prefix / prefill KV caching: out of scope (use oMLX).
- Building on mlx-lm's speculative loop: closed upstream, buggy — no.
- Tree-attention verification in a custom engine: mlx-vlm already has it
  per family; reuse, do not write.
- Writing matmul / attention kernels: reuse `small_m_qmm.py` and
  `sdpa_split.py` from mlx-dspark or mlx-vlm's exact verifiers.
- Block/page-layout storage for the datastore: not needed (in-RAM index,
  append-only log); revisit only if history outgrows RAM (mmap'd suffix
  array as a cold tier).
- Rapid-MLX as a contribution target: no license, zero external PRs
  merged; use as a measurement reference only.
- Rewrites in another language: after receipts show where time goes.
