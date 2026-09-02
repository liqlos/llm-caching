# Current handoff — Line D: persistent cross-session datastore speculation on Apple Silicon

`NOW.md` is the sole authority for scope, plan, gates, and execution rules.
`docs/CLOSED_LINES.md` records what was tried and closed. Everything else
in the tree (`PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `qualification/`, the
`llamacpp_*` / `exact_prompt_*` / `live_server_*` lab modules) is historical
prior art and binds nothing.

## Scope decision (owner, 2026-09-02)

**Not prefix / prefill KV caching** — oMLX ships it; see
`docs/CLOSED_LINES.md`.

**What this project builds:** decode-side acceleration by speculating from
a **persistent, cross-session datastore of previously generated and seen
token sequences** ("computed token sequences saved to SSD"), verified
exactly by the target model. Output is bit-identical to plain decoding.
Engine: MLX on the owner's Macs. Traffic: agentic coding.

## Research memos (read before designing or measuring anything)

- `docs/research/PRIOR_ART_DATASTORE_SPECULATION.md` — who has what;
  verdict: the niche is open.
- `docs/research/DATASTORE_DESIGN_STUDY.md` — index structures,
  persistence, draft policy, offline evaluation, logging schema.
- `docs/research/APPLE_SILICON_VERIFY_ECONOMICS.md` — why verify cost is
  what it is on M-series, per-target speedup ceilings, kernels to reuse.
- `docs/research/REPO_AUDIT_2026-09-02.md` — the post-pivot cleanup record.

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
  4.5x on copy-heavy code editing (Gemma-12B, 75 tok/s) with bit-identical
  output. It is **per-request only**: the index dies with the request.
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
drafter** on the owner's representative traffic, with bit-identical output
(greedy: token identity; sampling: distribution-preserving). Evidence bar
as before: raw receipts, pre-registration, paired arms, no post-hoc
workload selection. A clean negative satisfies the goal.

Exactness decision: gates are run on the **bit-identical** verify path
(singleton-numerics exact verifiers). The non-identical small-M kernel is
a D2 lever measured separately with a token-identity rate; it may be
offered as an opt-in, never as the default the claims rest on.

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
4. **Exactness gate (hard):** token identity vs plain greedy on every
   corpus prompt at every width; any divergence stops the line.
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
