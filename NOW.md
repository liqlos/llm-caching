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

## Strategic position (research pass, 2026-09-02)

The field confirms the idea and has already built most of the stack. What
exists must be reused, not rebuilt; the one open gap is where this project
lives.

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

**Apple Silicon economics that shape every decision:**

- Verification cost is *not* free: quantized Metal verify cost grows
  ~linearly with draft width below ~8 (per-position matvec), plateaus at
  width 16–32 at ~2.5x the cost of one step (mlx-dspark, M-series);
  EAGLE-3 on MLX measured ~1 ms per extra verified token
  (mlx-lm discussion #890). So drafts of "hundreds of tokens" do not pay
  here; the useful window is ~8–32 tokens per round, and speedup comes
  from *acceptance* inside that window, where a datastore is strongest on
  repetitive text.
- Small fast targets have no headroom: 8B-4bit at ~114 tok/s gave
  EAGLE-3 1.05x; mlx-dspark reports minimal/negative gains under ~4B and
  ratios peaking at 8-bit, not 4-bit (MLX matmul cost cliff). **Target
  the model the owner will actually run for agentic coding; on a 24 GiB
  M4 Pro the candidates are 8B-8bit or 27B-4bit, not 8B-4bit.**
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
   metrics. This is the number that counts.
3. **D0c — verify-cost curve (10 min, M4).** Measure the D0 model's
   verify cost vs width 1..32 on this host (or read mlx-dspark's cached
   curve) so tokens/pass converts to wall-clock honestly.

**Gate D0 (pre-registered):** global (c) must beat per-request (a) by
≥ 20% in projected wall-clock at the best width on D0b, and (c) must
project ≥ 1.3x over plain decoding. Below either → earned NO_GO for this
workload; record entropy so the result transfers. Between → proceed only
if a per-project split shows a clearly profitable subset.

### Phase D1 — Global datastore drafter inside the existing stack

Build **into `mlx-dspark`** (fork `liqlos/mlx-dspark`, MIT), not from
scratch and not on mlx-lm:

1. **Datastore:** global suffix tree over previous prompts + outputs with
   frequency counts; persisted on disk (write-behind at idle, load on
   start, per-project / per-tokenizer namespaces, byte quota + LRU by
   sequence). Reuse ArcticInference's C++ tree via nanobind if it builds
   cleanly on macOS; otherwise a minimal implementation. Persistence of
   *the datastore* is where this project's disk expertise now applies.
2. **Drafter:** extend `lookup.py`'s proposal step — consult the global
   tree when the per-request index has no (or a shorter) match; keep its
   hybrid with DFlash and its long-draft gate. Score candidates by
   frequency (adaptive length α·match), cap at the measured verify
   plateau.
3. **Exactness gate (hard):** token identity vs plain greedy on every
   corpus prompt at every width; any divergence stops the line.
4. **Gate G1-D (pre-registered, M4, the D0 model):** paired decode
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
- Rewrites in another language: after receipts show where time goes.
