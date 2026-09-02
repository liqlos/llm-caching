# Current handoff — Line D: datastore speculative decoding on MLX

`NOW.md` is the sole authority for scope, plan, gates, and execution rules.
`docs/CLOSED_LINES.md` records what was tried and closed. Everything else
in the tree (`PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `qualification/`, the
`llamacpp_*` / `exact_prompt_*` / `live_server_*` lab modules) is historical
prior art and binds nothing.

## Scope decision (owner, 2026-09-02)

**This project does not build prefix / prefill KV caching.** oMLX
(`jundot/omlx`) already ships a block-based hot-RAM + cold-SSD prefix cache
for Apple Silicon, built for Claude Code-style traffic; when prefix caching
is needed, run oMLX. The two prefix-cache lines this project built were
removed from the tree (see `docs/CLOSED_LINES.md`).

**What this project builds:** decode-side acceleration for the owner's own
Mac — a persistent datastore of previously generated and seen token
sequences that proposes multi-token drafts during generation, verified
exactly by the target model in one forward pass. Output is bit-identical to
plain decoding; the only effect is more tokens per weight pass. The
engine is MLX / mlx-lm. The owner's traffic is agentic coding, where
generated text repeats heavily (re-emitted code, boilerplate, repeated tool
calls, retries).

## Prior art (verified 2026-09-02) — the niche on MLX is open

- **SuffixDecoding** (Snowflake / CMU, NeurIPS 2025 spotlight,
  arXiv 2411.04975): exactly this idea. Two suffix trees — a global one
  over previous outputs and a per-request one over the current prompt +
  generation — propose adaptive-length drafts (up to hundreds of tokens
  when the match is long); frequency statistics score a speculation tree.
  2.5x mean on SWE-Bench, up to 5.3x on a multi-agent SQL workload, 1.7x
  over prompt-lookup. ~10.75 bytes per cached token; lookups ~12 µs.
  Shipped in vLLM via ArcticInference (Apache-2.0, C++ suffix tree with
  Python bindings — reusable), SGLang PR under review. CUDA/datacenter only.
- **Prompt-lookup decoding** (n-grams from the current prompt): in HF
  transformers, vLLM, llama.cpp. Simplest form of the same thing.
- **REST** (NAACL 2024): retrieval from a static corpus via suffix array.
- **On Apple Silicon / MLX — nothing mature:** `fmsouza/lookup-spec-metal`
  (0 stars, Aug 2026, "Phase A0": offline acceptance on 20 agentic traces
  = 2.01x tokens/pass vs 1.24x on prose, no decode loop yet; same proxy
  caveat as ours — measured on human transcripts, not the model's own
  output); `vukrosic/qwen3-prompt-lookup-robustness` (fixed 2-token
  prompt lookup, +30% on Qwen3-0.6B, verification is >99% of decode time
  so search cost is irrelevant); `MidasMulli/ngram-engine` (n-gram drafter
  disabled in production over "phrase loops" — with exact verification
  accepted tokens are the model's own, so that is likely a non-exact
  verifier; treat as a warning to assert token identity, not as evidence
  against the idea).
- **mlx-lm** has draft-model speculation only (`speculative_generate_step`,
  linear drafts, exact verify + `trim_prompt_cache` rewind). **mlx-vlm**
  has MTP drafting and an exact Metal verifier for Qwen3.5 (hybrid
  candidate later). **oMLX** has MTP / dflash / specprefill, no datastore
  drafter.

Read `fmsouza/lookup-spec-metal` before writing the offline harness; if
its license permits, reuse rather than rewrite its acceptance replay.

## Why the 2026-08-31 NO_GO does not bind this line

That verdict (`NO_GO_CURRENT_REDESIGN`) had two causes, both specific to
the llama.cpp C++ bridge: (1) K>2 verification was not serial-canonical —
a bridge defect, since speculative verification is exact by construction
and mlx-lm's native path already is; (2) no datastore fit a 256 MiB
projection — that design stored something far heavier than a suffix tree
over text tokens (~10 B/token; a year of the owner's sessions is on the
order of 1 GB). Both conditions the old verdict required for reopening are
met by construction on MLX. This is a new hypothesis, not a rerun.

## Goal invariant

Receipt-backed evidence of decode speedup on the owner's representative
traffic with **bit-identical output** (greedy: token identity against
plain decoding, hard kill on any divergence; sampling: distribution-
preserving speculative sampling, later). Evidence bar as before: raw
receipts, pre-registration, paired arms, no post-hoc workload selection.
A clean negative satisfies the goal.

## Ordered plan (gated)

### Phase D0 — Offline opportunity study (no engine code)

The decisive question is acceptance on the owner's traffic. Measure it
before building anything.

1. **D0a — proxy on transcripts (minutes, local M1 is fine).** From the
   owner-trace corpus (`~/.claude/projects/` JSONL; rules below), take
   assistant outputs in chronological order. For each output, simulate
   greedy speculation: at every position, look up the last *n* tokens in
   (a) a per-request tree over the prompt + output so far, (b) a global
   tree over all earlier sessions' outputs, (c) both; propose up to
   MAX_SPEC continuation tokens; count accepted = longest exact match.
   Report mean accepted tokens per step and tokens/pass for MAX_SPEC 8 /
   16 / 32 / adaptive (α·match length), per project and overall. Also
   compute the paper's "structuredness" entropy on 100 outputs. Tokenize
   with the target model's tokenizer (Qwen3-8B). Pre-register thresholds
   before running: proceed at composite ≥ 1.5x tokens/pass, kill below
   1.25x.
2. **D0b — real-model outputs (hours, M4, background).** The transcripts
   are Claude's outputs, not the local model's; acceptance depends on the
   model's own repetition. Generate full responses with
   `mlx-community/Qwen3-8B-4bit` (greedy) on the corpus prompts in
   session order, build the global tree from sessions 1..k, measure on
   k+1, same metrics as D0a. This is the number that counts.

Gate D0 (pre-registered): D0b tokens/pass ≥ 1.5 at MAX_SPEC ≥ 16 →
Phase D1. Below 1.25 → earned NO_GO for this workload; record entropy so
the result transfers. Between: proceed only if the per-project split shows
a clearly profitable subset.

### Phase D1 — Decode loop MVP

1. **Drafter + datastore.** Suffix tree (per-request + global) with
   frequency counts; persistent global tree on disk (this is where
   persistence lives now — a small text-token structure, not KV);
   per-project namespaces keyed by the tokenizer/model fingerprint.
   Reuse ArcticInference's Apache-2.0 suffix tree if it binds cleanly on
   macOS; otherwise a minimal Python/C implementation.
2. **Generation loop.** Prefer a standalone package (`src/specd/`) that
   drives mlx-lm's public model / cache / `trim_prompt_cache` API with a
   drafter callable in place of the draft model, mirroring
   `speculative_generate_step`. Avoid forking mlx-lm unless a hook is
   truly missing; if it is, the hook itself is the upstream PR.
3. **Exactness gate (hard).** Token identity vs plain greedy on every
   corpus prompt at every MAX_SPEC; any divergence stops the line until
   fixed.
4. **Performance gate G1-D (pre-register, M4):** paired decode
   tokens/s, datastore-speculation vs plain mlx-lm at matched context,
   on the frozen corpus in session order (global tree learned online from
   earlier sessions, like real use). GO at ≥ 1.3x mean with CI excluding
   1.0 and no prompt slower than 0.95x; report bucketed by match length.
   Memory-bandwidth note: on Apple Silicon decode is bandwidth-bound, so
   verifying K tokens costs about one token's weight pass up to some K —
   measure that K first (verify cost vs K, one 10-minute microbench).

### Phase D2 — Ladder (only after GO at G1-D)

1. Adaptive MAX_SPEC from match length (the paper's α·p), scored by
   frequency.
2. Tree speculation (multiple candidates per step) — needs a tree
   attention mask in the verify pass; measure verify-cost vs tree size.
3. Hybrid with MTP (mlx-vlm, Qwen3.5) when the datastore has no match.
4. Datastore management: eviction, per-project trees, cross-project
   sharing, warm-start from disk.
5. Serving: an OpenAI/Anthropic-compatible endpoint so Claude Code uses
   it directly — either a thin server around the loop or a drafter
   contribution into oMLX's speculation stack.
6. Sampling correctness (speculative sampling) for temperature > 0.

Stop condition: two consecutive levers without receipt-backed improvement.

### Phase D3 — Upstream / product (owner-driven)

Drafter interface PR to mlx-lm, or suffix drafter into oMLX, whichever
the receipts justify. Owner writes and defends the PRs.

## Owner-trace corpus (authorized data source)

The owner authorized their local agent-session histories as workload data:
Claude Code transcripts under `~/.claude/projects/` (JSONL) and Codex logs
under `~/.codex/`. Data never leaves the owner's machines; receipts
reference corpora by manifest hash, never content. Workload definitions
are versioned and frozen before measured runs; selecting traces after
seeing results is prohibited. Prior corpus builders live in the untracked
`artifacts/mlx/20260901T193508Z-phase-m0-baseline/` run root on both
machines (v1: 15 `oculusprojects` sessions, 286 turns) — reuse the
extraction, redefine the workload for D0 (full assistant outputs matter
now, not just prompts).

## Execution host rule (binding)

Live model runs and benchmarks only on the Apple M4 Pro host (24 GiB),
`ai-server@mac2-llm.local`. The local M1 Pro is for code, unit tests,
lint, and data processing (D0a is fine there). Every receipt records host,
`mlx` / `mlx-lm` versions, model hash, and tokenizer fingerprint.

## Iteration speed rules (binding)

Development lives in minutes; hours-long runs are background receipts.
1. Unit tests (seconds) for the tree, lookup, and rewind logic.
2. Offline replay (seconds–minutes): acceptance simulation on the corpus
   — the primary loop for drafter policy work; no model needed.
3. Small-model decode loop (1–2 min, M4): `Qwen2.5-0.5B-Instruct-4bit`
   for exactness and mechanics, real model only for numbers.
4. Subset screen on the real model (~10 min) for ladder levers.
5. Full receipt (background, rare) only to confirm a gate.
Never re-run a control arm whose code path did not change; never wait
idle on a run.

## Anti-stall rules (binding)

- Two-strike loop-breaker: a third run of the same design needs a written
  new hypothesis; a code change to the mechanism or a policy change is
  one, a bare rerun is not.
- Measure the product, not a proxy: D0a is a proxy and says so; gates use
  the model's own outputs.
- Approach freedom within the evidence bar, with recorded rationale.
- Negative results are progress; a positive is not a finish line while
  the ladder has levers.
- Timebox before pivot; no silent scope changes; continuous journaling
  in the run root.
- In-scope commits and pushes need no per-action approval.

## Explicitly parked

- Prefix / prefill KV caching: out of scope (use oMLX).
- Rewrites in another language: after receipts show where time goes.
