# Where is the fundamental profit in Line D? An adversarial audit (2026-09-02)

Research memo. Input to the Gate D0 verdict in `NOW.md`; binds nothing by
itself. Every number below is either cited or computed by the offline replay
described in section 1 (scripts and receipts in
`artifacts/mlx/20260902T152845Z-phase-d0-marathon/d0b/profit_replay/`,
local-only like the rest of the run root).

## 0. Verdict

- **There is no fundamental lever left inside the datastore that rescues the
  global (cross-session) store on this workload.** With a *perfect* candidate
  selector and a *perfect* draft length, over the same trigram-keyed
  candidates, the global store beats the per-request index by **+10%** at the
  honest 8k-context verify cost (+14% under the placeholder curve; +19% in the
  last quarter of turns). Every realistic policy tested — SuffixDecoding's
  frequency-scored paths, adaptive length, trees, count-probability and
  calibrated Dinkelbach length rules — lands at **0–3%**. Gate D0 asks for 20%.
- **What SuffixDecoding adds over recency lookup is real but per-request:**
  majority-path drafting with best-match-length scoring lifts *both* arms by
  ~7% at 8k context. Tree drafting is neutral-to-negative on M-series
  (verify cost is by row count, not tree shape). Frequency statistics carry
  almost no information here: 62% of drafting rounds are singleton matches.
- **The largest real lever is not in the index at all:** the verify curve's
  context-depth term. At 8k tokens `c(7 drafted) = 1.97` vs `1.35` at 512
  (calibration.json, `verify_depth.slope`), which alone costs ~21% of
  tokens/pass. That is attention-kernel work (`sdpa_split.py`, oMLX's fused
  verify attention), already parked as "reuse" in `NOW.md`.
- Honest headline for the line: **a lookup drafter (any of them) is worth ~2.0x
  on this workload at 8k context and it is already shipped per-request in
  mlx-dspark; a persistent cross-session store adds ~2%.**

## 1. Method (what was replayed, and the reproduction check)

Data: the 252 real `Qwen3-8B-8bit` greedy outputs dumped by D0b v2 (200
tokens each, 50,400 tokens) over the frozen corpus in session order; the index
stream is the corpus text up to each turn (1.18M tokens: user/tool text,
transcript assistant text, and the real outputs of dumped turns). Acceptance
is exact greedy simulation (drafted token == recorded token). Pure Python,
M1, ~2 min per pass; no GPU.

Index used for every policy: trigram-keyed positions (≤1024 most recent per
key, bigram fallback), each occurrence extended backwards up to 64 tokens for
its match length `m`; occurrences with match ≥ `m` form a suffix-tree node
with count `n`. Majority-continuation paths with `prob = count/n` per step
are exactly `SuffixTree::_speculate_path`; the tree variant is
`_speculate_tree`. Unlike `replay_scoring.py`, the current turn's own tokens
are indexed online, as a real per-request tree does. Arms: **L** = current
session (prompt + output so far), **X** = all earlier sessions, **G** = L ∪ X,
**L8k** = last 8k tokens of the session (D0b's real prompt window).

Cost curves (tokens/pass = tokens ÷ Σ round cost in plain-step units):
`placeholder` (journal), `m4_512` = calibration.json rows 1..8 at 512
context (`c(4)=1.36, c(7)=1.35, c(8)=1.40`), `m4_8k` = same plus the measured
per-row depth slope at 8192 tokens (`c(4)=1.67, c(7)=1.97, c(8)=2.11`);
rows above 8 are extrapolated and only matter for the cap-12/32 rows.

Reproduction: re-implementing `replay_scoring.py`'s exact semantics on the
same data gives a=2.068, b=c=2.094 (journal: 2.061 / 2.092 on 229 turns).
Under `m4_8k` the same policy gives a=1.754, b=c=1.723 — **c/a = 0.98**: at
honest cost the journal's global arm is already slightly negative.

## 2. The evidence table (tokens/pass, 252 turns)

| policy | L (per-request) | X (cross-session only) | G (both) | G/L |
|---|---|---|---|---|
| recency, width 8 (mlx-dspark style), placeholder / 512 / 8k | 2.35 / 2.51 / 1.89 | – | 2.34 / 2.54 / 1.80 | 1.00 / 1.01 / 0.96 |
| SuffixDecoding path α=1, p≥0.1, cap 8 | 2.40 / 2.48 / 2.01 | 1.44 / 1.46 / 1.29 | 2.46 / 2.55 / 2.01 | 1.03 / 1.03 / 1.00 |
| SD path, best of 18 (α,p,cap) grid | 2.46 / 2.54 / 2.03 | 1.44 / 1.46 / 1.31 | 2.47 / 2.59 / 2.04 | 1.00 / 1.02 / 1.00 |
| SD both-trees (ArcticInference `draft1.score >= draft2.score`) | – | – | 2.48 / 2.57 / 2.04 | 1.03 / 1.04 / 1.00 |
| SD tree, 8 nodes / 16 nodes (G) | 2.40 / 2.56 / 1.93 | – | 2.41 / 2.61 / 1.87; 16: 2.31 / 1.94 / 1.35 | ≈1.00 |
| Dinkelbach on count-probabilities, cap 12 | 2.41 / 2.57 / 1.85 | – | 2.42 / 2.61 / 1.80 | 1.00 |
| Dinkelbach on calibrated P(acc≥k \| m,n) (in-sample), cap 12 | 2.43 / 2.58 / 1.95 | – | 2.50 / 2.65 / 1.99 | 1.03 |
| oracle length (majority path, exact accepted length, cap 32) | 3.13 / 3.15 / 2.62 | 1.63 / 1.63 / 1.54 | 3.32 / 3.34 / 2.73 | 1.06 / 1.06 / 1.05 |
| **oracle occurrence + oracle length** (cap 32) | 3.41 / 3.44 / 2.80 | 1.75 / 1.74 / 1.63 | 3.89 / 3.93 / 3.07 | **1.14 / 1.14 / 1.10** |
| plain decoding | 1.00 | 1.00 | 1.00 | – |

Copyable-token coverage (oracle occurrence; share of output tokens inside a
run of ≥k accepted): L 78/65/54/40% for k=1/4/8/16; X 48/32/21/12%;
G 83/68/56/41%; L8k 75/62/51/38%. The global store raises the ≥8-token
copyable share by 2.3 points. That is the whole cross-session opportunity.

Where accepted tokens come from (SD path, G, cap 8): 23% user/tool text the
agent read, 35% transcript assistant text, 42% the model's own earlier
outputs. Cross-session-only hits: 45% the model's own previous outputs, 27%
tool/user text, 28% transcripts; 73% same-project. The both-trees policy
routes 13% of accepted tokens via X, yet nets +3% (placeholder) / +0.2% (8k)
over L: L usually had a comparable draft.

History growth (quartiles of turn order): SD-path X arm 1.36 → 1.60 → 1.43 →
1.39 (no trend); SD-path G/L ratio 1.02, 1.02, 1.02, 1.04 (placeholder), 1.00,
1.00, 0.99, 1.00 (8k). Oracle G/L ratio 1.09 → 1.14 → 1.15 → 1.19: the
*ceiling* does grow with history; nothing realistic tracks it.

## 3. Ranked levers, with ceilings and the evidence behind each

1. **Verify cost at depth (kernel, not index).** Same policy: 2.55 at 512
   context vs 2.01 at 8k (−21%); the per-row depth slope is 4.8 µs/token/row
   at 8 rows, i.e. +37 ms on a 46 ms verify at 8k, because the vector SDPA
   path re-reads the KV stream per query row (`APPLE_SILICON_VERIFY_ECONOMICS`
   §1.2). Ceiling ≈ +25% wall-clock at 8k, more at 32k. Evidence: measured
   curve + replay. Owner: mlx-dspark `sdpa_split.py`, mlx-vlm exact verifiers,
   oMLX fused verify attention. GPU work, M4 only; not this line's thesis.
2. **Candidate selection (which occurrence to follow).** Oracle occurrence vs
   oracle length on the majority path: +12% (8k) / +17% (placeholder); the
   global-specific part of it is the +10–19% above. Realistic selectors tried
   (recency; majority path + best score; both-trees arbitration) capture
   ≤3% of it. Unknown whether a learned selector (features: m, n, source
   kind, same session/project, recency rank) can approach the oracle — the
   one experiment that could still move the global store (E2 below).
3. **Draft-length policy.** Oracle length: +35% over SD α=1 cap 8 at 8k
   (2.73 vs 2.01). But acceptance-length distributions conditional on (m, n)
   decay slowly and smoothly (`P(acc≥k)` for m≥32, n=1: .96 .92 .88 .85 .81
   .78 .76 .73 for k=1..8), so nothing in the index predicts *where* a draft
   diverges; the calibrated Dinkelbach rule gets +0–4%. Realistic ceiling
   ≈ +4%. Fixed cap 8 with α=1 is within 1% of the best grid point at 8k.
4. **Majority-path drafting with match-length scoring (SuffixDecoding's real
   contribution).** +7% over recency at 8k for both arms (1.89 → 2.01 L;
   1.80 → 2.01 G). Cheap; already the design in `datastore.py`'s plan. Not a
   global-store lever.
5. **Global cross-session store.** Realistic +0–3%; oracle +10% (+19% late).
   Gains concentrate in the model's own boilerplate and same-project content.
6. **Session-persistent store beyond the serving window.** L8k vs L: −1%
   realistic (1.99 vs 2.01), −6% oracle (2.62 vs 2.80). Only matters if the
   server truncates the prompt; Claude Code sends the whole conversation
   until compaction, so this is a ~1% lever.
7. **Frequency statistics per node.** ~0. 62% of drafting rounds and 52% of
   accepted tokens are singleton matches (count-ratio probability 1.0 by
   construction); at fixed m, n=4+ is not systematically better than n=1
   (m 8–15: 5.3 vs 4.4 accepted; m 3: 0.66 vs 0.85). Calibration of the
   count score is poor (score bucket 4.0 → 1.85 accepted; bucket 8.0 → 6.1).
8. **Tree drafting.** 0 to −3% at 8 nodes, −33% at 16 nodes under `m4_8k`.
   Matches the paper's own SWE-Bench numbers (linear 7.82 vs tree 7.55 mean
   accepted; speedup 2.45x vs 2.43x). Trees pay only where the corpus branches
   at high probability (AgenticSQL Extract booleans), not in code sessions.
9. **Prompt-side ingestion (what the agent reads).** Already exploited: the
   per-request index holds tool outputs and file contents, and 23% of accepted
   tokens come from them. Cross-session, tool/user text is 27% of X's hits.
   Not a new source; it is why L subsumes X (agents read before they write).
10. **Hybrid with the model drafter (DFlash).** On this host and model the
    bar is low: calibration.json prices a DFlash round at cap 7 as verify(8
    rows) 46 ms + drafter 10 ms + 66 ms fixed overhead = 122 ms at 512 context
    (3.6 plain steps; 3.8 at 8k) for 3.05 accepted → ≤1.14x. A lookup round
    at ~2.0x dominates it; the τ-rule hybrid (paper §3) adds at most the 17%
    of tokens from no-match rounds × (1.07 − 1) ≈ +1%. On `Qwen3.8-27B-4bit`
    (DFlash 2.4x) the bar flips and the datastore is the add-on, worth at most
    the 61% of tokens in rounds with ≥4 accepted minus DFlash's own coverage
    of them — needs the 27B arm.
11. **Token Recycling / previous-round top-k logits.** Not testable offline
    (no logits dumped); literature: below PLD on SWE-Bench (3.05 vs 3.17
    accepted; 1.36x vs 1.50x). Filler at best.
12. **Semantic fallback (AdaPLD).** Targets the 17% of tokens with no lexical
    match at all (G, k≥1) plus mis-selected rounds; unverifiable offline.

## 4. The optimal draft-length policy (question 2)

Let `q_k = P(accept ≥ k | context)` for the candidate path and `c(w)` the
verify cost of `w` drafted tokens in plain-step units (`c(0)=1`). A round of
width `w` yields `1 + Σ_{k≤w} q_k` tokens for `c(w)`.

- Long-run throughput is a ratio of expectations (renewal reward), so the
  optimal stationary policy is **Dinkelbach's**: pick `(candidate, w)`
  maximising `1 + Σ_{k≤w} q_k − λ·c(w)`, with `λ` = the achieved throughput
  (fixed point; two iterations suffice). Per-round ratio maximisation
  `(1+Σq)/c(w)` is *not* optimal and lost 1–2% against it here.
- Marginal form: extend the draft to `w+1` iff `q_{w+1} > λ·(c(w+1) − c(w))`.
  On a plateau (`Δc = 0`) fill it; on the linear 8k curve
  (`Δc ≈ 0.14/row`, `λ ≈ 2`) a token must carry `q > 0.28`; on the 512 curve
  the free window to row 4 makes width 3–4 nearly always right.
- Statistics the index must carry: not counts, but a calibrated
  `q_k` table keyed by what actually predicts acceptance — match length `m`
  (dominant: 0.85 → 6.5 accepted from m=3 to m≥32), source kind, session
  membership — learned online from `RoundLog` rows (the `datastore.py` design
  already logs them). Counts add nothing once `m` is known; singleton nodes
  make `count/n` degenerate.
- Ceiling check: even in-sample calibration reaches only +0–4%, because the
  accept-length distribution at a given `m` is broad. "Draft the longest match
  you have, capped at the plateau" is within 1–4% of the best achievable
  policy on this data; that policy was already good.

## 5. Index structures (question 3)

| structure | gives | missing for us |
|---|---|---|
| suffix tree with counts (ArcticInference) | exact longest match, counts, tree drafts, 4 µs update / 12 µs lookup | counts uninformative here; no selection features; no serializer |
| suffix automaton (SAM-Decoding) | exact longest match O(1)/step, ≤2n states | no counts, no delete; draft = one position, exactly our recency policy |
| suffix array / FM-index (REST, Infini-gram) | counts free (interval width), 7 B/token, mmap persistence | no online update; same information as above |
| trigram hash + backward extension (ours / mlx-dspark) | longest match up to 64, all occurrences, µs | equivalent to the tree for selection purposes at 10^6–10^7 tokens |
| PPM / Kneser-Ney backoff over orders | calibrated `P(next | ctx)` mixing orders, discounting singletons | the right theory for calibration, but what we need is `P(accept ≥ k | m, kind)`, a 30-cell table, not a language model |
| count-min sketch / LZ dictionaries | memory bounds | cannot enumerate continuations; strictly worse |

Conclusion: lookup latency and structure are not the bottleneck (every option
is µs); the missing information — which occurrence, and where it diverges —
is stored in none of them. A learned selector is the only addition with a
non-zero ceiling.

## 6. Sources of drafts not used (question 4)

Already used: tool outputs and file contents (23% of accepted tokens), the
model's own earlier outputs (42%), Claude transcripts drafting Qwen output
(35% — cross-model reuse is not a new lever). Not used and not testable
offline: previous-round top-k logits (literature: ≤ PLD on code), template
completion (the branching case where trees pay; our order-2 entropy is 1.62
bits/token vs AgenticSQL Extract's 0.09), semantic retrieval.

## 7. The honest counterfactual (question 5)

- SuffixDecoding (NeurIPS 2025): the global-vs-per-request ablation (Fig. 7)
  exists **only for AgenticSQL**, a 7-stage pipeline of thousands of
  near-identical structured requests (entropy 0.09–1.5 bits); global wins
  everywhere except the one step that re-uses its context (Combine). **No such
  ablation is reported for SWE-Bench**; there the paper's 1.64x edge over PLD
  (2.45x vs 1.50x; 7.8 vs 3.2 accepted) comes from adaptive length and a
  per-request tree over the full prompt, with linear drafts beating trees.
- SAM-Decoding (ACL 2025), Table 2, Spec-Bench, Vicuna-7B: full 1.84x; without
  the static corpus automaton 1.64x; without the dynamic (per-request) one
  1.33x; PLD 1.56x. The static corpus adds +12% over per-request on chat; on
  HumanEval adding it to EAGLE-2 "changed slightly" because "code generation
  is less likely to copy the generated text".
- llama.cpp's static+dynamic n-gram cache: ≤1.11x, never adopted.
- Our data: +0–3% realistic, +10% oracle. The literature and the replay agree:
  **the per-request tree captures most of the value on coding/agentic
  traffic**; a global store is decisive only for pipeline workloads whose
  requests repeat a template thousands of times.

## 8. Experiments that would confirm or kill the top levers (offline, data in hand)

Harness: `profit_replay/analyze.py` (`SECOND_PASS=0|1|2` selects the policy
set), inputs `corpus_tokens.jsonl` (tokenised frozen corpus, same flatten
order as `replay_scoring.py`) and `turns_snapshot_252.jsonl` (replace with the
full `d0b_results_qwen3-8b-8bit.json.turns.jsonl` when the 600 turns exist).
Each pass ≈ 2 min on the M1. Pre-register thresholds before running.

**E1 — Learned occurrence selector (lever 2; the only remaining path to the
global store).** For every position and every candidate occurrence record
features `(m, n, kind, same_session, same_project, recency_rank, source turn
role, position-in-turn)` and the label `LCP(continuation, truth)`. Fit a
small model (logistic or gradient-boosted, ≤ 20 features) on turns 1–126,
evaluate on 127–252 by replaying `select = argmax predicted LCP`, length =
Dinkelbach on the predicted `q_k`, under `m4_8k`. Compare G vs L. Confirm the
global store only if **G/L ≥ 1.20 out-of-sample**; the oracle ceiling is
1.10–1.19, so realistically this kills it. Report also L alone vs SD path:
anything ≥ +10% is a per-request win worth upstreaming to mlx-dspark.

**E2 — Global-store growth at 600 turns and per-project isolation.** Re-run
passes 0–2 on the full dump; report oracle and SD-path G/L per quartile, and
with X restricted to same-project. Kill if the SD-path ratio at 8k stays
< 1.05 in the last quartile.

**E3 — Depth-term removal (lever 1; needs the M4 GPU, after D0b ends).**
Measure `verify_depth.slope` with `sdpa_split.py` enabled and with the
mlx-vlm exact verifier at 8k/16k/32k, dependent chains, rotated weights;
re-price the replay with the new curve. Expect +20–25% at 8k. This is the
number to quote as the line's ceiling if the datastore is dropped and only
the per-request lookup is kept.

Reconcile before any wall-clock claim: D0b's loop measured ~4.4 tok/s at 8k,
while calibration.json's row-1 verify is 42 ms at 8k (24 tok/s) and 34 ms at
512 (29 tok/s: the ~26 tok/s roofline was not optimistic). One is wrong.

## 9. If all of these fail, the line is dead because

the information a cross-session store adds is bounded by the tokens the
model writes that it has not read in the current session but has seen in an
earlier one; on this owner's traffic that is 2–3 points of copyable-token
coverage (56% → 58% at ≥8-token runs), an oracle-bounded +10% in tokens/pass,
and 0–3% with every scoring, length, and tree policy in the literature —
because agents read before they write, so the per-request index already holds
what the global one would return. No index structure changes that bound; only
a selector that approaches the oracle could, and nothing tried approaches it.
The remaining profit on this host is a per-request lookup drafter (shipped)
plus attention-kernel work on the verify depth term, neither of which is this
line's thesis.

## Sources

- SuffixDecoding, NeurIPS 2025 (v3): https://arxiv.org/abs/2411.04975 — §3
  scoring and `MAX_SPEC = αp`; §4.2 Fig. 4 and App. A.1 SWE-Bench tables
  (suffix linear 7.82 / tree 7.55 / PLD 3.17 accepted; speedups 2.45 / 2.43 /
  1.50); §4.4 Fig. 7 (AgenticSQL only); App. A.2.3 entropy; App. B Table 2
  (4 µs update, 12 µs lookup, 10.75 B/token); App. B.1 hybrid τ tables.
- ArcticInference: https://github.com/snowflakedb/ArcticInference —
  `csrc/suffix_decoding/suffix_tree.{h,cc}` (`speculate`, `_speculate_path`,
  `_speculate_tree`), `arctic_inference/suffix_decoding/cache.py`
  (`draft1 if draft1.score >= draft2.score else draft2`), `simulator.py`,
  `docs/suffix-decoding.rst`.
- SAM-Decoding, ACL 2025: https://arxiv.org/abs/2411.10666 (Table 2 ablation).
  REST: https://arxiv.org/abs/2311.08252. Token Recycling:
  https://arxiv.org/abs/2408.08696. PLD: https://github.com/apoorvumang/prompt-lookup-decoding.
  AdaPLD: https://arxiv.org/abs/2606.05742. Infini-gram: https://arxiv.org/abs/2401.17377.
- mlx-dspark (fork `liqlos/mlx-dspark`): `src/mlx_dspark/lookup.py`
  (`NGramIndex`, `LongDraftGate`), `calibrate.py` (`measure_verify_depth_slope`,
  `measure_dspark_round_overhead`, `CapController.rate`);
  M4 `~/.cache/mlx_dspark/calibration.json` (Qwen3-8B-8bit rows 1..8:
  34.3 … 46.4 ms; depth slope 1.0 / 3.1 / 4.8 µs per token per row at rows
  1 / 5 / 8; dspark overhead 66 ms).
- Run root `artifacts/mlx/20260902T152845Z-phase-d0-marathon/`: `JOURNAL.md`,
  `d0b/replay_scoring.py`, `d0b/profit_replay/` (this memo's harness and
  receipts). Context: `NOW.md`, `docs/research/DATASTORE_DESIGN_STUDY.md`,
  `docs/research/APPLE_SILICON_VERIFY_ECONOMICS.md`.
