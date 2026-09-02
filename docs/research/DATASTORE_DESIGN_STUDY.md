# Datastore design study — persistent cross-session speculation on Apple Silicon

Research pass, 2026-09-02. Input to Phase D1 of `NOW.md`; binds nothing by
itself. Every number below is either cited or marked as an estimate to be
measured in D0a (`estimate_memory()` and the offline simulator exist).

## 0. What the hardware window demands of the index

- Round cost on M-series (mlx-dspark, 8-bit, M4 Pro): verify cost grows
  ~linearly with draft width below ~8 and plateaus at width 16–32 at ~2.5x
  one plain step. Let `c(w)` be that measured curve with `c(0) = 1`.
- A round that drafts `w` tokens and gets `a` accepted yields `a + 1` tokens
  for cost `c(w)`. It pays iff `a + 1 > c(w)`, i.e. on the plateau the draft
  must expect **> ~1.5 accepted tokens**; at width 4 it needs only ~0.3.
  Once a round is on the plateau, extra draft tokens are free: fill it.
- Consequence for the index: it must return, per round, in well under
  100 µs (a decode step is ~10–30 ms; 1% budget), (a) the longest matching
  suffix, (b) a continuation of 8–32 tokens, and (c) an **expected-accepted
  score** so the policy can decide draft/no-draft and which source to use.
  Frequency statistics are what make (c) possible; recency-only indexes
  (prompt-lookup) cannot produce it without online calibration.

## 1. Candidate index structures

Notation: `n` tokens in history, `D` depth cap, `V` vocabulary (~150k).

### 1.1 Suffix tree with counts (SuffixDecoding / ArcticInference)

`csrc/suffix_decoding/suffix_tree.{h,cc}` (Apache-2.0, ~900 lines, no CUDA,
nanobind bindings): depth-capped, path-compressed suffix tree over many
sequences, each node holding `count`, `(ref_seq, ref_idx, length)` into the
stored token vectors, an `Int32Map` of children and of per-sequence
endpoints, and children kept in a count-ordered doubly linked list with
same-count groups so `_increment_count` is O(1) amortized.

- Update: `append(seq_id, token)` walks at most `D` active nodes (sliding
  window), so O(D) worst case per token; paper Table 2 measured ~4 µs/token
  update and 11–12 µs per speculated token lookup.
- Lookup (`speculate`): for `match_len = 1..min(len(ctx), D)` matches the
  suffix, draft cap `min(max_spec, α·match_len + offset)`, builds a path
  (`head_child` chain, prob = product of `count_child / count_node`) or a
  tree (max-heap by cumulative prob), returns the draft with the highest
  `score = Σ D(N)`, which *is* the expected number of accepted tokens under
  the empirical distribution. `min_token_prob` (default 0.1) prunes.
- Scoring by frequency: native. Recency: not native; `ref_seq` of the match
  identifies which stored sequence supplied the path, so recency can be
  layered on (see §3).
- Tree speculation: native (`use_tree_spec`), with parent indices.
- Eviction: `remove(seq_id)` walks every suffix of the sequence, so O(len·D);
  the Python cache does FIFO by request count (`max_cached_requests`,
  default 100000). The paper says LRU is "straightforward"; nothing is
  implemented for byte quotas.
- Memory: the paper reports 6.15 GB for 572M tokens = **10.75 B/token** on
  its agentic workloads. That figure is workload-dependent: the `Node`
  struct is ~128 B, plus `Int32Map` slots (children 16 B, endpoints 8 B,
  min capacity 2, 25–75% load) and a shared `Group` (24 B), so ~170–250 B
  per node; 10.75 B/token means roughly one node per 15–25 tokens, which is
  what heavily repeated outputs give. On high-entropy text a depth-64
  truncated suffix tree approaches ~2 nodes per token, i.e. several hundred
  B/token (estimate). The owner's corpus will sit between; D0a must report
  `estimate_memory()/tokens` per project, and `max_depth` (32 vs 64) is the
  main lever.
- Persistence: none upstream. Sequences are stored verbatim in `_seqs`, so an
  append-only token log is a lossless external representation from which the
  tree is rebuilt by replay; a snapshot serializer would be new code.

### 1.2 Suffix automaton (SAM-Decoding)

Static SAM over a corpus plus dynamic SAM over the current text; O(1)
amortized per step for update and longest-suffix retrieval (exact longest
match, not capped by an n-gram size). Draft = tokens following `min_endpos`
of the matched state (a single occurrence, position-based, not frequency
based); static draft used only if `l_static > l_dynamic + l_bias`, and the
model drafter (EAGLE-2) takes over below `l_threshold` (both 5); draft size
40 (16 for code). Memory: ≤ 2n states, ≤ 3n transitions; with hashed
transitions ~50–100 B/token (estimate, paper gives none). Weaknesses for this
project: no counts, so no expected-acceptance score and no principled tree
speculation; deletion of old sequences from a SAM is not supported (rebuild).

### 1.3 Suffix array / FM-index over the whole history (REST, Infini-gram)

- Suffix array: 5 B pointer + 2 B token = **7 B/token** (Infini-gram, ~3.5x
  the raw corpus); mmap-able, served from disk at 1.4T tokens with
  <20 ms counts and 39–180 ms next-token distributions (that latency is
  disk-bound at that scale; for 10^8 tokens in page cache it is µs).
  Build: libsais-class algorithms do ~10^8 tokens in seconds to tens of
  seconds on one core; no online update (rebuild or LSM-style segments).
- REST retrieves the longest suffix (n_max = 16) from an SA-backed datastore,
  builds a trie from the retrieved continuations weighted by frequency and
  keeps the top-64 nodes; retrieval plus trie build < 1 ms; datastore growth
  0.9 → 27 GB raised speedup 1.83x → 2.36x, tokens/pass 1.96 → 2.65.
- Counts are free (interval width `hi − lo`); the next-token histogram costs
  a scan over the interval (cap by sampling ~256 positions, as REST does).
- FM-index (Infini-gram mini) shrinks the index to 44% of the corpus but
  makes every step a rank query and is strictly harder to update. Not
  needed below ~10^10 tokens.
- Tree speculation: via the retrieved-continuation trie (REST). Recency:
  positions are ordered, so "latest occurrence" is one comparison.
- Persistence-friendliness: best of all options (two flat files); eviction is
  segment-level (drop old segments and rebuild).

### 1.4 n-gram hash tables

- llama.cpp `common/ngram-cache`: `unordered_map<ngram(4×int32),
  unordered_map<token,count>>` for n = 1..4, three tiers (context / dynamic
  = previous generations / static corpus), draft token-by-token while
  `sum_count ≥ min_sample` and `max_count/sum ≥ min_percent` (lax for
  context: {2,2,1,1} and {66,50,50,50}%, strict for dynamic). Has
  save/load/merge, so it is the one persistent n-gram tier with counts in
  a shipping engine. Memory: two nested std maps per distinct n-gram,
  hundreds of B per key; fine for 10^6 tokens, poor for 10^8.
- llama.cpp `ngram-mod` (PR #19164, merged 2026-01-30): LCG rolling hash of
  the last `n = 24` tokens → one `int32` next token in a fixed table (~16 MB,
  4M entries), shared across server slots; constant memory, no counts, lossy
  (collisions overwrite), draft 48–64 with a low-acceptance reset. Trivially
  persistent (dump the array). No scoring, no tree.
- llama.cpp `ngram-map` (PR #18471): keys = n-grams occurring ≥ `min_hits`,
  up to 4 value m-grams each with occurrence count **and `n_accepted` from
  the last draft** — acceptance feedback stored per candidate, the cheapest
  form of online calibration.
- mlx-dspark `NGramIndex` (`src/mlx_dspark/lookup.py`): dict of 3–4-gram
  tuples → (previous, latest) end positions; draft from the latest earlier
  occurrence; the match is extended backwards to measure `m`, and `m ≥ 8`
  earns a `2m` draft capped at 32; `LongDraftGate` parks long drafts after
  two consecutive rounds accepted under half length and probes every 8
  rounds. Per-request only; ~100 B per n-gram in Python.
- Verdict: hash tables give O(1) lookup and easy persistence but either
  no counts (mod), no long contexts (cache: n ≤ 4), or no cross-request
  state (mlx-dspark). They do not return a scored tree.

### 1.5 Tries with count-min sketches, LZ dictionaries, successor matrices

- Depth-capped count trie (= suffix tree without path compression) costs
  ~D nodes per token; a count-min sketch cuts memory to O(buckets) with
  ±ε·n count error but cannot enumerate children, so it can validate a
  candidate's frequency, not propose one. Useful only as a memory-bounded
  *filter* over a positional index; not a primary structure here.
- LZ78/LZW dictionaries (phrase tries): compact, streaming, but phrases end
  at arbitrary boundaries, carry no successor counts and cannot match a
  context that straddles two phrases; poor draft quality.
- Token Recycling / Graft adjacency matrix (`V × k`, < 2 MB, ~2x): a
  first-order successor table; complementary to a long-context index and
  what Graft uses to fill pruned EAGLE tree slots (60-node budget split
  8/52 → 40/20). Cheap to add later as a last-resort filler.

### 1.6 Comparison

| Structure | B/token (RAM) | Update/token | Lookup | Freq score | Tree draft | Aging/evict | Persist |
|---|---|---|---|---|---|---|---|
| Suffix tree (ArcticInference) | 11 (paper) – few hundred (est.) | ~4 µs, O(D) | ~12 µs/tok | yes | yes | per-sequence remove, O(len·D) | replay log; snapshot = new code |
| Suffix automaton | ~50–100 (est.) | O(1) amort. | O(1)+draft | no | weak | rebuild only | serialize states |
| Suffix array (mmap) | 7 | none (rebuild/segments) | O(m log n), <1 ms w/ trie | yes (interval) | yes (REST trie) | segment drop | best |
| FM-index | ~3 | none | slower (rank/step) | yes | via extract | segment | good |
| ngram-cache (n≤4) | 100s per n-gram | O(4) | O(1) | yes | no | none | save/load/merge |
| ngram-mod hash | fixed 16 MB | O(n) hash | O(1) | no | no | overwrite | dump array |
| mlx-dspark NGramIndex | ~100/ngram (py) | O(2) | O(1)+backscan | no (recency) | no | per-request | none |
| Count-min trie | O(buckets) | O(D) | O(D) | approx | no | decay counters | dump |

## 2. Persistence on SSD

### 2.1 Sizing

| Tokens in history | Token log (4 B/tok) | Tree at 15 B/tok | Tree at 300 B/tok | Replay at 4 µs/tok |
|---|---|---|---|---|
| 10^7 | 40 MB | 150 MB | 3 GB | 40 s |
| 10^8 | 400 MB | 1.5 GB | 30 GB | 7 min |
| 10^9 | 4 GB | 15 GB | out | 70 min |

The owner's transcript corpus (Claude Code + Codex, prompts + tool outputs +
outputs) is plausibly 10^7–10^8 tokens; on a 24 GiB host with a 9–16 GiB
model the tree gets **1–4 GiB**, so the datastore is quota-bound before it is
disk-bound. Everything the drafter reads at decode time must be in RAM: a
page fault on the lookup path costs ~100 µs on NVMe, the whole per-round
budget. Out-of-core is therefore only a question for the *history*, never for
the *live index*.

### 2.2 Simplest correct design (recommended for D1)

1. **Append-only token log** per namespace `(tokenizer fingerprint, project)`:
   records `[u32 len][u32 seq_id][u8 kind: output|prompt|tool][tokens…][crc32]`,
   4 B/token (varint or 3-byte ids are an optional 25–45% saving). Written
   at request end (or per N tokens) by the generation thread's write-behind
   queue; `fsync` at idle and on close, never per token. On load, a record
   with a bad CRC or truncated tail is dropped: the log is always a prefix
   of the truth, which is all a cache needs (no WAL, no torn-page problem).
2. **In-RAM tree rebuilt by replay on start**, in a background thread, most
   recent sequences first, while the per-request index serves immediately.
   At 10^7 tokens replay is ~40 s of one core; acceptable as a warm-up.
3. **Quota and eviction:** byte quota checked with a sampled
   `estimate_memory()` (walks the tree; run at idle) or a node counter
   added to the tree; evict whole sequences, oldest first (FIFO, upstream)
   or by last-hit time (LRU, needs one timestamp per `seq_id`, updated from
   `ref_seq` of the winning draft — cheap). Evicted sequences are appended
   to a tombstone list; the log is compacted at idle when > 50% dead.
4. **Snapshot** only if replay time is measured to matter (≥ 10^8 tokens):
   pre-order dump of nodes `(token, length, count, ref_seq, ref_idx,
   n_children, endpoints)` ≈ 30–40 B/node; loading is a sequential read
   plus allocation, roughly 10x faster than replay. The log remains the
   source of truth; a snapshot is a cache of the log up to an offset.

This is the whole design. No mmap of live structures, no page layout, no
compaction scheduler beyond "rewrite one file at idle".

### 2.3 Next step up if history outgrows RAM (D2, only with receipts)

LSM-style two tiers: the tree holds the hot recent window (quota-bound);
older history is a **suffix array over the compacted log**, built at idle
with libsais (int32 SA for < 2^31 tokens), stored as two flat files,
mmap'd. Lookup = longest-suffix binary search on the SA (top levels stay in
page cache; ~30 probes) → interval → REST-style trie from ≤ 256 sampled
continuations with frequency weights → score exactly as the tree does.
Consult the SA only when the tree's match is short (< 8) and only
asynchronously prefetched, never on the critical path cold. Aging: rebuild
the SA over a sliding window of segments. This is what Infini-gram and REST
already validated; it adds ~7 B/token on disk and ~0 in RAM.

## 3. Draft construction policy

Sources per round: **P** = per-request index (recency, current prompt +
output), **G** = global tree (frequency, previous sessions), **M** = model
drafter (DFlash/MTP, ~3–6 accepted tokens per round on its own).

1. **Score every source in the same unit: expected accepted tokens `E[a]`.**
   G has it natively (`score`). P and M do not: keep a running mean of
   realized acceptance bucketed by match length `m` for P (buckets 3–4,
   5–7, 8–15, ≥16) and a plain EMA for M. All three are logged every round
   (§4), so the estimates are receipt-driven.
2. **Draft or not:** draft from source S at width `w` iff
   `E[a_S](w) + 1 > c(w)·(1 + margin)`; with the measured plateau that is
   `E[a] > ~1.5` for `w ≥ 16` and `E[a] > ~0.3` for `w = 4`. Choose `w` to
   maximise `(E[a](w) + 1) / c(w)`; on the plateau, fill to the cap.
3. **Source choice:** the source with the highest `E[a]`. Ties or missing
   calibration (cold start): longest match wins, then P over G (recency is
   the better prior within a session; SuffixDecoding's ablation shows both
   trees beat either alone). ArcticInference already picks
   `draft1 if draft1.score >= draft2.score else draft2`.
4. **Length from match and frequency:** SuffixDecoding's `α·match_len` (α
   in [1, 4], default 1) plus `min_token_prob` (0.1) is the right shape;
   mlx-dspark's `2m` at `m ≥ 8` is the same rule with α = 2. On this host
   the cap is the plateau (32), not "hundreds". Start with α = 2,
   offset 0, `min_token_prob` 0.1 for G; keep `LongDraftGate` semantics as
   the reset for P (llama.cpp ngram-mod's "reset criteria" is the same
   idea); tune α and the threshold against the calibration plot, not by
   feel.
5. **Tree vs linear:** cost is by total verified width, so a 16-node tree
   costs what a 16-token path costs; the tree wins only where the top
   child's probability drops early (branching). mlx-dspark's `verify` is
   linear; tree verification needs the tree mask that mlx-vlm/DDTree has.
   D1: linear paths. Log the second-best child probability at each depth;
   if a large share of rounds shows `p2 > 0.3` at depth < 8, D2 evaluates
   `use_tree_spec` (AdaPLD's K = 8 branches and REST's 64-node trie both
   report gains from branching at the same widths).
6. **Combining with the model drafter:** per-round choice by `E[a]` (rule
   3) is the safe baseline and is what ArcticInference's hybrid does.
   Chaining is the D2 lever: when M's draft is chosen, match its *drafted*
   suffix in P/G and append the continuation to fill the plateau (Graft
   fills pruned model-tree slots from retrieval for +10–22% over EAGLE-3).
7. **Prompt-side ingestion:** index tool outputs and file contents seen in
   prompts as `kind = prompt|tool` sequences in G, deduplicated by content
   hash so a file read ten times costs one entry, under a separate quota
   (prompt tokens outnumber outputs ~10:1 in agent traces). What agents
   re-emit is mostly what they read, and the cross-session value is the
   same file being read again in later sessions on the same project.
   Measure the hit share by kind before giving it more than half the quota.
8. **Namespaces:** per tokenizer fingerprint (hard: ids differ), per
   project (soft: share by default, measure cross-project hit share in D2).

## 4. Metrics and offline evaluation

- **Acceptance simulation is exact for greedy.** Given ground-truth greedy
  outputs, a drafted token is accepted iff it equals the ground-truth token
  at that position; ArcticInference's `simulator.py` does exactly this
  (walk `parents`, stop at first mismatch, add the bonus token, update the
  cache, record `match_len`, `score`, `num_spec_toks`, `num_accept_toks`,
  `spec_ms`, `update_ms`). Reuse its loop. The only proxy in D0a is the
  text (Claude-generated stand-in); D0b removes it.
- **Projected speedup without the model:** rounds `r` with widths `w_r` and
  accepted `a_r` give `S = N / Σ_r c(w_r)` in plain-step units, with `c`
  the host's measured curve (D0c). Report at widths 8/16/32 and with the
  optimal-width policy, mean accepted per round, hit rate, tokens per pass,
  and per-source attribution; also `score` vs realized `a` (calibration).
- **Structuredness:** build a suffix tree from 100 outputs, entropy of each
  node's child distribution, weighted average over nodes. Paper range
  0.0862 (Extract) → 3.43 (WildChat), inversely related to speedup; report
  it per project so a NO_GO transfers.
- **Datastore-size curve:** speedup vs tokens cached (REST: 0.9 → 27 GB
  gave 1.83x → 2.36x; SuffixDecoding says gains keep rising) — the D2.2
  question, answerable offline by truncating the training sessions.
- **Online log, one row per round:** `ts, req, project, source
  (plain|P|G|M), match_len, score (E[a] predicted), drafted, accepted,
  committed, width, verify_ms, draft_ms, gate_state`; per request: tokens,
  rounds, tokens/pass, wall; per process: tree tokens, nodes, bytes, hit
  rate by kind, load/replay time, flush time. Every tuning knob in §3 is a
  function of these columns.

## 5. "Block storage to remove file overheads" — is there a problem?

No. The live index is 10^2–10^3 MB, lives in RAM, and is materialised from
one append-only log plus at most one snapshot per namespace. That is one
`open`, one sequential read at NVMe bandwidth (400 MB in well under a
second), one background writer. File-system overhead per operation is
microseconds and happens once per session, not per token. The 4 KiB page
question exists only for a mmap'd suffix array larger than the page cache
(§2.3), where binary-search probes are random 4 KiB reads; even there the
fix is prefetching and keeping the top levels resident, not a custom block
store. The concern is a leftover from the KV-cache tiers, where entries were
multi-MB tensors with a fixed per-entry load cost and per-turn rewrites
(the "quadratic amplification" finding in `CLOSED_LINES.md`). The
transferable lessons are only: write on eviction or at idle, never per
token; decide admission by a cost model; report load time as a receipt.

## 6. Recommendations, ranked by simplicity

1. **D1: ArcticInference `SuffixTree` (nanobind, or a faithful port if the
   macOS build fights back) + mlx-dspark `NGramIndex` unchanged**, global
   tree consulted when P has no match or a shorter one, arbitration by
   `E[a]`, linear drafts, α = 2, `min_token_prob` 0.1, cap 32, depth 32–64
   (measure bytes/token first). Persistence = append-only CRC'd token log,
   background replay on start, byte quota with FIFO by sequence. Log the
   §4 rows from day one.
2. **D1.5:** LRU-by-last-hit eviction; content-hash dedupe of prompt/tool
   sequences under their own quota; snapshot file if replay > 10 s.
3. **D2:** tree speculation only if the branching log says so; draft
   chaining after the model drafter; per-project quota split.
4. **D2, only if history > RAM:** suffix-array cold tier over the compacted
   log (libsais, mmap), REST-style trie, tree stays the hot tier.
5. **Not recommended:** suffix automaton (no counts, no delete), FM-index /
   CSA (slow per step, static), count-min tries (lossy for no gain at this
   scale), LZ dictionaries, a custom block store. `ngram-mod`'s 16 MB hash
   is the fallback if the tree's bytes/token measure badly on this corpus.

## Sources

- SuffixDecoding, NeurIPS 2025: https://arxiv.org/abs/2411.04975 (Sec. 3
  scoring and `MAX_SPEC = αp`; Sec. 4.4 global vs per-request; App. A.2.3
  structuredness; App. B Table 2: 572M tokens, 6.15 GB, 4 µs update,
  11–12 µs lookup).
- ArcticInference suffix tree and cache:
  https://github.com/snowflakedb/ArcticInference/tree/main/csrc/suffix_decoding
  (`suffix_tree.h/.cc`, `int32_map.h`),
  `arctic_inference/suffix_decoding/cache.py` (defaults: depth 64, α 1.0,
  offset 0, `min_token_prob` 0.1, FIFO by request), `simulator.py`,
  `docs/suffix-decoding.rst`.
- SAM-Decoding (ACL 2025): https://arxiv.org/abs/2411.10666.
- REST: Retrieval-Based Speculative Decoding: https://arxiv.org/abs/2311.08252.
- Infini-gram (suffix array, 7 B/token, on-disk): https://arxiv.org/abs/2401.17377;
  Infini-gram mini (FM-index, 44% of corpus): https://arxiv.org/abs/2506.12229.
- AdaPLD (June 2026): https://arxiv.org/abs/2606.05742.
- Graft, "Draft Less, Retrieve More" (May 2026): https://arxiv.org/abs/2605.20104.
- Token Recycling: https://arxiv.org/abs/2408.08696.
- llama.cpp `common/ngram-cache.{h,cpp}`, `common/ngram-map.{h,cpp}`
  (PR #18471), `common/ngram-mod.{h,cpp}` and PR #19164:
  https://github.com/ggml-org/llama.cpp/pull/19164.
- mlx-dspark `src/mlx_dspark/lookup.py` (`NGramIndex`, `LongDraftGate`,
  verify-width plateau note): https://github.com/ARahim3/mlx-dspark.
- ik_llama.cpp SuffixDecoding feature request (closed 2026-04):
  https://github.com/ikawrakow/ik_llama.cpp/issues/1602.
- Project context: `NOW.md`, `docs/CLOSED_LINES.md`.
