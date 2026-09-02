# Apple Silicon headroom for datastore speculation (M4 Pro host, MLX 0.32.2)

Research note, 2026-09-02. Companion to `APPLE_SILICON_VERIFY_ECONOMICS.md`.
Question: does the hardware offer a structurally better deal than what the
current MLX-based engine extracts, or are we near the wall? Every number is
tagged **measured** (this host, or a cited source), **fit** (derived from this
host's logs), or **model** (arithmetic from measured constants).

## 0. Summary

- The "~4.4 tok/s" is not a decode number. It is turn wall-clock / 200 tokens,
  and ~2/3 of every turn is an ~8k-token prefill that the prompt cache never
  skips: the runner's 32,000-char sliding tail window makes each turn's prompt
  a different string, so the cache "hits" (79/80) but reuses only the ~300-token
  preamble (253 of 260 turns, **fit** from the corpus, section 2). Decode itself
  runs at ~55-65 ms/token (~16 tok/s) at 4-8k context. The memo's 26 tok/s
  roofline was for short context and all 8.7 GB; the host's own calibration
  shows a width-1 step of **34.3 ms at 512 ctx (29 tok/s)** and **~43 ms at 8k**.
- Verify cost at long context is the real structural gap. This host's
  measured depth slopes say each extra verify row costs ~0.54 us per context
  token: at 8k, width 8 = 2.0 steps; at 32k, 3.0 steps. A bandwidth-bound
  multi-row attention would make width 8 cost 1.28 / 1.18 steps. That lever is
  worth up to ~1.5x on the round at 8k and ~2.5x at 32k, and nobody has shipped
  it as a reusable MLX-Python piece.
- The per-round fixed overhead is the second gap: the calibration measured
  **66 ms per DFlash round** beyond the verify and drafter kernels, on a 34 ms
  step. A datastore-only round should be far cheaper, but that is unmeasured.
- Storage is a non-issue at every size the project will reach on 24 GiB; the
  index is CPU-side and its lookup is ~0.03% of a step.
- The host is a 16-core-GPU M4 Pro (`applegpu_g16s`), not 20-core, and 18.4 GB
  of its swap is held by a dormant 8B-BF16 embedding sidecar. Neither explains
  the decode speed; the second is a risk to every long run.

## 1. What the host actually is (CPU-side inspection, 2026-09-02 22:10)

| Item | Value | Source |
|---|---|---|
| Machine | Mac16,8 (MacBook Pro), Apple M4 Pro, 8P+4E CPU, **16 GPU cores** | `sysctl`, `ioreg gpu-core-count` |
| Metal arch string | `applegpu_g16s` (MLX takes the `'s'` branch in its SDPA block heuristics) | `mx.metal.device_info()` |
| Memory | 24 GiB; max recommended working set 23.0 GiB; 21 GB used, 139 MB compressed | `top`, `vm_stat` |
| Swap | 18.4 GB used of 19.4; 18.75M swapouts since boot (300 GB cumulative); **0 MB/s disk I/O during decode** (two 5 s `iostat` samples) | `sysctl vm.swapusage`, `iostat` |
| Who owns the swap | pid 539 `persona-rag-sidecar embedding_sidecar.py --model nvidia/Nemotron-3-Embed-8B-BF16`, RSS 4 MB, vsz 505 GB: a ~16 GB model paged out | `ps` |
| Power/thermal | AC, `powermode 2` (high power), no thermal or performance warnings | `pmset -g`, `pmset -g therm` |
| Software / job | mlx 0.32.2 wheel, mlx-lm 0.31.3, mlx-dspark 0.18.0 editable; `run_d0b_v2_resume_cache.py` RSS 9.15 GB (weights + KV) | `pip show`, `ps` |

Consequences: bandwidth is unchanged (273 GB/s spec, ~226 GB/s measured by
mlx-dspark's microbench), but prefill and occupancy-limited kernels run ~20%
below 20-core numbers, and the compute crossover K* from the memo shifts down
~20% (8-bit: ~9.5-12.6). If any process queries the embedding sidecar, it swaps
16 GB back in on a 24 GiB machine while the model is resident; kill or unload
it before the next measured run.

## 2. Where the 227 ms/token goes

Evidence: the run log prints `last gen` every 10th turn (44-50 s for the
33,202-char prompts; 15.7 / 23.7 / 27.8 / 28.1 / 29.1 s for the turns whose
prompts are shorter or partially reused), the runner's own docstring records
"~30 s of ~46 s attributed to prefill", and `prefix_reuse_check` (a local
replay of `build_prompt` over the frozen corpus) shows the longest common prefix
between consecutive same-session prompts is the 1,202-char preamble on 253 of
the first 260 selected turns, >90% on 0 turns. The bimodal turn times line up
with prompt length: t ~ 11.4 s + 4.3 s per 1k prompt tokens (**fit**, three
points: 1k -> 15.7 s, 4.3k -> 23.7 s, 8.3k -> 47 s).

| Component of a 44-50 s turn (200 new tokens) | ms per emitted token | Confidence | Basis |
|---|---|---|---|
| Prefill of ~8k tokens at ~230-260 tok/s, library default path (no wide-GEMM, no CPU split: the runner never calls `calibrate.apply_*`) | ~150-170 | high | log bimodality + corpus replay + runner docstring |
| Weight stream, decode step at 512 ctx: 8.04 GB streamed (embeddings are a gather, lm_head is not) / 234 GB/s | 34.3 | high (**measured** `calibration.json` verify width 1) | |
| KV read at 8k: 1.2 GB, vector 2-pass kernel at ~62% of bandwidth (slope 1.03 us/ctx token) | ~8.5 | high (**measured** depth slope) | |
| Loop overhead in `greedy_generate`: 36-layer graph build, `create_attention_mask`, argmax, `item()` sync, streamer detokenize, prefix-cache trim/store per turn | ~10-20 | medium (**fit** residual: 57 ms/token intercept minus 43) | |
| Total | ~227 (turn) / ~55-65 (decode only) | | |

Not the cause (checked): paging (no disk I/O during decode), low-power mode,
thermal capping, memory compression, other GPU users, SDPA misdispatch at
q_len 1 (`'s'`/GQA takes `sdpa_vector_2pass` from 1,024 keys: the right kernel).

**The single measurement that settles it:** log `GenResult.prefill_seconds`,
`seconds - prefill_seconds`, `num_tokens` and the `reuse_len` returned by
`PrefixCache.acquire` per turn. The fields already exist; it costs no GPU time
beyond the run itself. Expected: prefill 28-35 s, decode 11-13 s, reuse_len
~300. If decode comes out above ~14 s, the residual is loop overhead and the
next probe is a `mx.compile`d step or a bare `target.plain` chain timed with
the calibration harness at 8k.

Corollary for the benchmark, not the product: strict-prefix prompts (token-
aligned or no window) turn 46 s turns into ~13 s with zero engine work. Prefix
caching stays oMLX's; a G1-D receipt must not carry an unreused prefill per arm.

## 3. Verify cost at long context: derivation against measured slopes

Host calibration (`~/.cache/mlx_dspark/calibration.json`, Qwen3-8B-8bit, small-M
kernel on, dependent chain, warm cache) gives verify(w) at 512 ctx and the extra
ms per context token beyond 512 per width:

| width w | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| verify ms @512 (**measured**) | 34.3 | 36.3 | 36.5 | 40.0 | 46.6 | 45.8 | 46.5 | 46.4 |
| depth slope us/ctx-token (**measured** w=1,5,8; interpolated) | 1.03 | 1.6 | 2.1 | 2.6 | 3.14 | 3.7 | 4.3 | 4.82 |

Per extra row: (4.82 - 1.03)/7 = **0.54 us per context token per row**. The
bandwidth floor for the whole KV at width 1 is 0.65 us/ctx-token (147 KB/token
at 226 GB/s), so width 1 runs at ~62% of bandwidth and every added row costs
another ~80% of a full KV read. The kernel is latency-bound per row, not
bandwidth-bound; rows in one threadgroup share the KV through cache but each
simdgroup walks its N/blocks keys serially (`sdpa_vector_2pass`, `blocks` = 64
for n_simds <= 4 on `'s'`, 128 for N <= 8192 otherwise).

Verify cost in decode steps, c(w) = verify(w, ctx) / verify(1, ctx) (**model**
from the rows above):

| ctx | step (w=1) | c(4) | c(5) | c(8) | ideal c(8): KV read once + matmul curve |
|---|---|---|---|---|---|
| 512 | 34 ms | 1.17 | 1.36 | 1.35 | 1.35 |
| 8k | 43 ms | 1.44 | 1.69 | **2.00** | 1.28 |
| 16k | 51 ms | 1.62 | 1.90 | 2.45 | 1.23 |
| 32k | 68 ms | 1.85 | 2.16 | **3.00** | 1.18 |

Answer to "does wide verification get relatively cheaper or dearer with
context": dearer, linearly. The attention term is ~0.54 us x ctx x (w-1) while
the weight term is flat, so the width that maximises accepted tokens per ms
falls from 8 at short context to ~4-5 at 32k on stock kernels; with a
bandwidth-bound multi-row attention it stays at the matmul plateau (8 on
8-bit with the small-M kernel, ~12 on the qmm side).

Published corroboration: DDTree-mlx 1.52x at 2k, **0.98x at 16k** ("attention
cost over the long prefix"); sdpa_split on this chip (Qwen3.8 shape, 32k: q5
2.48 ms -> q6 7.09 ms, tiled path recovers only at q16); the M3 Ultra sweep
(mlx #3209): width-1 decode loses only 8-12% from 1k to 8k on a 32B model, so
width-1 attention is near bandwidth, matching the 1.03 us slope here; mlx-qsdpa's
fp16 vector SDPA at 16k runs ~40 GB/s effective for H_kv = 2 (latency-bound).

End-to-end at width 8 (**model**, speedup = (A+1) / (c(8) + overhead), A =
accepted tokens per round, per-round overhead as measured or removed):

| case | 8k, A=3 | 8k, A=6 | 32k, A=3 |
|---|---|---|---|
| stock kernels, 15 ms datastore-round overhead (0.35 step, assumed) | 1.7x | 3.0x | 1.2x |
| + multi-row attention (ideal c(8)) | 2.5x | 4.3x | 2.6x |
| + overhead removed | 3.1x | 5.5x | 3.4x |

## 4. Tree / batched verification in MLX

What it takes: a tree of W nodes is W rows through every matmul (identical cost
to a linear draft of W tokens: the qmv/qmm dispatch sees only M) plus (a) an
ancestor-only boolean or additive mask of shape [W, ctx+W] passed to
`mx.fast.scaled_dot_product_attention`, (b) per-row RoPE positions
(depth-indexed, not offset+arange), (c) a KV trim to the accepted path. MLX's
SDPA accepts array masks; the vector-path gate is `q_len <= 8` and
`q_len * gqa <= 32` (Qwen3-8B, gqa 4: a tree of up to 8 nodes stays on the
vector path, Qwen3.8-27B, gqa 6: 5 nodes). mlx-lm's `KVCache`/RoPE take one
scalar offset, which is why mlx-lm cannot do it; mlx-dspark's target wrappers
call `create_attention_mask(h, cache)` and the model's own RoPE, so both (a)
and (b) need a small per-family verifier change, and hybrids additionally need
the recurrent state forked per branch.

Who has it: mlx-vlm `speculative/ddtree.py` plus per-family exact verifiers with
masked argmax kernels (`_TARGET_VERIFY_MASKED_QARGMAX_*` in
`models/qwen3_5/speculative_verifier.py`, fused q8/q4 lm_head + argmax,
bit-identical to singleton `qmv`); DDTree-mlx (ancestor masks, per-token RoPE,
GatedDelta fork kernels; 27.9 -> 42.3 tok/s at 2k, break-even at 16k); MTPLX
(precompiled verify shapes, `verify_qmv`; native app); mlx-serve (Rust,
packed-KV verify read, "32k within 2% of unquantized"; not Python-reusable);
oMLX `custom_kernels/decode_fast/sdpa_decode.metal` (no per-kernel numbers).

Realistic gain at our widths: a tree helps only where the datastore holds
several plausible continuations; the D0b replay's single path already accepts
1.5-1.65 tokens per round at width 8, and the paper's tree gains are largest on
branchy chat, not copy-heavy code. With c(8) = 2.0 at 8k, trading depth for
breadth is a wash unless breadth lifts acceptance >30%. Reuse mlx-vlm's DDTree
slot if logged branching justifies it; do not write masks. Cross-request
batching shares the weight read but not the linear attention term (measured
0.97x at B=4); single-stream agentic traffic gains nothing.

## 5. Storage for the datastore: the honest answer is "non-issue"

Sizes (**measured** by the paper / **model**): suffix tree 10.75 B/token on
repetitive output up to ~300 B/token on high-entropy text; the owner's corpus
is 10^7-10^8 tokens, so 0.1-3 GB hot in RAM on a host with 9-16 GB of model,
under the 1-4 GiB quota already in `NOW.md`. Lookup ~12 us, update ~4 us
(ArcticInference Table 2) against a 43 ms step: 0.03%. Append-only CRC'd log at
4 B/token: 10^8 tokens = 400 MB, replay ~40 s.

Apple-specific angles, checked and dismissed with numbers:

- Unified-memory zero-copy: the payload is <= 32 int32 draft ids per round and
  one accept length back. Nothing to copy.
- mmap + page cache on APFS/NVMe: matters only past RAM (10^9 tokens = 7 GB
  suffix array + 4 GB log). A cold longest-suffix search is ~30 probes; all
  misses at ~100 us NVMe latency = 3 ms (7% of a step), hidden by issuing the
  next round's lookup during the current verify's `async_eval`.
- kqueue / GCD async I/O: an append-only log needs one background thread and a
  batched `fsync` (F_FULLFSYNC costs ms; CRC per record makes a torn tail
  harmless). Writes are 4 B/token x 20 tok/s = 80 B/s.
- Huge pages: Apple Silicon already uses 16 KiB pages, no user superpage API
  for mmap. Metal buffer aliasing: the index never touches the GPU.

The design study's recommendation (in-RAM tree + append-only log, mmap'd suffix
array only as a receipted cold tier) stands; no Apple-specific storage work is justified.

## 6. Structurally different options considered

- Drafter on CPU/ANE concurrently with GPU verify: a datastore lookup is
  CPU-side and 12 us; nothing to overlap. A DFlash drafter needs the current
  round's fused hidden states, so true overlap is impossible; "speculative
  pipelining" (draft r+1 assuming full accept during verify r) caps at the
  drafter share (~8-10 ms of a ~110 ms round) minus waste. ANE: fixed shapes,
  second weight copy, ~42 ms per call (memo section 3). No.
- Quantized KV: at 8k the KV is 1.2 of ~9.3 GB per step; 8-bit KV saves ~6%
  ideal, and mlx-lm's two-call quantized SDPA is slower than fp16 (mlx-qsdpa:
  902 vs 427 us at 16k, fused path 588 us, still slower with GQA). A memory-fit
  lever for 27B-4bit at 32k, not a speed lever. Park.
- Speculation x prefix caching: cached prefixes make every turn start at
  8-30k, where c(w) is worst; the controller must price depth (mlx-dspark's
  `verify_depth` slope does) and lever 2 below is what keeps width 8 affordable.
- GPU idle during CPU lookup: with `async_eval` the GPU runs the next scheduled
  step while Python runs; the lookup fits inside that window.

## 7. Ranked levers, ceilings, cheapest confirming experiment

1. **Fix the measurement, then the engine loop overhead.** Ceiling: turn
   wall-clock 46 s -> ~13 s in D0b (benchmark hygiene, 3.5x); decode
   55-65 -> ~43 ms/token at 8k (1.3-1.5x) if the ~15 ms residual is loop
   overhead. Experiment: log `prefill_seconds` / decode seconds / `reuse_len`
   per turn (free); then time a bare 200-step `target.plain` chain at 8k with
   the calibration harness vs `greedy_generate` (2 min GPU after the job).
2. **Bandwidth-bound multi-row decode attention for widths 2-8 at 4k-32k.**
   Ceiling on the verify round: 1.56x at 8k, 2.55x at 32k (c(8) 2.0 -> 1.28,
   3.0 -> 1.18); on end-to-end speculation ~1.5x at 8k, ~2.2x at 32k. Cheapest
   experiment (10 min, no model): time `mx.fast.scaled_dot_product_attention`
   for Hq=32/Hk=8/D=128 at q_len 1,4,8 and 8,16,32 (zero-padded rows, attention
   only, sliced after) at 8k/16k/32k; if the tiled q=16 path beats the vector
   path at q=6-8 by >1.5x, an attention-only padding monkeypatch in the style
   of `sdpa_split.py` captures part of the ceiling with no kernel work. The
   full ceiling needs a `simdgroup_matrix` multi-query decode kernel (the
   FlashDecoding shape); candidates to measure first: oMLX `decode_fast`,
   mlx-serve's packed-KV read, manishklach/mlx-metal-kernels "decode attention".
3. **Per-round overhead of a datastore-only round.** The 66 ms DFlash-round
   overhead includes drafter context projections; a lookup round should be one
   verify + argmax + cumprod + one sync. Ceiling: 0.35 -> ~0.15 step per round
   (1.15-1.2x at width 8). Experiment: `measure_dspark_round_overhead`-style
   timing of the `lk_draft` branch of `speculative_generate` with a synthetic
   8-token draft at 512 and 8k (5 min).
4. **Small-M matmul kernel (exists).** Stock 8-bit cliff at width 6; the
   vendored kernel holds width 6-8 at width-5 cost (1.2-1.6x per matmul).
   Ceiling ~1.3x on the matmul share at width 8; already in the calibration
   (`|smm`). Cost: 1-2 ulp non-identity, so it is a D2 opt-in per NOW.md.
5. **Width policy by depth.** No kernel: choose w(ctx) from the measured
   slopes; at 32k stock, w=4-5 beats w=8 whenever A < ~4. Free; the controller
   already has the slope.

## 8. Not worth pursuing

- Chasing the "missing 3-5x" in kernels: it is prefill in the turn clock plus
  loop overhead; the weight-stream step runs at 234 GB/s effective, the wall
  every engine hits on this chip (llama.cpp 7B Q8_0, 16-core M4 Pro: 30.5
  tok/s = ~220 GB/s).
- Apple-specific storage engineering (mmap tiers, page-cache tuning, huge
  pages, zero-copy, kqueue): the index is small, CPU-side, 0.03% of a step.
- ANE / CPU-AMX drafting or verification, cross-request batching, quantized KV
  for speed at <= 16k, tree verification before logged branching justifies it,
  and matmul kernels (small-M kernel and mlx-vlm's exact verifiers already
  bracket the exactness trade-off).
- Any ceiling quoted from the 26 tok/s roofline or turn wall-clock: use 29
  tok/s at 512 ctx and 23 tok/s at 8k as plain-decode denominators, c(w, ctx)
  from section 3.

## Sources

- Host inspection over ssh (CPU-only), 2026-09-02 22:05-22:20: `sysctl`, `ioreg`,
  `vm_stat`, `top`, `iostat`, `pmset`, `ps`, `mx.metal.device_info()`;
  `~/.cache/mlx_dspark/calibration.json` (17:46); marathon `d0b/` runner, log
  and `logs/warmup-qwen3-8b-8bit.log` (M4 host, local-only); local corpus
  replay of `build_prompt` (`prefix_reuse_check.py`, scratchpad).
- mlx-dspark (`third_party/mlx-dspark`): `calibrate.py`, `generate.py`,
  `prefix_cache.py`, `sdpa_split.py`, `small_m_qmm.py`, `roofline.py`.
  https://github.com/ARahim3/mlx-dspark
- MLX v0.32.2 `mlx/backend/metal/scaled_dot_product_attention.cpp` (vector-path
  gate, `sdpa_vector_2pass` block heuristics) and `quantized.cpp`
  (`get_qmv_batch_limit`, `dispatch_qmv`). https://github.com/ml-explore/mlx
- llama.cpp Apple Silicon performance table (M4 Pro 16/20-core, 7B Q8_0 TG
  30.5/30.7 tok/s). https://github.com/ggml-org/llama.cpp/discussions/4167
- mlx discussion #3209, context sweep on M3 Ultra (Qwen 32B Q8: 18.3 -> 16.9
  tok/s from 1k to 8k). https://github.com/ml-explore/mlx/discussions/3209
- mlx-lm issue #763, long-context TG ~50% of llama.cpp at 30k-146k (M3 Ultra).
  https://github.com/ml-explore/mlx-lm/issues/763
- humanrouter/ddtree-mlx BENCHMARKS.md (1.52x at 2k, 0.98x at 16k).
  https://github.com/humanrouter/ddtree-mlx
- Blaizzy/mlx-vlm `mlx_vlm/speculative/ddtree.py`,
  `models/qwen3_5/speculative_verifier.py`. https://github.com/Blaizzy/mlx-vlm
- Thump604/mlx-qsdpa (fused quantized SDPA decode, M2 Ultra numbers).
  https://github.com/Thump604/mlx-qsdpa
- jundot/omlx `custom_kernels/decode_fast/csrc/sdpa_decode.metal`
  https://github.com/jundot/omlx ; manishklach/mlx-metal-kernels
  https://github.com/manishklach/mlx-metal-kernels ; youssofal/MTPLX
  https://github.com/youssofal/MTPLX ; https://github.com/ddalcu/mlx-serve/releases
- BaseRT, arXiv:2607.00501 (1.35x over MLX decode on M3/M4 Pro, native Metal).
  https://arxiv.org/abs/2607.00501
- SuffixDecoding, arXiv:2411.04975 (10.75 B/token, 4 us update, 12 us lookup).
- Qwen3-8B-8bit config (36 layers, 32/8 heads, D=128, gs64).
  https://huggingface.co/mlx-community/Qwen3-8B-8bit
