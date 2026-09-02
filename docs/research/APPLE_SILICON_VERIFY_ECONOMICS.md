# Apple Silicon verify economics for speculative decoding (MLX, M4 Pro / M1 Pro)

Research note, 2026-09-02. What it costs to verify K draft tokens in one target
forward on Apple Silicon under MLX, why, what already exists to make it cheaper,
and what that means per candidate target on a 24 GiB M4 Pro. Numbers are quoted
from the sources at the end or computed from the cost model in 1.3 ("model").

## 0. Summary

- The verify curve is not hardware. Below MLX's `qmv` batch limit (M < 13 on
  M3/M4/M5 for large layers, M < 6 on M1/M2 Pro/Max for layers wider than 4096)
  the quantized matmul dispatches a matrix-vector kernel that re-reads and
  re-dequantizes the whole weight matrix per input row (in tiles of 5 rows since
  MLX 0.32.0's `qmv_wide`). The bf16 path is flat from M=2; the quantized path
  is linear. MLX maintainers confirmed the dispatch constant is right and the
  fix is a kernel, not a threshold (issue #4265).
- A physics roofline says a 4-bit model on an M4 Pro should verify ~6 tokens for
  the price of one and an 8-bit model ~12 before compute takes over (model).
  Stock MLX is 2.5-3.5x off that in exactly the 2..12 window.
- A reusable fix exists: avlp12's `simdgroup_matrix` split-K kernel
  (`fast_qmm.py`, MIT), vendored in mlx-dspark as `small_m_qmm.py`, makes
  widths 6-8 cost about width 5 on 4/8-bit gs64 weights (1.3-1.7x per matmul
  on M4 Pro). It is not bit-identical to the singleton path (1-2 bf16 ulps).
  mlx-vlm's per-family "exact" verifiers go the other way: they reproduce the
  singleton `qmv` numerics row by row, buying bit-identity at linear cost. This
  is a real fork in the road for a project promising bit-identical output.
- A second cliff lives in attention: MLX's vector SDPA kernel is used only when
  `q_len <= 8` and `q_len * gqa_factor <= 32`, so on Qwen3.8-27B (24 Q heads, 4
  KV heads) width 6 already drops to the full-attention kernel (2.9x per call at
  32k). Each vector-path query row re-reads the KV cache, so verify cost also
  grows with context depth; mlx-dspark's `sdpa_split.py` works around it.
- Neither the ANE nor the CPU matrix units help verification at width 8-32.
  Both are prefill tools (fixed shapes, weight duplication, shared bandwidth).
- The gain from the same drafter is a ratio: accepted tokens per round divided
  by verify cost in decode steps. On a 24 GiB M4 Pro the best hosts are 8B-8bit
  and 27B-4bit; 8B-4bit has little headroom above width ~8, and the 35B-A3B MoE
  is structurally poor (each extra draft token activates new experts).

## 1. Why verify cost grows ~linearly below width ~8-13

### 1.1 The kernel-level reason in MLX (main, v0.32.2)

`mlx/backend/metal/quantized.cpp`, `QuantizedMatmul::eval_gpu`:

```
int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
if (M >= vector_limit) { qmm_splitk(...) or qmm(...); return; }
dispatch_qmv(...)   // qmv_quad | qmv_wide (M >= 2, gen >= 15) | qmv
```

`get_qmv_batch_limit(D, O, d)` by GPU generation (gen = digits in the Metal
architecture name; M1 = 13, M2 = 14, M3 = 15, M4 = 16, M5 = 17; `'d'` = Ultra):

| GPU | D,O <= 2048 | D,O <= 4096 | larger (all LLM MLP / head shapes) |
|---|---|---|---|
| M5 (non-Ultra) | 33 | 25 | 13 |
| M3 / M4 (non-Ultra), incl. M4 Pro | 13 | 15 | 13 |
| M1 / M2 Pro/Max, incl. M1 Pro | 14 | 10 | 6 |
| M1 / M2 Ultra | 32 | 18 | 12 |

Below the limit, `qmv` launches grid `(M, N/bn, B)`: one threadgroup per (input
row, output tile). Every row streams the full quantized weight matrix and
dequantizes it again, so bytes moved scale with M. Since MLX 0.32.0 (PR #3764
`qmv_wide`) rows are tiled: `n_tiles = ceil(M/5)`, each tile re-reads the
weights, and affine `qmv_wide` is only used on gen >= 15. That is why 8-bit
verify is "flat to width 5, cliff at 6" (mlx-dspark: 0.41 -> 0.66 ms on 27B-8bit
MLP shapes) and why M1 Pro (gen 13, affine `qmv_wide` off) is linear from M=2.

Above the limit, `qmm_splitk` (tiled matmul, split-K, BM = 32 for M <= 32) reads
the weights once but carries a fixed cost of ~2-2.5x a single step; on M5 the
NAX tensor path takes over. That is the "flat plateau" at width 13-32: the qmm
floor, beyond which (K > K*, section 1.3) the pass is compute-bound and rises.

Evidence that the constant is right and the kernel is the problem (MLX #4265,
closed; dependent chains, 4-bit gs64, shape (M,5120)@(17408,5120)^T):

| M | M4 Pro forced qmv (ms) | M4 Pro forced qmm (ms) |
|---|---|---|
| 8 | 0.568 | 0.915 |
| 12 | 0.845 | 0.920 |
| 13 | 1.051 | 0.918 |
| 16 | 1.125 | 0.921 |

"Zero rows on either machine where the current constant picks the slower path."
Maintainer (zcbenz): "For small M we currently rely on the qmv_wide kernel, PRs
are welcomed if you can optimize it." MLX 0.32.1 raised the limit only on
M5-class GPUs (#3791); 0.32.2 touched only NAX `qmm_t_nax` (M5). Nothing in
0.32.1/0.32.2 changes the M4 Pro curve.

Whole-model effect (M3 Ultra, Qwen3.8-27B-4bit hybrid, one forward at S rows,
#4265): S=1 29.7 ms, S=4 45.4, S=7 62.5, S=8 77.1, S=32 105.4, S=128 347.9.
Efficiency vs roofline is worst (3.65x) at exactly S=8.

### 1.2 The attention cliff (second knee)

`mlx/backend/metal/scaled_dot_product_attention.cpp`: the vector kernel
(`sdpa_vector` / `sdpa_vector_2pass`) runs when `q.shape(2) <= 8` and
`query_sequence_length * gqa_factor <= 32`; otherwise the full tiled kernel.
Its grid is `(B*H, q_len, 1)`: one threadgroup per query row, so each row
re-reads the KV cache. Measured by mlx-dspark on M4 Pro / mlx 0.32.1:

- Qwen3.8 shape (Hq=24, Hk=4, gqa 6, D=256, L=32k): q5 2.48 ms -> q6 7.09 ms
  (2.9x), flat through q15, tiled path recovers at q16. The gate is 6 x 6 = 36
  > 32. Qwen3-8B (32/8, gqa 4) and Gemma-4-12B keep the vector path to q=8.
- Fixed cap-7 verify on Qwen3.8-27B-4bit fell from 1.17-1.41x at 2k context to
  0.53-0.97x at 32k; cap 3 still gave 1.48x. Verify cost is not depth-flat,
  which matters for agentic-coding contexts of 8-32k.
- mlx-serve saw the same ("verify steps 6-9 tokens wide at head-dim 256 ran
  MLX's slow attention fallback") and split them into two passes.

### 1.3 Theoretical cost model and the crossover K*

Decode step = weight bytes / bandwidth (memory-bound, ~1 FLOP/byte). Ideal
verify of K tokens = max(weight bytes / BW, 2 * params * K / F) plus ~1-2 ms
fixed. The crossover where compute equals memory time is independent of model
size; it depends only on bytes per parameter and the chip's FLOP/byte ratio:

    K* = (bytes per param) * F / (2 * BW)

M4 Pro (20-core GPU): 273 GB/s spec, ~226 GB/s measured by mlx-dspark's decode
microbench; GPU FP32 peak 20 cores x 256 FLOP/clock x ~1.58 GHz ~= 8.1 TFLOPS
(Apple GPUs retire 128 FP32 FMA per core-clock; FP16 is at best 2x for
vectorised half4 code, and MLX's dequant kernels accumulate in fp32). The "~17
TFLOPS fp16" figure belongs to an M4 Max 40-core, not the Pro. Achievable
dequant-GEMM throughput is lower: #4265 used a ~20 TFLOPS roofline for an
80-core M3 Ultra (0.25 TFLOPS/core), i.e. ~5 TFLOPS on 20 cores. Model results:

| bits/param | F = 5 TFLOPS eff, 226 GB/s | F = 8.1 peak, 273 GB/s |
|---|---|---|
| 4-bit (0.5625 B/param) | K* = 6.2 | K* = 8.3 |
| 8-bit (1.0625 B/param) | K* = 11.8 | K* = 15.8 |

Ideal verify(K)/step at 5 TFLOPS eff, 226 GB/s, +1.5 ms fixed (model):

| K | 1-6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| 27B-4bit (69.7 ms step) | 1.00 | 1.28 | 1.91 | 2.54 | 3.80 | 5.06 |
| 8B-8bit (38.6 ms step) | 1.00 | 1.00 | 1.02 | 1.35 | 2.00 | 2.66 |

Against measured curves: stock MLX on 4-bit is ~1.5x at K=4 and ~2.6x at K=8
(#4265 whole model) where the ideal is 1.0 and 1.28; the qmm plateau at 13-32
(~2.5x, mlx-dspark; 3.5x at S=32 on M3 Ultra) sits where the ideal is
compute-bound anyway (2.5x at K=16 for 4-bit). The kernel gap is concentrated
in K = 2..12; past K ~ 16 on 4-bit weights an M4 Pro is genuinely compute-bound
and no kernel makes width 32 cheap. On 8-bit weights the memory-bound window
extends to K ~ 12-16, the physical reason mlx-dspark's ratios peak on 8-bit
targets (3.63x on 27B-8bit vs ~2.1x on 27B-4bit at the same verify width).

Consistent external measurements: "Lossless but Not Free" (arXiv 2607.17283,
M3-class 18 GB MacBook, llama.cpp Metal, quantized target): verification 51.0
ms at K=2, 74.7 at 3, 164.9 at 5, 218.5 at 7, then flat 141-142 ms for 9-13,
attributed to per-position mat-vec products re-reading the weights until a
kernel-path switch at 8-9 (same dispatch structure as MLX). mlx-lm discussion
#890 (EAGLE-3, M3 Ultra, 8B-4bit, 8.8 ms step): ~1 ms per extra verified token,
0.4 ms draft head, 0.5 ms sync, 34% acceptance -> 1.05x.

## 2. Existing custom verify kernels on Apple Silicon

| Project | What it is | Gain | Reusable? |
|---|---|---|---|
| avlp12/mlx-lm `fast_qmm.py` (MIT) -> mlx-dspark `small_m_qmm.py` | 8x8 `simdgroup_matrix` MMA tile, split-K over 8 simdgroups, dequant once per 64-group; gated to M in [6,8], N >= 4096, K % 512 == 0, 4/8-bit gs64 affine | 1.31-1.70x per matmul at M=6-8 on real 27B shapes (M4 Pro, mlx 0.32.0); forward S=8 77.1 -> 43.3 ms; 27B-4bit cap 5: 23.7 -> 29.2 tok/s | Yes: one `mx.fast.metal_kernel` file, per-shape runtime probe, force-disabled on M5 (stalls after ~105 s). Not bit-identical (1-2 ulps). |
| mlx-vlm `models/*/speculative_verifier.py` + `exact_speculative_verify.py` (Qwen3.5/3.8, Gemma 4, LFM2, GLM) | "Singleton-equivalent numerics": Metal kernels reproducing MLX's `qmv` pack/scale/accumulate order per row, fused argmax, token-tiled/streamed and tree-masked variants | Exactness (argmax identical to K singleton steps), not speed; per-row cost | Yes for bit-identity; consumed by oMLX ("per-family exact verifiers"). |
| MTPLX | `verify_qmv` small-M qmv with `unroll_count(4)` for M=3..6, fused GDN verify kernel, precompiled "GraphBank" verify shapes | Qwen3.8-27B ~20 -> ~47 tok/s on M5 Max at MTP depth 3 | Native app; kernels not packaged. Same idea as `qmv_wide`, tuned. |
| mlx-serve (Rust/Metal) | fused GDN verify prework (mlxfast-challenge `qwen35_packed_gdn_prework`), packed-KV verify read kernel, per-machine round-cost table, auto depth 8 on 27B | 28 -> 61 tok/s at 11k with quantized KV; depth 8 vs 6 up to +31% on 27B-4bit | Not an MLX-Python library; a reference for what to measure. |
| oMLX `bonsai/csrc/spec_decode.metal` | Greedy prefix-match accept kernel (one thread per row) plus Bonsai 1/2-bit `qmv_wide` for M=1..5 | Removes host accept loop; matmuls are the same small-M story | Trivial; not a wide-verify path. |
| oMLX `decode_fast/sdpa_decode.metal`, "fused target-verification attention" | Decode/verify attention variants | Addresses the cliff of 1.2 | Through oMLX only. |
| BaseRT (arXiv 2607.00501) | Native Metal runtime, 1.35x over MLX decode on M3/M4 Pro | No speculative decoding in the paper; engine binary proprietary | No. |
| DDTree-mlx (humanrouter) | Tree masks + per-token RoPE positions; Metal kernels forking GatedDelta state per branch in hybrids | +10-15% over DFlash on code; 27.9 -> 42.3 tok/s on Qwen3.5-27B-4bit (M3 Ultra), budget 4 | Yes as tree-verify reference (section 4). |

Recommendation: do not write a wide-verify matmul kernel. Adopt mlx-dspark's
vendored small-M kernel (probe-gated per chip x mlx version) if the project
accepts "greedy-correct under verify" numerics, or mlx-vlm's exact verifier if it
insists on bit-identity, and spend effort on acceptance length and the
attention/depth behaviour. Upstream invited a PR on `qmv_wide`; the durable win
for everyone is raising its 5-row tile to 8 with the MMA formulation, which
avlp12 offered to do.

## 3. Other silicon: ANE and CPU matrix units

- ANE (oMLX 0.6.1-0.6.3 "Qwen ANE prompt processing", experimental, private
  runtime interfaces): prefill MLP (optionally GDN) projections only, in fixed
  2048-token chunks compiled per shape. M3 Max, Qwen3.8-27B-oQ4e: prefill +4.1%
  at 4k, +13.2% at 16k, +57% at 32k (includes GPU thermal relief); decode
  unchanged ("still GPU"), -16% at 32k. Costs: +4.15 GB peak memory (16 -> 33 GB
  in another report), load 3 s -> 27-29 s. hybrid-ane-mlx-bench (M2 Ultra):
  ~42 ms per sequential ANE call regardless of model size; CoreML compile for
  seq512 took 97 min. mlx-dspark tried the ANE for co-prefill and rejected it:
  its fast formats cannot hold MLX group-quantized weights exactly, and the
  exact fp16 form is a 34 GB copy of a 27B's MLP. Verdict for verify width 8-32
  on 24 GiB: not realistic (fixed shape per width, a second weight copy that
  does not fit, per-call latency comparable to the whole verify step).
- CPU matrix units (AMX/SME via Accelerate; mlx-dspark "CPU co-prefill",
  FusionML arXiv 2607.22785): a calibrated ~0.3 fraction of rows of each wide
  quantized matmul (weights dequantized once on the GPU, rows to MLX's CPU
  stream at ~3 TFLOPS) gives 1.29-1.41x prefill on M4 Pro, +0.4 GB. FusionML
  states the boundary: "decode cannot benefit, bound by shared bandwidth
  co-execution does not add." At M=8-32 both engines read the same weights over
  one 273 GB/s bus; only the compute-bound tail (4-bit, K >= 16) could gain,
  ~40% compute at best, with bandwidth contention and an fp16 layer copy. Not
  worth building; if tried later, mlx-dspark's mechanism (same arrays, no copy,
  no thread) is the way.
- M5's NAX tensor units change the picture (qmv limit 25-33, NAX qmm). The
  owner's M4 Pro and M1 Pro have none.

## 4. Tree verification on MLX

- mlx-lm: `KVCache` offset is one integer and RoPE takes one offset, so tree
  drafts are blocked without model changes (#890: "Tree verification requires
  different position IDs for candidates at different depths"). Not a target.
- mlx-vlm: `speculative/ddtree.py` builds the DDTree (Algorithm 1, node budget,
  per-depth top-k) and the Qwen3.5/Gemma-4 exact verifiers carry masked argmax
  variants (`_TARGET_VERIFY_MASKED_QARGMAX_*`) for tree masks. DDTree-mlx
  (depends on dflash-mlx) has the full path: ancestor-only masks, per-token RoPE
  positions, Metal kernels forking GatedDelta state per branch. Default budget
  is 4 nodes because hybrid layers pay per node.
- mlx-dspark: linear drafts only (verify width up to 32); batched speculative
  decoding is dense-only because per-row rollback of different lengths "needs a
  masked batched re-run that does not exist yet".
- For a datastore drafter: a tree of W nodes costs the same verify matmuls as a
  linear draft of W tokens (M = W rows) plus a mask and per-row positions in
  attention (MLX SDPA accepts an array mask; the vector-path gate of 1.2 is by
  q_len, so a 4-node tree behaves like width 5). Hybrid GDN targets need the
  recurrent state forked per branch, which is DDTree-mlx's kernel. Cheapest
  route: feed several datastore continuations as a tree into mlx-vlm's DDTree
  slot; stay linear on hybrids unless DDTree-mlx's kernels are adopted.

## 5. What it means per target on a 24 GiB M4 Pro

Assumptions: 226 GB/s measured bandwidth; 0.5625 B/param at 4-bit gs64 and
1.0625 at 8-bit; verify width W ~= 1.33 x accepted tokens A; drafter cost 5% of
a step (datastore lookup is CPU-side). c(W) = verify cost in decode steps,
"stock" = MLX 0.32.2 as shipped, "kernel" = with the small-M kernel; anchored on
#4265 whole-model numbers (S=4 1.53x, S=8 2.6x stock / 1.46x kernel),
mlx-dspark's 8-bit curve (flat to 5, cliff at 6; 27B-8bit round = 1.52x step at
width 8), oMLX's 27B-4bit k=3 round (1.28x step incl. MTP head), and the
compute-bound tail of 1.3 for W >= 16. Speedup = A / (c(W) + 0.05). Model.

| Target | Fits 24 GiB | Step ms / tok/s | c(4) stock/kernel | c(8) | c(16) | c(32) | A=3 | A=6 | A=12 | A=24 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B-4bit | yes (4.6 GB) | 20 / ~49-60 | 1.45 / 1.25 | 2.5 / 1.55 | 2.6 | 5.0 | 2.0 / 2.3x | 2.4 / 3.8x | 4.5x | 4.8x |
| Qwen3-8B-8bit | yes (8.7 GB) | 39 / ~26 | 1.15 / 1.10 | 1.9 / 1.35 | 2.0 / 1.8 | 2.8 | 2.5 / 2.6x | 3.1 / 4.3x | 5.9 / 6.5x | 8.4x |
| Gemma-4-12B-8bit | yes (12.8 GB) | 56 / 17.8 measured | 1.15 / 1.10 | 1.9 / 1.35 | 2.0 / 1.8 | 2.8 | 2.5 / 2.6x | 3.1 / 4.3x | 5.9 / 6.5x | 8.4x |
| Qwen3.8-27B-4bit | tight (15.8 GB + KV) | 70 / ~14 | 1.45 / 1.25 | 2.5 / 1.5 | 2.6 | 5.1 | 2.0 / 2.3x | 2.4 / 3.9x | 4.5x | 4.7x |
| Qwen3.8-27B-8bit | no (29.8 GB) | 132 / 8.4 measured on 48 GB | - | - | - | - | - | - | - | - |
| Qwen3.6-35B-A3B-4bit MoE | barely (19.7 GB) | 11.5 / 86.9 measured | 3.3 | 5.8 | 9.0 | 12.0 | 0.9x | 1.0x | 1.3x | 2.0x |
| Qwen3.8-Flash-Next (125B-A6B) | no (~70 GB at 4-bit) | - | - | - | - | - | - | - | - | - |

For a non-expert owner:

- The method is model-agnostic: the drafter proposes tokens, the target checks
  them all in one forward, and the target decides the output either way.
- The gain is not: it is (tokens accepted per round) / (cost of that round in
  decode steps), and the denominator is what changes between models. A step on
  27B-4bit is 70 ms of pure weight streaming; adding rows is cheap until the GPU
  runs out of arithmetic (K ~ 6-8 at 4-bit). A step on 8B-4bit is 20 ms and so
  short that every fixed overhead (launches, sync, the drafter) is a large
  fraction of it, which is why EAGLE-3 got 1.05x there. 8-bit weights double
  the bytes per step without doubling compute, so the free-verify window
  extends to K ~ 12-16 and the same accepted length buys a bigger ratio
  (mlx-dspark: 3.63x on 27B-8bit vs ~2x on 27B-4bit).
- MoE targets look fast (87 tok/s) but each extra draft token routes to other
  experts: with 128 experts, top-8, verifying 8 tokens touches ~52 distinct
  experts per layer, 5.6x the bytes of one token (model). Bytes grow almost
  linearly with K until the whole 19.7 GB is read, so A=3 is a wash and A=24
  gives ~2x; mlx-dspark measured 1.32x.
- On a 24 GiB M4 Pro the hosts where a datastore drafter pays are Qwen3-8B-8bit
  (best ratio, 26 tok/s baseline, 8.4x ceiling at A=24) and Qwen3.8-27B-4bit
  (best quality, 14 tok/s, ~4.5x ceiling, no benefit past width ~16).
  Gemma-4-12B-8bit sits between and is the model mlx-dspark measured at 4.5x on
  copy-heavy edits with lookup drafts.
- M1 Pro (16 GiB, 200 GB/s, gen 13): affine `qmv_wide` is off, the qmv limit is
  6 for wide layers, no NAX; the stock curve is linear from width 2 and the
  small-M kernel matters more. Only 8B-4bit / 12B-4bit fit with context.

## 6. Bandwidth-independent levers

- Prefix caching: removes prefill (mlx-dspark 38.3 s -> 0.24 s on repeats; oMLX
  SSD cache) but makes deep contexts the norm, and every verify row re-reads
  the KV cache in the vector attention path, so at 8-32k the optimal width
  drops (cap 3 beats cap 7 at 32k on 27B-4bit). A datastore drafter must fold
  context length into its width decision, as mlx-dspark v0.14 does with a
  measured depth slope.
- Batching two requests' verifies: rows from different requests share one
  weight read like draft rows do, and past the qmm knee the curve is flat in
  rows ("at B=4 the measured verify curve flattens past the qmm knee"). But
  per-row verify on a batched target is no longer cheap: verify(4)/verify(1)
  rose from 1.11x at B=1 to 2.10x at B=16 on Qwen3-4B, and batched dspark at
  B=4 was 0.97x of batched baseline. Batched speculation also cannot roll rows
  back by different lengths without a masked re-run, and mlx-lm cannot rewind
  hybrid caches (#1446). Speculation's value is largest for a single
  interactive stream, which is the agentic-coding case.
- Quantized KV (kv-bits 8): attention on a quantized cache is a chain of small
  quantized matmuls with M = q_len, so it inherits the per-row behaviour;
  mlx-serve measured "28 tok/s where the dense read does 60" past 8k with
  speculation on, until a packed-read verify kernel brought 32k within 2% of
  unquantized at half the memory. On stock MLX, kv-bits 8 makes the verify
  curve worse at long context; it is a memory lever, not a speed lever, though
  it may be required to fit 27B-4bit with 32k context in 24 GiB.
- Exactness vs amortization: the batched/MMA verify path differs from the
  singleton path by 1-2 ulps and flips ~0.5% of near-tie tokens; bit-identity
  to plain decoding forces per-row numerics (mlx-vlm exact verifiers) and hence
  the linear curve, unless plain decoding itself uses the same row-independent
  kernel (avlp12: the MMA kernel is 0.62x of qmv at M=1). Define
  "bit-identical" against the verify kernel's own greedy path, or accept ~1.5x
  per-round cost at width 8 for true singleton equivalence.
- Measurement doctrine every source converged on: time dependent chains, rotate
  weights across calls, re-measure per (chip x mlx version); independent-call
  microbenchmarks made a kernel look 1.13x that was 26% slower in the model.

## Sources

- MLX source (main, 2026-09-02): `mlx/backend/metal/quantized.cpp`,
  `scaled_dot_product_attention.cpp`, `device.cpp`. https://github.com/ml-explore/mlx
- MLX releases v0.32.0 (PR #3764 qmv_wide), v0.32.1 (PR #3791), v0.32.2.
  https://github.com/ml-explore/mlx/releases
- MLX issue #4265 (avlp12, erwinzhang7, zcbenz). https://github.com/ml-explore/mlx/issues/4265
- avlp12/mlx-lm `mlx_lm/fast_qmm.py` (MIT). https://github.com/avlp12/mlx-lm
- ARahim3/mlx-dspark README, `src/mlx_dspark/small_m_qmm.py`, `sdpa_split.py`,
  `roofline.py`. https://github.com/ARahim3/mlx-dspark
- mlx-lm discussion #890. https://github.com/ml-explore/mlx-lm/discussions/890
- P. Chordiya, "Lossless but Not Free", arXiv:2607.17283, section 6.5.
- BaseRT, arXiv:2607.00501; https://github.com/basecompute/baseRT
- FusionML: Prefill, Not Decode, arXiv:2607.22785.
- Blaizzy/mlx-vlm `models/exact_speculative_verify.py`,
  `models/qwen3_5/speculative_verifier.py`, `models/gemma4/speculative_verifier.py`,
  `speculative/ddtree.py`. https://github.com/Blaizzy/mlx-vlm
- humanrouter/ddtree-mlx. https://github.com/humanrouter/ddtree-mlx
- jundot/omlx releases 0.6.1-0.6.3, issue #2781 (ANE prefill on M3 Max),
  `custom_kernels/bonsai/csrc/spec_decode.metal`, `bonsai/fast.py`. https://github.com/jundot/omlx
- AtomGradient/hybrid-ane-mlx-bench. https://github.com/AtomGradient/hybrid-ane-mlx-bench
- ddalcu/mlx-serve release notes. https://github.com/ddalcu/mlx-serve/releases
- MTPLX, "How it works". https://mtplx.com/how-it-works/
- Weschera/Qwen3.8-27B-oMLX-MTP-Mac (M4 Max: 24.6 -> 72.1 tok/s, k=3).
  https://github.com/Weschera/Qwen3.8-27B-oMLX-MTP-Mac
- oMLX benchmark, Qwen3.8-27B-oQ4e-mtp on M4 Pro 20c. https://omlx.ai/benchmarks/performance/yd0ekfoi
- philipturner/metal-benchmarks (Apple GPU FMA rates). https://github.com/philipturner/metal-benchmarks
- M4 Pro specs (273 GB/s, 20-core GPU). https://www.macrumors.com/2024/10/29/apple-announces-m4-pro-chip/
- Model sizes: https://unsloth.ai/docs/models/gemma-4 , https://unsloth.ai/docs/models/qwen3.8 ,
  https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
