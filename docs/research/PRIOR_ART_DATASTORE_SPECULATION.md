# Prior art: persistent cross-session datastore drafting for speculative decoding

Research pass 2026-09-02. Question: has anyone shipped a **persistent, cross-session,
on-disk datastore of previously generated/seen token sequences** used as a speculative
drafter (SuffixDecoding-style global tree / n-gram cache) for local LLM inference,
especially on Apple Silicon, composed with a model-based drafter (MTP / DFlash / EAGLE)
and measured on agentic coding traffic?

Method: GitHub API (repo + code search, PR/issue bodies, file contents), web search,
arXiv. Star counts and dates are as fetched on 2026-09-02. "Persistent" below means
survives process restart; "cross-request" means shared inside one server process.

## 1. Cross-request / persistent datastores in mainstream engines

### llama.cpp `examples/lookup` + `common/ngram-cache.*` (the only true persist-to-disk implementation)
- Origin: PR #5479 "lookup: complement data from context with general text statistics"
  (JohannesGaessler, merged 2024-03-23). Three tiers: `nc_context` (current request),
  `nc_dynamic` ("based on previous user generations"), `nc_static` ("generated from a
  large text corpus, used for validation").
- Data structure (`common/ngram-cache.h`): `unordered_map<common_ngram, unordered_map<token,count>>`,
  n-gram sizes `LLAMA_NGRAM_MIN=1 .. LLAMA_NGRAM_MAX=4`, static cache built with
  `LLAMA_NGRAM_STATIC=2`. Drafts: rank continuation tokens by product of context and
  corpus frequencies, fall back to corpus-only. Format: flat binary
  (`common_ngram_cache_save/load`), tools `lookup-create` (build from a text file, e.g.
  WikiText-103), `lookup-merge`, `lookup-stats`.
- Persistence: the standalone `llama-lookup` binary loads `--lookup-cache-static` /
  `--lookup-cache-dynamic`, and at the end of a run merges the context cache into the
  dynamic cache and **saves it to disk** (`examples/lookup/lookup.cpp` lines 220-222).
  This is the one genuine cross-session learning path, but only in a single-prompt CLI demo.
- In `llama-server`, `--spec-type ngram-cache` loads both files but never writes:
  `common/speculative.cpp` `create_state_ngram_cache()` hardcodes
  `save_static = false; save_dynamic = false;` with a `// TODO` comment. So the server
  path is a read-only warm start, not a learning datastore.
- Measured results: PR #5479 reported at most ~1.11x on Miqu-70B q5_K_M (3x P40).
  thc1006/qwen3.6-speculative-decoding-rtx3090 (34 stars, 19 configs, RTX 3090,
  Qwen3.6-35B-A3B): `ngram-cache` -19.0%, `ngram-mod-n24` -10.9%, best n-gram variant
  -0.3% (net-negative across the board on that MoE + Ampere combination).
- Usage: no indexed Reddit/HN threads on `lookup-cache-*`; the feature is essentially
  unused. `lookup.bin` static files are not distributed.

### llama.cpp `--spec-type ngram-mod` "hash pool" (PR #19164, ggerganov, merged 2026-01-30)
- Rolling LCG hash of the last n tokens -> next token; ~16 MB, constant memory; variable
  draft length; **single pool shared across all server slots** ("different requests can
  benefit from each other"). Sibling types `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`
  (PR #18471, srogmann, self-speculative on current history only).
- Persistence: none. No save/load, no flag; PR thread and issue search show no
  persistence request. Open follow-up PR #19294 (score-based pruning of the pool).
- Composition: `--spec-type draft-mtp,ngram-mod` runs both, but as independent draft
  streams (issue #23184, closed: "no benefit over draft-mtp alone" in that user's test);
  router-mode config bug #24507. A May 2026 video claims Qwen3.6-27B 22 -> 56 tok/s with
  "MTP + ngram stacked" (unverified, no flags visible).
- Ported to ik_llama.cpp (PR #1261, merged 2026-02-13). ik_llama.cpp issue #1602
  (2026-04-09) requested SuffixDecoding with an optional mmap'd offline corpus tree;
  closed with no implementation.
- Metal: works on Apple Silicon like any backend, but nobody has published M-series
  numbers for it beyond anecdotes.

### vLLM SuffixDecoding via snowflakedb/ArcticInference (470 stars)
- vLLM PR #25784 merged 2025-11-03 (`vllm/v1/spec_decode/suffix_decoding.py`).
  Config: `suffix_decoding_max_tree_depth` (64), `suffix_decoding_max_cached_requests`
  (10000 in vLLM; 100000 in Arctic docs; 0 disables global cache), `max_spec_factor`,
  `min_token_prob`.
- Data structure: C++ suffix tree with path compression (`csrc/suffix_decoding/suffix_tree.{h,cc}`,
  nanobind, no CUDA dep), per-request tree + one **global tree of completed responses**,
  FIFO eviction by request. `arctic_inference/suffix_decoding/cache.py`
  (`SuffixDecodingCache.start_request/stop_request/evict_cached_response/speculate`).
- Persistence: **in-memory only**. No serialize/save/load in `suffix_tree.h`, no pickle
  support in `bindings.cc`, no warm-start flag. The tree dies with the process.
- Drafts: walk tree from the longest matching suffix, expand top children by frequency,
  adaptive length via `max_spec_factor` and `min_token_prob`.
- Measured: paper (arXiv 2411.04975, NeurIPS 2025 spotlight, 8xH100) 5.3x AgenticSQL,
  2.5x SWE-Bench (1.7x over PLD), ~10.75 bytes/token, global tree beats per-request tree
  on all AgenticSQL tasks but one; on SpecBench suffix alone loses to EAGLE-3 and a
  hybrid (suffix score > tau else EAGLE-3) gets 2.5x vs 2.4x. Snowflake Jan-2026 blog:
  up to 5.3x agentic, 1.96-3.12x BlazeEdit, 1.0-1.28x SpecBench.
- Composition: built in. "Arctic Speculator + Suffix" hybrid picks per iteration
  (docs table: SWE-Bench 75.8 -> 175 tok/s n-gram, suffix column higher).

### Others
- SGLang: `--speculative-algorithm NGRAM` (CUDA only, trie depth 18, capacity 10M
  entries); docs do not state cross-request sharing or persistence. Suffix decoding
  request #13165 (2025-11-12) closed "inactive"; 2026-Q2 roadmap lists DFlash/Spec V2, not suffix.
- TensorRT-LLM NGram: `is_public_pool` (shared pool across requests), `is_keep_all`,
  `is_use_oldest`; in-memory only.
- HF transformers prompt lookup (`prompt_lookup_num_tokens`), vLLM `ngram` proposer,
  MTPLX `context_copy.py`: per-request.
- Ollama: MTP on the MLX runner (experimental `DRAFT` directive); no n-gram drafting
  (issues #5800, #9216 open for years). LM Studio: `ngram-mod` UI request #1531 open
  (2026-02-17). koboldcpp: llama.cpp `--spec-type` not exposed (discussion #2160).
  llamafile: nothing found. exllama, ktransformers (NGRAM worker added 2026): in-memory.

## 2. Research systems (static corpus or per-request; none persist online learning)
- REST (NAACL 2024, FasterDecoding/REST, 221 stars): offline datastore + suffix array on
  disk, CPU retrieval, draft trie. Static; not updated with generations. CREST compacts it.
- SAM-Decoding (arXiv 2411.10666, hyx1999/SAM-Decoding, 53 stars): static suffix
  automaton from corpus + dynamic per-request SAM, O(1)/step; +3.28-11.13% on top of EAGLE-2.
- Cacheback (arXiv 2511.21699): frozen n-gram LRU table prepared offline (hosted on HF)
  + dynamic per-task table; 1.86x Vicuna-7B SpecBench; combinable with other SD as branches.
- Token Recycling (ACL 2025): adjacency matrix <2 MB, per-run; smpanaro/token-recycling (9 stars).
- AdaPLD (arXiv 2606.05742, June 2026): per-request unified memory with semantic
  fallback, up to 3.10x. Graft / "Draft Less, Retrieve More" (2605.20104): retrieval
  branches grafted into EAGLE-3 tree, 5.41x. RACER (ACL 2026 Findings), Goose (2604.02047):
  retrieval + logits hybrids. "Stateful Inference for Multi-Agent Tool Calling"
  (2605.26289, LayerScale): persistent KV across turns + prompt-lookup drafter.
- "An Empirical Study of Speculative Decoding on SE Tasks" (2604.26469): code tasks are
  predictable enough that model-free methods give tangible gains; agentic loops add noise.
- None of these papers persist an online-learned datastore across process restarts; the
  "persistent" component is always a frozen corpus index.

## 3. Apple Silicon / MLX (verified from source)
| Project | Stars | Datastore scope | Persist | Composes with model drafter | Measured |
|---|---|---|---|---|---|
| ARahim3/mlx-dspark `src/mlx_dspark/lookup.py` | 631 | `NGramIndex` constructed inside `lookup_generate` (line 189), per request | no | hybrid with DFlash/DSpark by default | 4.5x copy-heavy edit (Gemma-12B) |
| raullenchai/Rapid-MLX `vllm_mlx/speculative/suffix_decoding.py` | 3646 | dict suffix index over prompt+generation, per request ("Per-request bookkeeping") | no | separate route from MTP; PLD (`spec_decode/mtp/prompt_lookup.py`, immutable prompt only) composed with native MTP | suffix opt-in: Gemma-4-12B/31B 1.44-1.56x copy/edit/tool-XML, 0.77x JSONL, regresses GPT-OSS (0.86x) and Qwen3.6-27B (0.88-0.95x); MTP+PLD on Qwen3.8-Flash-Next-4bit 2.00-2.29x over MTP-only, 98.1% acceptance, SHA-identical (M3 Ultra, 2026-07) |
| marzukia/qMLX | 30 | same drafter (fork) | no | - | - |
| ddalcu/mlx-serve `src/pld_index.zig` | 1076 | linear scan over committed prompt+generated, per request | no | default layer under drafters and MTP with acceptance gating | up to 2.1x echo workloads |
| joongom/mlx-soloheaven `engine/pld.py` | 27 | prompt-only index (generated tokens excluded to avoid loops) | no | - | - |
| MidasMulli/ngram-engine | 0 | n-gram table over verifier output, conversation-global in session (n=4, K=16, truncate on miss); `ngram_table_27b.pkl` present | pickle file exists, drafter disabled in production (phrase loops under Gemma 4) | no | none reported |
| fmsouza/lookup-spec-metal | 0 (created 2026-08-29) | n-gram over prefix+generated, per request | no | plans merge into mlx-vlm MTP DDTree | offline proxy only: 2.01x tokens/pass on 20 agentic traces at ~6.4K prefix vs 1.24x WikiText; "no decode loop yet" |
| joelvarun/suffixspec | 0 | replay simulator, no model | no | no | 3.81x vs 4.21x fixed PLD, -49.9% wasted verify |
| vukrosic/qwen3-prompt-lookup-robustness | 1 | benchmark | no | no | adaptive PLD did not beat fixed-2 |
| jundot/omlx | 21309 | none (SSD cache is KV prefix only; speculative = mtp/dflash/eagle3 via mlx-vlm, `specprefill`) | n/a | n/a | n/a |
| Blaizzy/mlx-vlm, youssofal/MTPLX, dflash-mlx forks | - | no n-gram beyond MTPLX per-request `context_copy` | no | - | - |

GitHub searches ("suffix tree speculative mlx", "ngram cache speculative apple silicon",
"persistent draft cache", "speculative memory mlx", "lookup decoding mlx", "prompt lookup
mlx", "lookup-cache mlx", "draft cache llm persistent", "generation cache speculative")
returned nothing beyond the rows above. HF posts/spaces, r/LocalLLaMA and HN yielded no
persistent-datastore drafter on Mac.

## 4. Verdict
- **Persistent cross-session drafting datastore, in general:** only llama.cpp's 2024
  `lookup` example truly writes a learned n-gram cache to disk, and only in a CLI demo with
  1-4-gram counts; the server path never saves, and the measured effect is <=1.11x or
  negative. Everything else that is "global" (vLLM/Arctic suffix tree, llama.cpp
  ngram-mod pool, TensorRT-LLM public pool, SGLang trie) is process-lifetime RAM.
- **On Apple Silicon:** every lookup/suffix drafter (mlx-dspark, Rapid-MLX, mlx-serve,
  mlx-soloheaven, fmsouza, ngram-engine) is per request or per session; none persists;
  none has a global tree over past outputs.
- **Composition + agentic coding measurement:** Rapid-MLX is the closest: MLX, PLD
  composed with native MTP, SHA-identical outputs, measured on code copy/edit and tool-XML
  loops on an M3 Ultra; and a separate SuffixDecoding route. It explicitly does not cover
  "continuous multi-request speculation", keeps prompt history request-scoped, and its
  suffix route regresses on Qwen3.6/GPT-OSS. fmsouza/lookup-spec-metal targets the same
  niche (agentic coding traces, MTP composition) but is per-request and unfinished.
- **Conclusion: the niche is open.** No project combines (a) a datastore that outlives the
  process, (b) a global tree/index over prior outputs, (c) MLX on Apple Silicon,
  (d) composition with DFlash/MTP, (e) agentic-coding measurement. The nearest artifacts
  are ArcticInference's suffix tree (has a,b in RAM only, CUDA serving, no Mac) and
  Rapid-MLX (has c,d,e, per-request only). The missing pieces for the closest thing
  (Rapid-MLX or mlx-dspark) are: an index that is fed completed responses across
  requests, serialized to SSD and reloaded, plus a hybrid policy that merges datastore
  drafts with the DFlash/MTP draft instead of running a separate route.
- **Cautions from the evidence:** datastore drafting is workload- and model-sensitive
  (Rapid-MLX 0.77x JSONL, 0.86-0.95x GPT-OSS/Qwen3.6; thc1006 all-negative on an A3B MoE;
  ngram-engine disabled for phrase loops), so a per-request acceptance gate is mandatory;
  the SuffixDecoding ablation showing global > per-request is on AgenticSQL with H100s,
  not on Metal where verify width is capped (~8-32); the marginal gain must be shown on
  top of DFlash (3.6x) rather than against plain decoding.

## Sources
- https://github.com/ggml-org/llama.cpp/pull/5479
- https://github.com/ggml-org/llama.cpp/blob/master/common/ngram-cache.h
- https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/examples/lookup/lookup.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/pull/19164
- https://github.com/ggml-org/llama.cpp/pull/18471
- https://github.com/ggml-org/llama.cpp/pull/19294
- https://github.com/ggml-org/llama.cpp/issues/23184
- https://github.com/ggml-org/llama.cpp/issues/24507
- https://github.com/ggml-org/llama.cpp/pull/19493
- https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
- https://github.com/ikawrakow/ik_llama.cpp/issues/1602
- https://github.com/ikawrakow/ik_llama.cpp/pull/1261
- https://github.com/vllm-project/vllm/pull/25784
- https://docs.vllm.ai/en/latest/api/vllm/v1/spec_decode/suffix_decoding/
- https://github.com/snowflakedb/ArcticInference (csrc/suffix_decoding/, arctic_inference/suffix_decoding/cache.py, docs/suffix-decoding.rst)
- https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/
- https://arxiv.org/abs/2411.04975
- https://github.com/sgl-project/sglang/issues/13165
- https://docs.sglang.io/advanced_features/speculative_decoding.html
- https://github.com/sgl-project/sglang/issues/23005
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/ngram/README.md
- https://github.com/FasterDecoding/REST
- https://arxiv.org/abs/2411.10666
- https://arxiv.org/abs/2511.21699
- https://aclanthology.org/2025.acl-long.338/
- https://arxiv.org/abs/2606.05742
- https://arxiv.org/abs/2605.20104
- https://arxiv.org/abs/2604.14885
- https://arxiv.org/abs/2604.02047
- https://arxiv.org/abs/2605.26289
- https://arxiv.org/abs/2604.26469
- https://github.com/ARahim3/mlx-dspark/blob/main/src/mlx_dspark/lookup.py
- https://github.com/raullenchai/Rapid-MLX (vllm_mlx/speculative/suffix_decoding.py, vllm_mlx/spec_decode/mtp/prompt_lookup.py, docs/suffix_decoding_eligibility.md, evals/results/SUFFIX_POC_REPORT.md, docs/engineering/performance/qwen4-prompt-lookup-qualification.md)
- https://github.com/marzukia/qMLX
- https://github.com/ddalcu/mlx-serve/blob/main/src/pld_index.zig
- https://mlxserve.com/speculative-decoding/
- https://github.com/joongom/mlx-soloheaven
- https://github.com/MidasMulli/ngram-engine
- https://github.com/fmsouza/lookup-spec-metal
- https://github.com/joelvarun/suffixspec
- https://github.com/vukrosic/qwen3-prompt-lookup-robustness
- https://github.com/jundot/omlx
- https://github.com/youssofal/MTPLX
- https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1531
- https://github.com/ollama/ollama/issues/9216
- https://github.com/LostRuins/koboldcpp/discussions/2160
- https://github.com/mybigday/llama.rn/issues/359
- https://www.youtube.com/watch?v=71T4aQiwZ_I
