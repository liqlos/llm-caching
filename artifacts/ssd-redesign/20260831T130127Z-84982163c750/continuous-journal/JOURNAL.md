# Continuous cache research journal

## 2026-08-31T00:00:00+02:00 - Round 000: baseline and decision boundary

Hypothesis: a compact aggregate-prefix representation with an admission policy better than recency-only may recover useful warm-cache coverage that the current append-only per-observation LRU loses under pressure.

Established baseline: exact full-production reuse is valid only at K2/K4; K4 is 0.9192x baseline in the accepted benchmark, and K8/K16 have output/state mismatches. The verified K2 perfect-draft ceiling is about 1.089x, so it cannot satisfy the required 1.20x decode-only improvement even at complete coverage. SSD lookup itself is negligible relative to target generation.

Evidence preserved before this journal: `REVIEW.md`, `status.md`, `cache-policy-audit/result-v3.json` with SHA-256 `b8b9bb9d4e349270fdfc4b6ca9e0e32d746566365b2c4a79c3f9c4a0db169bf1`, and M4 qualification receipts in the parent artifact directory.

Next action: define a bounded native aggregate-prefix plus TinyLFU experiment, then run a warmed, cache-pressure M4 benchmark only if exactness and resource gates can be preserved.

## 2026-08-31T00:05:00+02:00 - Round 001: native representation audit

Hypothesis: replacing append-only duplicate observation entries with compact modal aggregates can materially raise the number of useful exact K2 candidates retained under a fixed RAM budget.

Method: inspected the PCSS speculative provider and store implementation. The provider remains target-verified, so a representation-only change can retain fail-closed exactness if it preserves an exact token-keyed suffix and uses a modal continuation only as a draft.

Finding: the native store appends observations as independent records, reconstructs candidate order from file order, and keeps recency state only in memory. This is sufficient to explain the policy simulator's large LRU-to-TinyLFU gap under pressure, but does not establish an end-to-end gain. The M4 is reachable and no model server was running at the resource gate.

Conclusion: proceed only with an isolated cache-layer prototype plus a warmed-pressure benchmark. Do not alter K2's public correctness envelope or enable K4-plus paths.

Next action: map the smallest store API change that can aggregate equal prefix/continuation candidates and expose admission counters; benchmark baseline, existing LRU, and prototype with identical request traces.

## 2026-08-31T20:59:29Z - Round 002: aggregate implementation boundary

Hypothesis: exact duplicate coalescing is a small cache-layer patch.

Method: traced the native store insertion, index rebuild, lookup, and eviction paths, plus the focused PCSS test surface.

Finding: coalescing cannot safely be bolted onto `put_batch` by comparing only existing in-RAM metadata: the metadata contains prefix hash and token counts, not a continuation identity. Exact comparison would require disk reads on every put or a durable full-key fingerprint/versioned index. A correctness-preserving aggregate design therefore needs a new compact directory record or a versioned sidecar index, and TinyLFU additionally needs an explicit bounded frequency sketch and restart semantics. A naive de-dup patch would trade capacity pressure for unbounded insertion I/O and would not be a valid speed benchmark.

Conclusion: do not edit the native store yet. The minimal valid prototype is a sidecar metadata index keyed by exact prefix-plus-continuation fingerprints, with a bounded admission sketch and a new-store namespace to avoid reinterpreting existing files.

Next action: inspect the existing durable index format and choose whether its schema can carry a sidecar aggregate directory without weakening crash recovery or physical-byte accounting.

## 2026-08-31T21:05:29Z - Round 003: durable-index feasibility

Hypothesis: the current hot index can carry an aggregate directory cheaply.

Method: inspected the entry schema, index parser/serializer, publication protocol, and physical-capacity accounting.

Finding: `index.pcss` is an atomically published snapshot bound to committed record and manifest byte counts. Its 48-byte entry stores only record offsets, prefix hash/count, and `last_used`; there is no continuation fingerprint, frequency state, or generation field for a separate admission structure. The file is included in physical-byte accounting and its parser requires the shared record format version. Extending it in place would couple a metadata-policy experiment to record-format migration and recovery compatibility. A separate sidecar would also require new secure-file and generation recovery handling, and it cannot reclaim physical capacity until compaction removes duplicate records.

Conclusion: an aggregate TinyLFU experiment is an on-disk format/recovery project, not a small cache-policy patch. It remains a legitimate single-digit path, but it cannot be responsibly rushed into the currently qualified store.

Next action: pivot to a no-format-change experiment: quantify whether changing candidate selection among existing exact records can improve K2 acceptance under pressure, using the existing trace/simulator before touching native persistence.

## 2026-08-31T21:10:59Z - Round 004: candidate-selection counterfactual

Hypothesis: selecting the most recent existing exact candidate instead of a modal candidate can materially improve legal K2 proposal quality without changing persistence.

Method: ran chronological dense-posting replay for `dense_recent`, `dense_modal`, and `dense_confident` across E3/E4/E5, both sequence modes, and K2/K4. Receipt: `candidate-selection-v1.json`, SHA-256 `3593b62f4290debc65ed8f2f8365b83e077f06d7394c90026883c21b19e0d340`; runner: `run_candidate_selection.py`.

Measurements at K2 prompt-plus-completion: E3 recent/modal both captured 68 of 288 output tokens (23.61%); E4 modal was 561/1152 (48.70%) versus recent 558/1152 (48.44%); E5 recent was 490/576 (85.07%) versus modal 486/576 (84.38%). Confidence gating improved precision but reduced E3/E4/E5 captured tokens to 67/486/465 respectively. Candidate-hit counts did not reveal a material selection advantage.

Conclusion: candidate ordering is not the missing 20% mechanism. The native oldest-first tie behavior merits correction only as quality hygiene, not as a performance project, and no M4 build is justified by these small trace-only deltas.

Next action: examine the residual K2 ceiling under actual server overhead and identify whether request batching, prompt reuse, or target-side state reuse could change the end-to-end denominator without weakening exactness.

## 2026-08-31T21:17:59Z - Round 005: target-side state and prompt reuse audit

Hypothesis: an unused exact target-side state path can remove sufficient end-to-end work to rescue PCSS K2.

Method: traced llama-server prompt-cache, idle-slot state, save/restore, and decode scheduling paths; opened two bounded read-only reviews for independent cross-checks.

Finding: the server already enables prompt caching by default and reuses the common prompt prefix; idle slots can be retained in the prompt cache, and slot state supports target and draft state serialization. These paths reduce prompt evaluation or reload cost, not the serial target decode that dominates the PCSS benchmark. The server documentation additionally states that prompt caching may alter bit-level determinism with different batch sizes, so it cannot silently substitute for the exact-output qualification gate.

Conclusion: prompt/KV reuse is complementary latency optimization already present, not a route to a 20% decode-speed gain. Persisting target slot state would only help exact repeated contexts and would need its own identity, ABI, storage, and output-equivalence qualification.

Next action: collect the two independent findings, then choose between a narrowly bounded warmed-server overhead benchmark and a state-cache architecture proposal; do not conflate either with PCSS decode acceleration.

## 2026-08-31T21:17:59Z - Round 006: independent state-path confirmation

Hypothesis: server state reuse has an unexploited target-decode effect when composed with PCSS.

Method: independent read-only review of server state paths, with no model process, build, or remote workload started.

Finding: the reviewer confirmed exact server prompt-state cache (`server_prompt_cache`, `prompt_save`, `prompt_load`), common-prefix KV reuse, and non-prefix chunk reuse. PCSS begins only after prompt evaluation, and speculative rollback checkpoints are correctness machinery rather than cross-request acceleration. Target state serialization is already used through host-side prompt-cache buffers; its on-device API does not itself reduce decode work.

Conclusion: no hidden target-decode reuse path was found. End-to-end long-prompt latency may benefit from existing prompt reuse, but it is a separate claim from the required PCSS decode gain.

Next action: recover a precise existing M4 K2 harness inventory before launching any server-overhead benchmark; the second bounded review was stopped before completion to keep the agent/resource cap conservative.

## 2026-08-31T21:25:29Z - Round 007: M4 harness inventory

Hypothesis: the existing M4 live-benchmark harness can measure warmed server overhead not covered by the decode qualification.

Method: recovered the remote harness location and inspected its controlled live-decode contract without launching a server or model process.

Finding: `src/pcss/llamacpp_live_benchmark.py` is deliberately a best-case decode benchmark. Its stated scope excludes provider lookup, prompt prefill, and state restore; it requires exact output and final state, matching host/model/binary/runtime manifest bindings, and blocks K2 state checkpointing. It therefore cannot answer server-overhead or warmed prompt-cache questions. The prior K2 decode ceiling and K4 qualification remain the applicable evidence for decode claims.

Conclusion: no existing valid warmed-server harness was found in the recovered inventory. Running the controlled bridge again would duplicate an already-bounded decode experiment and would not test a new hypothesis.

Next action: design a separate end-to-end benchmark contract for exact repeated prompts with cache-prompt on/off and PCSS K2 on/off; it must report prompt and decode phases separately and must not be used to inflate a decode-only claim.

## 2026-08-31T21:31:29Z - Round 008: warmed-server benchmark contract

Hypothesis: a four-arm end-to-end experiment can separately quantify existing prompt reuse and PCSS K2 without contaminating the decode claim.

Method: recovered the existing harness boundary, checked the remote checkout for an existing server runner, and wrote `WARMED_SERVER_BENCHMARK_CONTRACT.md`.

Finding: no reusable remote server benchmark runner or pinned server runtime configuration is present in the checkout. The new contract defines A/B/C/D arms for cache-prompt off/on crossed with PCSS off/K2, serial execution, fixed-order randomization, output/state/resource receipts, and phase-specific attribution.

Conclusion: this is the smallest valid next M4 experiment, but it needs a fresh runner with explicit model and server bindings. It can establish end-to-end prompt-cache value; it cannot make the existing PCSS K2 decode ceiling disappear.

Next action: inspect the prior production qualification receipt for the exact target model/server bindings and adapt a fresh runner to those immutable inputs rather than guessing them.

## 2026-08-31T21:37:29Z - Round 009: warmed-server runner staged, execution held

Hypothesis: the fixed Q4_K_M target and six-request production workload can support the four-arm warmed-server contract without changing the target binding.

Method: recovered the exact M4 binary, model, PCSS K2 flags, fingerprint, and workload shape from the accepted K4 receipt; staged `run_warmed_server.py` in the journal artifact.

Finding: before execution, the runner was found to hash the model using `Path.read_bytes()`. That would load the entire GGUF into process memory and violates the explicit low-memory safety boundary.

Conclusion: no server was started and no M4 benchmark was run. The runner must be corrected to stream the model hash before it may execute.

Next action: await approval to correct this newly introduced runner defect, then execute the bounded four-arm study with one server at a time.

## 2026-08-31T21:37:29Z - Round 010: four-arm warmed-server result

Hypothesis: existing prompt reuse and PCSS K2 may compose into a meaningful end-to-end acceleration on the fixed M4 target.

Method: fixed the runner to stream file hashes and bound every request to 64 generated tokens. A two-measurement pilot passed exact output and cleanup. Then ran the fixed six-request workload with two warmups and eight measurements per arm, one server at a time.

Receipt: `warmed-server-full-v1-result.json`, SHA-256 `01f14f16f9431db3feede1d4d0104425f4c817e0749c4e56aad574284f28a569`. Every measurement output in B/C/D exactly matched A. Peak RSS ranged from 5,225,072 to 5,230,160 KiB; no model server remained after the run.

Measurements: A cold baseline wall time was 74.358 s, B warm baseline 70.006 s, C cold PCSS K2 70.880 s, D warm PCSS K2 68.672 s. Relative to A, B improved wall time 5.85%, C 4.89%, and D 7.65%. Predicted-token phase sums were A 67.795 s, B 68.704 s, C 64.458 s, D 67.370 s. Thus PCSS K2 improved cold predicted-token time 5.18%, while the warm-path gain is primarily prompt evaluation: prompt phase was 6.396 s in A and 1.141 s in B.

Conclusion: the four-arm end-to-end result is output-verified and useful, but it remains below the 20% requirement. It must not be reported as a state-verified decode gain because this runner version did not yet record target state.

## 2026-08-31T21:37:29Z - Round 011: slot-state gate pilot

Hypothesis: the output-verified warmed path also preserves serial-canonical target slot state.

Method: fixed the runner to save and SHA-256 the server slot state after each response, then ran the same two-measurement four-arm pilot with the same target bindings.

Receipt: `warmed-server-state-pilot-v1-result.json`, SHA-256 `a200d10439693ee100ff67b68fcbdd37d9a17b563c16ceb5d3e34507b74d7a9d`. Output equality passed for all arms. Slot-state equality to cold baseline passed for A and C, but failed for B and D, the two `cache_prompt=on` arms. All server processes were cleaned up.

Conclusion: cache-prompt provides a real output-verified end-to-end reduction, but it changes target slot state on this backend. It cannot be used in a serial-state-qualified decode claim, and a full state-aware repetition would only replicate the pilot failure.

Next action: investigate serial-canonical target integration, where accepted multi-token candidates are reconstructed token-by-token before continuing generation; this is the only remaining route that can potentially make wider accepted spans exact.

## 2026-08-31T21:37:29Z - Round 012: serial-canonical reconstruction decision

Hypothesis: a PCSS-specific serial target commit can make K8/K16 exact while retaining enough wide verification economy for the 20% goal.

Method: two independent read-only reviews traced the server acceptance path and derived the target-compute lower bound from the existing M4 receipts.

Finding: the minimal correctness patch is in `server-context.cpp` plus a PCSS query in `speculative.cpp`/`speculative.h`; it would feed each accepted PCSS token through a separate target decode, instead of adding the wide draft to one target batch. This avoids rollback because unverified draft tokens never enter target KV. However, canonical reconstruction requires at least one serial target forward per output token, the same work as baseline, while wide verification and lookup remain additional work. Its strict decode-only upper bound is therefore below 1.00x. Existing K4 measured 0.9192x; K8/K16 only gained economics through the non-canonical wide state that this patch removes.

Conclusion: do not implement or benchmark the serial reconstruction patch for the 20% objective. It is a valid correctness separation tool but a provable performance dead end on this architecture.

Next action: move to reuse objects that can actually replace target compute: strict whole-response memoization or exact prefix/KV-state reuse on long repeated contexts. Evaluate their hit-rate and identity-key requirements before any runtime implementation.

## 2026-08-31T21:37:29Z - Round 013: strict whole-response memoization pilot

Hypothesis: a disk-backed cache keyed by the entire canonical request plus immutable target binding can remove target compute for exact repeats and exceed 20% after warmup.

Method: created `run_response_memo.py`. It keys endpoint and normalized request body on the fixed binary/model/flags/temperature/seed/output-cap binding, stores only a completed response, and returns it only on exact key equality. On M4 it ran the frozen six-request workload twice: baseline always invoked the target; memo wrote three unique responses then replayed exact keys.

Receipt: `response-memo-pilot-v1-result.json`, SHA-256 `7beb91009ab17614eede260acf1e8fab7d017e79807ef9d4cfa0774f0b07e91a`. Exact text equality against the baseline passed. Baseline wall time was 17.218 s. Response memo wall time was 4.598 s, including 4.597 s for three misses and 1.212 ms total for nine SSD hits: a 73.3% reduction for this intentionally repeat-heavy workload. Server cleanup passed.

Conclusion: this is the first architecture that clears the hard performance threshold, because it replaces target compute rather than verifying it. It is not a PCSS token-decode acceleration claim and requires a complete response-contract/identity review before product integration.

Next action: validate the response memo key and response semantics against server parameters, stop conditions, streaming, tool calls, model/runtime changes, and authorization boundaries; then choose a constrained production integration surface.

## 2026-09-01T00:00:00+02:00 - Round 014: response memo safety boundary

Finding: the pilot is not product-safe yet. It checks only completion text, keys only path strings plus request data, and has no authorization scope. It must not be integrated into PCSS or generic llama-server handling.

Conclusion: the only viable production surface is a separate post-authorization local gateway for deterministic non-streaming `POST /v1/completions`, with an explicit allowlist and fail-closed bypass for every unsupported field or request mode. Key material must include opaque authorization scope/policy version, canonical request, and content fingerprints for target model, tokenizer/template, adapters, binary, and deterministic generation profile. Full response protocol equality, at-rest privacy, concurrency, and independent natural-recurrence qualification remain required.

Next action: specify and implement only the gateway's deterministic eligibility/key/record contract before any upstream forwarding integration.

## 2026-09-01T00:05:00+02:00 - Round 015: fail-closed response gateway contract

Method: wrote `RESPONSE_MEMO_GATEWAY_CONTRACT.md`.

Conclusion: the initial surface is intentionally only authenticated non-streaming deterministic completions with an exact request-field allowlist. The cache key is HMAC-scoped by authorization domain, policy, endpoint, canonical request, and target execution fingerprints. Any unsupported request, identity mismatch, corruption, expiry, or dynamic response semantics bypasses.

Next action: locate the repository's actual authorization/upstream boundary; only then can this contract be implemented without inventing a security model.

## 2026-09-01T00:10:00+02:00 - Round 016: integration boundary audit

Finding: the only local forwarding boundary is `project_capture_proxy.py`. It is a loopback workload-capture tool that deliberately records request bodies and has no authenticated principal or authorization policy domain. `scripts/run_server.sh` launches llama-server directly.

Conclusion: no product authorization/upstream boundary exists in this repository. Integrating response memoization here would invent tenant semantics and create a privacy regression. Keep the 73.3% result as a bounded architecture result; do not merge it into the capture proxy or PCSS store.

Next action: continue measuring recurrence on independent naturalistic workloads and investigate exact KV-prefix reuse where the existing server cache can be safely controlled.

## 2026-09-01T00:15:00+02:00 - Round 017: naturalistic response-recurrence audit

Measurement: exact full-prompt duplicate counts were E3 0/6, E4 0/24, and E5 0/12. Only synthetic E1 and cross-repo E2 contain the constructed repeat pairs.

Conclusion: strict response memoization clears 20% only for genuinely repeat-heavy traffic. It is not an acceleration lane for the available naturalistic workloads, so retain it as a conditional architecture result and return attention to exact prefix/KV reuse.

## 2026-09-01T00:20:00+02:00 - Round 018: naturalistic prefix recurrence

Measurement: `prefix_recurrence.py` found E3 max/mean shared prefixes of 10/5.0 tokens, E4 11/5.33, and E5 75/7.83.

Conclusion: existing exact prompt/KV reuse has no realistic path to 20% end-to-end acceleration on these traces. All known cache objects have now been tested or bounded: token PCSS, cache policy, response memo, and prompt/KV reuse.

Next action: preserve the evidence and wait for a new workload with demonstrably long repeated prefixes or full request recurrence; no further same-architecture tuning is justified.

## 2026-09-01T00:30:00+02:00 - Round 019: long exact-prefix cache pilot

Hypothesis: prompt/KV caching can deliver a material cache-specific gain when the workload has sufficiently long exact repeated prefixes, independently of MLX or PCSS.

Method: fixed the warmed-server runner to translate workload `max_tokens` into llama-server `n_predict` with a 64-token cap. The first 700-repeat prompt exceeded the fixed 1024-token context and failed closed with HTTP 400; the preserved v2 attempt was stopped. The v3 workload uses six identical approximately 601-token prompts, two measurements per arm, one server at a time.

Receipt: `long-prefix-cache-pilot-v3-result.json`, SHA-256 `94d2dd19590e607eed6c59a656dabd3d7ec8cf0a6e17caa6810d027ea03d4983`. All B/C/D responses exactly matched A.

Measurements: cold baseline wall time was 37.006 s; warm baseline was 19.077 s, a 48.45% reduction. Cold PCSS K2 was 36.085 s. Warm PCSS K2 was 17.433 s, a 52.89% reduction from cold baseline and 8.62% from warm baseline. Prompt phase fell from 20.013 s to 1.959 s, while predicted-token phase stayed about 17 s in baseline arms and 15.46 s with PCSS K2.

Conclusion: cache-specific acceleration above 20% is real for long exact-prefix recurrence; it comes primarily from existing prompt/KV cache, with a smaller K2 PCSS composition effect. This is an exact-output end-to-end workload result, not a serial-state-qualified decode-only claim; state snapshots remain a separate gate.

Next action: run partial-prefix and cache-pressure/eviction matrix to determine the recurrence threshold and cache-RAM policy needed to preserve this gain.

## 2026-09-01T00:40:00+02:00 - Round 020: half-prefix cache pilot

Hypothesis: prompt/KV cache remains above the 20% end-to-end threshold with only half of a long prompt shared.

Method: six approximately 600-token completions shared a 300-token prefix and used unique suffixes. The same four-arm server matrix and 64-token cap as Round 019 was used.

Receipt: `half-prefix-cache-pilot-v1-result.json`, SHA-256 `484724868c93765532d6f8aae8aa5bcebb00061b4c46b7b1df5224f5729df876`. Exact output passed for all arms.

Measurements: cold baseline was 33.667 s. Warm prompt/KV cache was 25.250 s, a 25.0% reduction. Warm PCSS K2 was 24.318 s, a 27.8% reduction. Prompt phase fell from 16.716 s to 8.244 s; predicted-token time remained about 17 seconds, with K2 reducing it to 15.951 seconds.

Conclusion: cache-specific gain remains above 20% with a 50% shared long prefix. The remaining question is cache pressure and eviction, not basic viability.

Next action: add cache-RAM controls to the warm-server runner and run an over-capacity cyclic-prefix workload to measure eviction behavior.

## 2026-09-01T00:50:00+02:00 - Round 021: multi-prefix cache capacity threshold

Method: added `--cache-ram-mib` to the warm-server runner and ran a cyclic workload of four distinct approximately 600-token prompts followed by the same four prompts again. Each arm ran two measurements and produced exact outputs.

Results: 1 MiB, 16 MiB, 128 MiB, and 512 MiB did not retain useful prompt states: warm arms were at or slower than cold, with prompt phase about 45 seconds. At 1024 MiB, cold wall time was 53.631 s and warm wall time was 20.048 s, a 62.62% cache-specific reduction. Warm plus PCSS K2 was 19.967 s, a 62.77% total reduction. Prompt phase fell from 45.381 s to 11.869 s.

Conclusion: the cache is capacity-bound, not search-bound, for alternating full target-state entries. Four Qwen3-8B states of this shape require more than 512 MiB; 1 GiB restores multi-prefix reuse and exceeds the prior 48.45% exact-repeat cache result.

Next action: complete this cache-specific optimisation increment. Future work should evaluate compressed/segmented state records before attempting more eviction-policy changes.

## 2026-09-01T14:50:00+02:00 - Round 022: adaptive sequence datastore grid correction

Method: ran the preregistered 41-configuration variable-suffix matrix across E0-E5. The first preserved result exposed two study defects: persistent-unique coverage counted overlapping all-position proposals against an output-token denominator, and the storage gate assumed every key/posting remained RAM-resident at 200 GiB. The study now reports conservative non-overlapping online reuse separately, models a bounded 64 MiB disk-index page cache, and fails closed when verifier model fingerprints differ from measured traces. Targeted adaptive/replay tests pass, 48 total.

Receipts: `adaptive-grid-pass-v1.json` preserves the invalid diagnostic; `adaptive-grid-pass-v2.json` is the corrected no-profile grid; `adaptive-grid-profile-v1.json` applies the compatible M4 Qwen3-8B verifier profile. Target future tokens are not used for retrieval.

Finding: E1 and E2 positive controls pass. With speed-first preregistered ranking restored, the selected policy is stride-16 confident prompt-plus-completion at K16, not the denser maximum-coverage policy. The bounded on-disk index estimate remains below 64 MiB RAM by construction while keys, postings, and continuations reside on SSD.

## 2026-09-01T15:05:00+02:00 - Round 023: final sequence datastore decision

Method: ran a 2,000-replicate clustered shortlist and a 10,000-replicate final pass with the M4 Pro/Qwen3-8B Q4_K_M verifier receipt. Bootstrap now resamples whole session cost components and recomputes total baseline/cache cost ratio, so point estimates and intervals use the same estimand.

Receipt: `adaptive-final-pass-v1.json`. Selected stride-16 confident prompt-plus-completion K16 results at a 10 microsecond lookup budget were E1 1.238x with LCB95 1.039, E2 1.417x with LCB95 1.210, E3 1.003x with LCB95 0.959, E4 1.051x with LCB95 0.982, and E5 0.863x with LCB95 0.736. Persistent-unique online coverage was 2.78%, 15.54%, and 27.26% for E3, E4, and E5 respectively. Zero-accept proposal rates were 50.0%, 43.75%, and 50.0%.

Conclusion: `ALGORITHM_DECISION = NO_GO_CURRENT_REDESIGN`. The exact variable-suffix linear sequence cache works on synthetic controls but does not provide qualified economics on non-control workloads. The multibranch diagnostic adds only 0-2.8% over the comparable linear all-position result, far below the required 1.5x signal multiplier. Native SSD draft-store implementation is therefore gated off.

## 2026-09-01T17:00:00+02:00 - Round 024: final receipt correction and bounded disposition

Method: reconciled the final v19 qualification receipt and v16 capacity receipt against the reproducible implementation and patch identities. The final evidence set reports 349 tests PASS, native PASS, and successful process cleanup. Implementation hash is `782e518e85d7cbdc8cd844469114715be573e59856dd50de8b11d4c322d2c0ef`; reproducible patch SHA-256 is `7f4b5ea6871ade51060e45314bca9e3da3671a1fff8d6fdf4f79e6de68d1739c`.

Receipts: final v19 SHA-256 `88754009441449411316c1aa23d076a9947bd16914d02116085f1f922e193862`; final v16 SHA-256 `b073e6571c5434efeabc07ec1cd05f610faab1e8478164cc460c5d293e34e950`. The caller-attested synthetic qualification reports exact reuse at 4.9897x / 79.9588%, partial reuse at 1.5825x / 36.8093%, and novel-input ratio 0.99948, approximately 0.052% overhead. Asynchronous learning overhead is approximately 0.62%. Capacity qualification passes 3/4 at 512 MiB and 4/4 at 1024 MiB.

Evidence boundary: these exact, partial, novel, learning-overhead, and capacity results are caller-attested synthetic claims. They establish the bounded native SSD L2 behavior represented by the final receipts; they do not establish representative owner-workload economics, caller-independent compatibility identity, or a production-wide speedup.

Conclusion: the independent sequence-datastore decision remains `ALGORITHM_DECISION = NO_GO_CURRENT_REDESIGN`. The positive caller-attested synthetic native SSD L2 receipts do not reverse or supersede that decision. Historical v14/v15 receipts remain provenance only; v19/v16 and the identities above are the current final evidence.

## 2026-09-01T15:10:00+02:00 - Round 024: SSD L2 prompt-state pivot

Hypothesis: the only cache mechanism with measured greater-than-20% end-to-end benefit is exact or partial prefix target-state reuse, but its current full-state RAM representation is capacity-bound.

Method: traced `server_prompt_cache`, `prompt_save`, `prompt_load`, and slot selection in the pinned llama-server. Cache entries serialize complete target and optional draft sequence state into host RAM, use common-prefix selection, consume the selected entry on restore, and evict oldest entries under the RAM cap. The source itself marks moving purged slots to a level-2 cache as unfinished work.

Conclusion: proceed with a crash-safe SSD-backed level-2 prompt/KV state cache using bounded RAM metadata and exact token-prefix verification. Keep sequence drafting disabled as the primary speed claim; measure SSD restore, restart, partial-prefix reuse, capacity pressure, and no-hit overhead separately from backend effects.

## 2026-09-01T16:00:00+02:00 - Round 025: native SSD L2 implementation and tiny-model gate

Method: added a server-only append-only prompt-state store with manifest-last commits, exact token verification, bounded metadata, restart tail truncation, quota eviction, streaming compaction, previous-generation recovery, caller-attested compatibility namespaces, and RAM-to-SSD demotion. Added `--cache-ssd`, `--cache-ssd-path`, and `--cache-ssd-key`. Native store and parser targets built in isolated clean trees.

Finding: a local tiny-model A/B/A plus restart smoke produced `cache_n=153` on same-process and restart hits, `cache_n=0` under an incompatible key, and equal output. The first Release test build was not accepted because standard assertions were compiled out; tests were changed to unconditional failure checks and rerun.

## 2026-09-01T16:20:00+02:00 - Round 026: first M4 SSD L2 pilots

Method: built the patch over clean pinned llama.cpp `9723942adc518b43c4b95dc4dce6906903eb5e09` with Metal and Qwen3-8B Q4_K_M. Baseline and SSD used the same binary, model, context, batch, thread, sampler, and output settings. Model startup was excluded from request wall and recorded separately.

Finding: the first exact restart pilot reduced wall time from 3.764 s to 0.737 s with 1000 cached prompt tokens. Its first partial comparison was invalid because the baseline inherited 503 active-slot tokens. The corrected fresh-process pilot measured exact 3.766 s to 0.726 s, an 80.72% reduction; 50% partial 3.745 s to 2.362 s, a 36.92% reduction; novel 3.763 s to 3.765 s, 0.05% overhead. All decoded outputs matched.

Conclusion: SSD state restore preserves the proven greater-than-20% prefix-cache effect across process restarts. Full-state partial-prefix restoration already exceeds 20%; ordinary Qwen KV does not create the rollback checkpoints exposed by `--ctx-checkpoints`, so the attempted segmented flag pilot did not change the stored record or result.

## 2026-09-01T16:45:00+02:00 - Round 027: paired qualification and capacity pressure

Method: ran six paired fresh-process iterations with alternating arm order and 10,000 iteration-cluster bootstrap resamples. Ran four-prefix 512 MiB and 1024 MiB quota arms with restart after priming.

Finding: the pre-final paired run measured exact 5.014x and partial 1.584x; novel remained at parity. At 512 MiB, three of four approximately 150 MiB states survived, with the recently reused A retained and B evicted; physical bytes were 452,594,440 below quota. At 1024 MiB, all four survived and physical bytes were 603,459,296. Output equality passed after compaction and restart.

## 2026-09-01T17:00:00+02:00 - Round 028: asynchronous learning and preserved failure

Finding: synchronous 150 MiB demotion added 235 ms to the following request. A single background writer reduced measured learning-request overhead to 0.71% while clean shutdown still waited for durable completion. The first asynchronous 512 MiB pressure run produced a preserved `Bus error` when capacity compaction overlapped GPU inference.

Fix: asynchronous writes are now permitted only when the record fits without eviction or compaction. Capacity-pressure writes execute synchronously before inference. The exact failed A/B/C/D/A scenario was rerun: no crash, exact output, deterministic `[hit, miss, hit, hit]`, and 452,594,440 physical bytes.

## 2026-09-01T17:30:00+02:00 - Round 029: final M4 result and review boundary

Receipt: `ssd-l2-qualification-v14-token-final/result.json`, SHA-256 `8c7cb9e6ebf2d7ea5ddde58002a76a8becb70630bb576a2881bb43850807512c`. It is bound to model SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, server implementation SHA-256 `849b0444029aac41d348321d5316eb985a5466df0c4417caf137e63c51106353`, and compatibility key `sha256:ba64b8d1abd350583a9f28394102d79f344765b9fea10f057d0a3965069beeb4`.

Results: exact baseline/SSD was 5.004x, an 80.02% latency reduction with LCB95 4.974x. Fifty-percent partial was 1.586x, a 36.94% reduction with LCB95 1.584x. Novel baseline/SSD ratio was 0.9981 with interval 0.9959-1.0005, corresponding to 0.19% point overhead. Every baseline and SSD response returned 23 generated token IDs; all IDs and output text matched in all 18 pairs. Learning-request wall was 3.792 s versus a 3.768 s novel baseline mean, about 0.64% overhead.

Pressure receipts: 512 MiB `ae5e8e4e0b3a6e059e5e7a103915cbcae0c93fc8f7dfaffc4ff25db7b711fecd`; 1024 MiB `fd130c7c57c168458cc9fe79ac8a07c3312e7119020c42b860d9cf5a050cdb29`. Reproducible patch SHA-256: `c1c76e3155b1ca911e1a9726a292291ad83937ebc7113faf122f6566932d70b7`, fresh apply-check PASS.

Conclusion: SSD L2 prompt/KV caching is a qualified cache-specific win for this synthetic long-prefix M4/Qwen workload, including restart and 50% partial prefixes. It is not a representative owner-workload, Windows durability, caller-independent identity, or universal decoding claim. The future-token-free sequence datastore remains `NO_GO_CURRENT_REDESIGN`; its corrected final receipt is `adaptive-final-pass-v2.json`, SHA-256 `5c22b3ca5d4c505a45338402d1ae80ed2efe6c72e8df686fdaebc02cfe70de6e`.

## 2026-09-01 - Reconciliation: archive-only claim boundaries

Canonical authority: `docs/SSD_L2_CACHE.md` is the sole current-state authority for SSD L2 behavior, qualification claims, receipts, and limitations. This journal remains append-only experimental history; earlier v14 labels and conclusions are historical rather than current qualification authority.

Reconciled boundaries: the only learning measurement claimed is `learning_request_wall_s=3.791445`; no clean learning-overhead component was identified. The novel scenario supports only no detected latency difference, not zero or negligible latency impact. The pressure receipt supports bounded capacity, eviction, and restart behavior for its probe, but does not establish baseline/SSD equality of generated token IDs or output under pressure. The caller-attested synthetic M4 boundary remains in force, and the sequence datastore remains `NO_GO_CURRENT_REDESIGN`.

## 2026-09-01 - Final handoff reconciliation

Any earlier journal overclaim about novel overhead, learning-write overhead, or pressure output/token equality is superseded by the canonical boundary in `docs/SSD_L2_CACHE.md`. `NOW.md` is the only ordered next-plan handoff; historical PCSS and sequence entries do not authorize continuation.
