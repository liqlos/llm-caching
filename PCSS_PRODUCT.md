# PCSS native product path

> **HISTORICAL / ARCHIVE-ONLY:** This entire file records evidence and decisions
> from the 20260831 PCSS sequence line. Every product path, command, `current`,
> `immutable`, `final`, default, and manifest requirement below is bounded to
> that historical line and is never a current instruction or next action. Use
> `docs/SSD_L2_CACHE.md` as the sole current-state authority and `NOW.md` for the
> ordered continuation boundary.

For the 20260831 sequence line, the product executable was the pinned
`third_party/llama.cpp` `llama-server`, not `native/pcss_llama_bridge`. The
bridge was an isolated verifier oracle.

Build and run model-independent native checks:

```sh
bash scripts/native-check.sh
```

Launch the OpenAI-compatible server with a local GGUF model:

```sh
export PCSS_MODEL=/absolute/path/to/model.gguf
bash scripts/run_server.sh
```

The historical server listened on port 8080 by default and used a private store at
`.pcss-store`. Set `PCSS_STORE`, `PCSS_NAMESPACE`, `PCSS_MAX_DISK_GIB`,
`PCSS_RAM_MIB`, `PCSS_FINGERPRINT`, or `PCSS_PORT` to override them. The run
script verifies the local model hash and binds the store to that artifact plus
the declared template/thinking policy. The namespace was required and defaulted
to the then-current local OS uid; a server instance was a single-principal boundary.
Passing extra server options requires an explicit `PCSS_FINGERPRINT`. A missing,
unreadable, or hash-mismatched model fails before startup; no downloader or
system llama.cpp installation is used.

Historical terminal status as of 2026-08-31: **IMPLEMENTED_NOT_QUALIFIED**. The native
store/provider compiled; the then-current Python suite had 313 passing tests; and
`bash scripts/native-check.sh` passes the Release store/provider test, store
tool, and `llama-server` help contract against maintained patch SHA-256
`9c7de25f952acab80197bb1581625d3ce696eb94cc9bd2208c35ed6127f28b6b`.
Real-model API equality, cancellation, restart persistence, live metrics,
concurrency, bounded context, failure fallback, and same-slot long-to-short
equality are recorded locally. Representative comparative qualification is
was still absent, so persistent PCSS was experimental and disabled by default
for that sequence line.

## Architecture and numeric bounds

PCSS is a first-class speculative implementation in the pinned llama.cpp
orchestration. `begin`, target-committed token observation, speculative
feedback, successful end, cancellation/abort, and slot reuse have distinct
state transitions. Only a successful request writes learning records; an abort
discards its observed tail. Each request writes at most 32 exact windows, with
at most 64 prefix tokens and eight continuation tokens per window.

The append-only store keeps token payloads on SSD and publishes a versioned,
checksummed hot-index checkpoint bound to a committed record/manifest prefix.
Successful requests append only a bounded durable tail; the full index is
checkpointed periodically rather than rewritten per request. Startup reads the
final manifest row and checkpoint metadata, then scans only the bounded tail;
missing/corrupt indexes rebuild only inside explicit scan limits. Compaction
publishes records/index/manifest as a crash-recoverable triple with manifest
last and retains the previous triple until publication is durable. Peak
directory usage, including temporary generations, is checked against the disk
cap before writing. A
sparse 1 GiB scale test opens below a 1 KiB startup budget with less than 4 KiB
read, then verifies the authoritative record checksum on exact lookup. RAM
contains only compact metadata; lookup is a sorted hash/length probe followed
by exact collision verification from disk. Native
server defaults bound lookup to 256 probes and 256 KiB read, PCSS drafts to
two tokens (`--pcss-max-draft-tokens` permits an explicitly requested 1--2),
RAM to `--pcss-ram-mib`, and disk to
`--pcss-max-disk-gib`. Physical quota includes records, manifests, and index.
One process owns the writable lock; read-only consumers may coexist. Store
directories/files are restricted to 0700/0600, and insecure read-only paths
fail closed. Successful requests publish at most 32 records in one durable
batch. Capacity eviction is deterministic and compacts only live records.
Store I/O, probes, RAM, physical bytes, corruption, incompatibility,
compaction, fallback, lifecycle suppression, source hits, complete provider
latency, current-context latency, and bounded fallback comparison counts are
exposed on `/metrics`.

Operations:

```sh
PCSS_STORE=/absolute/store bash scripts/store-inspect.sh
PCSS_MODEL=/absolute/model.gguf PCSS_STORE=/absolute/store bash scripts/store-compact.sh
PCSS_STORE=/absolute/store PCSS_CONFIRM_RESET=move-store-to-backup bash scripts/store-reset.sh
PCSS_SERVER=/absolute/llama-server PCSS_MODEL=/absolute/model.gguf \
  PCSS_MODEL_SHA256=<sha256> PCSS_PATCH_SHA256=<sha256> \
  PCSS_RESULT=/absolute/fresh-result.json bash scripts/real-model-check.sh
PCSS_PYTHON=/absolute/venv/bin/python \
  bash scripts/seal-project-workload.sh CAPTURE.jsonl WORKLOAD.json RECEIPT.json \
  [CURRENT-SOURCE-SIDECAR.jsonl]
```

Start a capture window with `capture-project-proxy.sh CAPTURE.jsonl`, then call
its loopback `/pcss/capture/begin-episode` endpoint at each genuine episode
boundary. It fsyncs a hash-chained visible request event before forwarding to
loopback upstream. `seal-project-workload.sh` accepts only those proxy
ledgers and rejects answers, candidates, expected outputs, hidden/internal
fields, reordered events, broken hash chains, and incomplete episode coverage
before creating immutable workload and capture receipts. See
`qualification/PROJECT_NATURALISTIC_CAPTURE.md` for the required protocol.
For an explicitly current-source-bearing ordinary capture, pass the proxy a new
`--current-source-sidecar SIDECAR.jsonl` and pass that sidecar as the optional
fourth `seal-project-workload.sh` argument. The ledger and public receipt retain
only commitments; the sidecar and resulting private workload are `0600`.

## Real-model smoke evidence

On 2026-08-30, the official `Qwen/Qwen3-8B-GGUF` Q4_K_M file was downloaded
locally and verified as SHA-256
`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`.
The patched server was launched once with `--spec-type pcss-cache` and once
without speculation, with the same model, `--ctx-size 1024`, four CPU threads,
and greedy sampling. A standard `/v1/completions` request with the repeated
visible prompt `alpha beta gamma alpha beta gamma alpha beta gamma alpha beta`
returned the identical 16-token completion in both arms. The PCSS response
reported `draft_n: 28` and `draft_n_accepted: 10`; its store wrote a 245-byte
`records.pcss` and a 32-byte `manifest.pcss`. Restarting the PCSS server with
the same store produced the same deterministic response and draft counters.

The 20260831 sequence line retained the immutable local receipt
`artifacts/product-e2e/native-lifecycle-ba62a540/receipt.json` (ignored by Git,
receipt SHA-256
`64456c1964b627b1ec044e04050d3d18f67e7f4b138f4d26f35ac855c6540078`).
It records identical deterministic output, persistent hits after learning and
restart, cancelled-tail learning suppression, a usable subsequent request,
same-slot long-to-short equality against a fresh baseline server, bounded store
inspection, and live metrics. This remains correctness smoke only: it is not a
representative-workload speedup or default-enable decision.

### Slot-reuse regression

The first real `--parallel 1` long-request test exposed a provider bug: an
uncapped current-context draft exceeded llama.cpp's output reservation and
triggered `GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)`. The provider
now caps every returned draft at its declared eight-token maximum. A clean-base
rebuild then completed a 96-token request A followed by an eight-token request
B in the same server slot. B's deterministic completion matched the same B on
a fresh speculation-disabled `--parallel 1` server.

### Store failure fallback

With `--pcss-store` pointed at a file instead of a directory, the native server
logged `pcss-cache disabled: cannot create PCSS store directory`, completed
model loading, and served a normal `/v1/completions` response. Store-open
failure therefore disables PCSS rather than target generation.

### Client cancellation smoke

Against a single-slot PCSS server, a streaming 256-token completion was cut off
by the client after one second (3,752 response bytes received). A subsequent
ordinary completion on the same server succeeded. This confirms that a client
disconnect does not leave the tested server slot unusable; it is not a complete
coverage claim for every transport cancellation path.

### Repeating-sequence capability control

The first frozen study (`synthetic-control-v1-ba62a540`) is preserved as an
invalid infrastructure receipt: every arm exited before readiness because the
harness changed its working directory while passing a relative model path. No
request was measured. The causal path-resolution defect was fixed and covered
by regression tests; v1 was not overwritten or rerun.

The separately frozen v2 study used manifest SHA-256
`9433573c5813e8b3480f2eada2ae3f7929f382708d835837abfda9ba1db8d173`.
All 12 requests completed, all eight cross-arm output comparisons were exact,
process cleanup and resource receipts were complete, and budgets held. Median
full-request latency was 4.065 s baseline, 2.165 s PCSS, and 1.673 s upstream
`ngram-cache`; descriptive baseline/arm ratios were 1.878 and 2.430. PCSS
proposed 157 tokens and accepted 151, performed eight persistent hits, wrote
45,568 bytes, and held an 8,704-byte index. The synthetic miss was 3.4% slower
than baseline by point estimate, so it does not clear the preregistered 3%
miss-overhead gate and has no confidence-bound result. Upstream ngram-cache was
also faster than PCSS on this control.

After the raw run, process inventory showed that an ASan/UBSan build reported
as completed by its controller had actually continued compiling concurrently.
Therefore v2 remains valid correctness/raw-path evidence, but all of its timing
and resource comparisons are excluded from performance qualification. The
benchmark runner now fails before hashing or launch when another llama-server,
CMake build, or Make build is active, terminates the full server process group,
verifies descendant absence, and enforces the declared compaction-amplification
ceiling.

The latest clean result-bearing synthetic study is frozen v4, manifest SHA-256
`9a27655f4fe1476dfd1702fdf06b907aa080c2acb010b658f3e96f9c7ea8b979`,
against the immediately preceding native patch SHA-256
`c74a276a88ba51ed7d619d66f7e14b17bc621338244af323462713e9ab184a72`.
It ran with exclusive local compute and completed all 12 requests. All eight
cross-arm output comparisons passed, every server process group was removed,
and request/wall/artifact/RAM/disk/startup/compaction-amplification ceilings
passed. Accounted median latency (endpoint plus post-response provider settle)
was 3.521 s baseline, 1.418 s PCSS, and 1.312 s upstream ngram-cache. The
descriptive baseline/arm ratios were 2.482 and 2.684; upstream ngram-cache
remained faster than PCSS. PCSS proposed 173 tokens, accepted 167, recorded
eight persistent hits, wrote 41,600 bytes, and used an 8,704-byte RAM index.
The frozen offline analysis is
`artifacts/benchmarks/synthetic-control-v4-c74a276a-analysis.json`, SHA-256
`1b06ad95c1953c64e82e8e918ce9e9f098ceb8fdfb38986e2714b1a79c3cb8d4`.
It recomputed valid raw hashes, a complete paired matrix, and exact output. The
cluster-bootstrap baseline-over-PCSS estimate was 1.894 with 95% interval
`[1.183, 2.446]`; upstream ngram-over-PCSS was 0.931 with 95% interval
`[0.919, 0.940]`, so upstream ngram remained faster. The preregistered miss
overhead gate did not pass because this workload produced no miss-heavy sample.
This remains synthetic capability evidence, not representative evidence. The
final patch adds subsequently tested writer-lock recovery, crash-consistent
three-file compaction, peak-directory quota enforcement, bounded checkpoint
tails, and complete fallback telemetry. V4 is therefore deliberately not
relabeled as current-patch performance evidence.

The automated real-model lifecycle run for the final patch has immutable result
`artifacts/product-e2e/real-model-329d8179.json`, SHA-256
`b2e950c5b4d086c1907aa363f32284ba4848f60d5829b04ade83ff2e95f3f17b`.
All eight gates passed: greedy PCSS-on/off equality, restart persistence,
cancellation suppression and recovery, long-to-short slot reuse, concurrent
slot equality, bounded-context equality, invalid-store fallback, and
read-only/disable-learning suppression. It used at most two loopback servers
and verified complete process-group cleanup.

This agent-authored exact-repeat workload is capability/regression evidence
only. It is not an owner representative holdout and cannot support a broad
speedup, best-in-class, frontier, or default-enable claim.

### Persistent-only calibration and K safety boundary

The v5/v6 source-to-edited-prompt calibration receipts are preserved negative
evidence: no persistent hit occurred, so their timing cannot be attributed to
PCSS reuse. V7 then tested exact reuse with an explicit eight-token draft cap;
it produced one output mismatch and is invalid evidence. The sequence-line
default was therefore `K=2`, the previously correctness-gated setting.

Under that historical policy, immutable V8
`artifacts/benchmarks/persistent-exact-reuse-v8-ed3f4bc5/result.json` (SHA-256
`0804a04bef786ad39ff2cd1daaa3fe4d7ed63ec416c9769ba329e4a74c5084f3`)
passed raw correctness/resource cleanup. Its independent analysis (SHA-256
`244ed2737150e97c6917b64f4250565990463968d78bf9801c42a1d0e0c13c62`)
found baseline-over-PCSS 1.255, with clustered 95% interval `[1.222, 1.305]`.
However, its frozen grid mistakenly still declared draft size 8 while the arm
ran 2; the receipt is therefore preserved as raw calibration evidence only and
its effect-gate verdict is not qualification evidence. The 20260831 line
required a new manifest pinning `draft_tokens: [2]` before any K=2 performance
conclusion; this is an unmet historical requirement, never a current action.
The prompts are
agent-authored calibration material in every case, so no result here alters the
representative/default-enable boundary.

### Captured agent-assisted project calibration

`project-naturalistic-v4-329d8179` is a valid raw baseline--PCSS comparison
for a sealed 32-request, operator-authorized agent-assisted development stream;
it is **not** the ordinary user-initiated, four-arm project-naturalistic
protocol in `qualification/PROJECT_NATURALISTIC_CAPTURE.md`. The trusted
capture receipt recorded 8 episodes and no responses; required baseline--PCSS
exact equality passed 32/32. Persistent hits were zero. Analysis SHA-256
`fdd3060110f59848fe5b4b1261f33bc708b0b3bc84410a7ff582740e7365c53d`
reports baseline-over-PCSS 1.024 with CI95 `[0.997, 1.050]`, below the frozen
5% LCB gate, and no-persistent-hit overhead UCB95 10.0%, above its 3% gate.
This is a workload-scoped no-go for this patch, K=2, prefix=64 and hybrid
policy; it is not a general exact-PCSS technical limit. Upstream ngram is
reporting-only because its own outputs diverged from greedy baseline.

### Historical current-source synthetic capability result

Before the comparative study, the fixed local lifecycle receipt
`artifacts/product-e2e/current-source-20260831T075244Z.json` passed (SHA-256
`f4bd3fb4466e1291f68a20e720447c9a7d6076db8b7f13355203fa0944d1ccd5`). It
checks matching and incompatible request-scoped source inputs, source-index
cleanup after the request, isolated process cleanup, and absence of source
material from server logs. This is lifecycle/privacy evidence only, not a
performance result.

The opt-in request-scoped `pcss_current_source` tier then had a valid live
Qwen3-8B Q4_K_M synthetic capability receipt. The sequence-line final study
`synthetic-current-source-long-v4-9c7de25f` used exact 32-token anchors,
K=2 target verification, a bounded 8 MiB source share of a 64 MiB PCSS budget,
and 28 frozen source/mismatch/ambiguity/ordinary requests across three fresh
server arms. It passed exact output, source redaction, private-store,
process-cleanup, resource, source-activation, and effect gates.

Baseline-over-PCSS was `1.0666` with clustered 95% interval
`[1.0644, 1.0692]`, clearing the frozen 1.05 LCB gate; miss-heavy overhead
UCB95 was `-3.35%`. The source tier recorded 20 matching hit requests, 1,680
target-accepted drafts, four bounded ambiguity abstentions, four mismatch
abstentions, and zero source rejections. Raw and analysis SHA-256 values are
`d804b0ae424757ae24bce8f6577285283d1e7cb008db5c9398c9710aeba00d18` and
`f4fc2be70a4622e225e1d34f0683c7f43db7f4f3c47ca3af9de6e6e086f50d13`.

This proves only a local hand-authored synthetic current-source capability. It
does not qualify persistent history, representative/general workload speedup,
default enablement, or best-in-class performance.
