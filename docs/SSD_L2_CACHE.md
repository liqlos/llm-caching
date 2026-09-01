# Native RAM + SSD L2 prompt/KV cache (llama.cpp line — FROZEN)

> **FROZEN LINE (2026-09-01):** the project moved to MLX / mlx-lm; see
> `NOW.md` (plan) and `docs/MLX_CACHE.md` (current MLX state). This file
> remains the authority for what the llama.cpp line established and for its
> receipts, and is the source of the transferable findings listed in
> `docs/MLX_CACHE.md`. It is not an instruction to continue this line.

## Status and claim boundary

The native llama-server cache supports the existing RAM prompt/KV tier plus an
optional persistent SSD L2 tier. SSD L2 is disabled by default. For the
caller-attested synthetic Apple M4 workload, the qualified measured
baseline-over-SSD ratios are `4.9897x` for exact reuse and `1.5825x` for partial
reuse. The novel scenario establishes only no detected latency difference. All
claims are bounded by the specific binaries, model, compatibility key, and
receipts below.

This is not evidence for representative owner-workload economics. It is also
not a sequence-datastore approval: that design remains
`NO_GO_CURRENT_REDESIGN`.

## Representative-workload status: Phase 1b `NO_GO` is PROVISIONAL, superseded by Phase 1c

The Phase 1b `NO_GO_SSD_L2_PRODUCT` verdict recorded below was reached under
a forced single-tier configuration (`--cache-ram 1`, no RAM tier, store at
its quota ceiling). It measured an SSD-only cache with per-turn full-state
writes, not the multi-level RAM+SSD product. It stands as a receipt for that
configuration only. The live gate is G1c, defined in `NOW.md` (Phase 1c),
which measures the multi-level configuration on owner traffic with the write
amplification addressed. Until G1c is recorded here, no product-level
NO_GO or GO is in force. What Phase 1b did establish, and what stands:

- the single-slot save trigger patch works (write path verified);
- the owner-trace corpus pipeline works (v2, v2b);
- on hit turns with reused prefix under ~4,000 tokens, the SSD arm was
  1.05x-1.3x faster than baseline in both corpora - the first real-traffic
  benefit measured in this project.

### Phase 1 — the write path structurally never fired

A representative multi-session workload (real repository bytes, a
byte-identical system preamble, growing per-session transcripts, real
server restarts between sessions, `--parallel 1` matching a solo
developer's sequential usage) was run twice on Apple M1 Pro. Full receipts:
`artifacts/ssd-redesign/20260901T152546Z-4c0f7eab23ef/RUN_SUMMARY.md`.

- **The persistent SSD tier never engaged in either attempt**: 0 files, 0
  bytes written to the disk store, 0 eviction/disk-write log lines, across
  both a generous (`--cache-ram 512`) and a minimal (`--cache-ram 1`) RAM
  budget.
- **Root cause (source-confirmed, not a tuning issue)**: neither pre-patch
  `prompt_save` call site fires for this traffic. The idle-sibling-slot loop
  (gated by `--cache-idle-slots`) runs over *other, currently idle* slots
  right after a task launches on a different slot; with `--parallel 1` the
  single slot is never idle at that check. The prompt-switch path in
  `get_available_slot` fires only on LRU selection or LCP `f_keep < 0.5`,
  which a single growing conversation per process never triggers. So the
  save-to-cache path structurally never fired — for any `--cache-ram` value. This was a multi-slot mechanism: it persisted a
  conversation that goes idle while a different one became active on
  another slot. It did not persist a single sequential session across a
  restart.
- This did not contradict the exact/partial qualification receipts below:
  those drove each request through a **separate, fresh server process**,
  which hits a different code path (initial slot load) than the idle-slot
  handoff this run needed. It narrows what those receipts cover: single
  fresh-process exact-repeat lookups, not sustained multi-session
  solo-developer persistence.
- The learning-write overhead question below ("a clean learning-overhead
  component was not identified") was answered here: paired population-pass
  measurements in both attempts showed no reproducible net cost beyond noise
  once the feature had nothing to write, consistent with the disk tier
  never engaging.

### Phase 1b — trigger patch closes the write path; `NO_GO` under forced single-tier config (provisional)

Full receipts: `artifacts/ssd-redesign/20260901T163602Z-phase1b-trigger-patch/RUN_SUMMARY.md`.

A patch (uncommitted in the `third_party/llama.cpp` submodule, pending
explicit user commit approval) added two new save triggers that fire for
single-slot sequential traffic: `save_slot_on_completion` (debounced via a
new `--cache-save-ms`, default 5000ms, gated by a new
`--cache-save-on-completion`/`--no-cache-save-on-completion` flag, default
on) fires when a completion finishes; `save_slots_on_shutdown` flushes every
slot's current state on graceful `SIGTERM`, bypassing the debounce. Locally
verified on the M1 (build + unit tests) per the execution-host rule.

The decisive rerun, on the M4 host, against two independent real
owner-trace corpora built from the owner's own local Claude Code session
history (see "Owner-trace corpus" in `NOW.md`), confirms the write-path fix
works and answers the reopened question, negatively:

- **Corpus v2** (one session per project, 4 distinct projects): disk tier
  engaged (3,666,303,328 bytes written, 151 eviction/disk log lines — the
  write gap is genuinely closed). But **0 of 3** restart boundaries showed
  a first-turn-after-restart cache hit (the unconfounded persistence
  signal — see the run's analyzer for why the naive 96% aggregate hit rate
  is not usable evidence, being saturated by llama.cpp's pre-existing
  same-slot continuity, present regardless of `--cache-ram`/`--cache-ssd`).
  Root cause: sampling unrelated projects per session leaves no real shared
  content across a restart boundary beyond a short constructed preamble.
- **Corpus v2b** (4 chronological sessions from a single project, the
  structurally correct test of "resuming the same project after a
  restart"): disk tier engaged (4,097,953,872 bytes, 124 eviction/disk log
  lines). **1 of 3** restart boundaries showed a genuine
  first-turn-after-restart hit — real evidence the mechanism can work for
  continuing-project traffic. But paired wall-clock on the hits that did
  occur was net negative for the SSD arm (mean ratio baseline/ssd = 0.866,
  95% CI [0.785, 0.947], excluding 1.0).

Both pre-registered Gate G1b conditions (persistent hit rate materially
nonzero AND paired wall-clock benefit real) must hold for `GO_PROFIT_LADDER`;
neither corpus cleared both. Per NOW.md's two-strike anti-stall rule, these
are two materially different corpus designs (not a bare rerun), each
addressing a named confound in the one before it — a third variant without a
new mechanism-level hypothesis is not a legitimate next iteration.

**Diagnostic finding (post-hoc analysis of already-collected data, does not
reopen the gate):** bucketing hit turns by reused-prefix size (`cache_n`)
shows a clean, monotonic pattern reproduced independently in both corpora —
genuinely faster under SSD at small `cache_n` (< 1,500 tokens, ratio
1.07-1.28x), inverting to ~1.7-1.8x *slower* by `cache_n` 8,000-15,000.
`ssd_s` grows disproportionately with `cache_n` while `baseline_s` tracks
`tokens_evaluated` as expected. Both decisive runs used a deliberately
minimal `--cache-ram 1` (to guarantee disk-tier engagement), which means
every restore paid the full disk round-trip with no RAM tier, every
completion save was demoted to disk immediately (a growing conversation
rewrites its full state each turn), and the v2b store sat at its 4096 MiB
quota ceiling with compaction inside the measured window. The slowdown at
large `cache_n` is therefore attributable to that write/restore path under
a degenerate config until a cost decomposition says otherwise. Full detail:
`RUN_SUMMARY.md`'s "Diagnostic finding" section.

**Gate G1b verdict: `NO_GO` for the forced single-tier configuration only;
superseded by Gate G1c** (see the status section at the top of this file and
`NOW.md` Phase 1c). The multi-level RAM+SSD product configuration has not
yet been measured on owner traffic. The trigger patch and both corpora's
receipts remain valid evidence for an upstream llama.cpp contribution
regardless of the eventual product verdict.

## CLI configuration

Configure all three SSD options together:

```sh
llama-server \
  --cache-ram <MiB> \
  --cache-ssd <MiB> \
  --cache-ssd-path /absolute/cache/root \
  --cache-ssd-key sha256:<64-lowercase-hex> \
  ...
```

- `--cache-ssd <MiB>` sets the maximum SSD L2 size. `0` disables SSD L2 and is
  the default.
- `--cache-ssd-path <PATH>` sets the root directory. The server uses a child
  directory named by the hexadecimal portion of `--cache-ssd-key`.
- `--cache-ssd-key <SHA256>` selects the compatibility namespace. Its accepted
  form is exactly `sha256:` followed by 64 lowercase hexadecimal characters.
- A nonzero `--cache-ssd` requires both path and key. Supplying only a subset of
  the three options is rejected.
- RAM and SSD can be enabled together. SSD remains optional; setting both cache
  size controls to zero disables the prompt cache.

The compatibility key is caller-attested. The implementation validates its
syntax but does not independently derive or prove that model, tokenizer,
template, llama.cpp build, or runtime settings are compatible. Reuse the key
only when the caller has established that full compatibility identity.

## Request behavior

The cache looks for the longest compatible stored prompt prefix and reports the
number of reused tokens as `cache_n`.

- Exact: after learning the 1,000-token prompt, an identical prompt reuses all
  1,000 tokens. The final receipt observed `cache_n = 1000` in every SSD arm.
- Partial: a prompt sharing the stored prefix reuses the longest matching
  checkpoint. The final receipt observed `cache_n = 503` in every SSD arm.
- Novel: an unrelated prompt is a miss and follows ordinary decode. The final
  receipt observed `cache_n = 0` in every SSD arm.

The qualification receipt records identical output text and generated token IDs
between baseline and SSD paths for exact, partial, and novel scenarios. The
novel arm's ratio was `0.99948`, with its confidence interval crossing `1.0`;
this supports only no detected latency difference, not zero or negligible
latency impact.

## Restart, corruption, and quota semantics

- Successful committed entries persist across server restart and remain
  eligible for exact or partial-prefix reuse.
- Startup recovers the last committed manifest state. Uncommitted record or
  manifest tails are not treated as committed cache entries, and the retained
  previous files provide recovery when the current pair is absent.
- Records are checksum-validated before use. A corrupt record is not served and
  increments the corruption counter.
- If a durable manifest update fails, the SSD store is disabled for the rest of
  that server process. After the underlying failure is corrected, restart can
  reopen the previously committed state; the failed write is not committed.
- `--cache-ssd` is a physical-byte quota, not a guaranteed entry count. When a
  new entry would exceed the quota, older entries are evicted and the store is
  compacted so physical storage remains within the configured limit.
- Quota pressure therefore permits misses. In the final restart pressure receipt,
  512 MiB retained 3 of 4 prefixes (`452,594,440` store bytes), while 1,024 MiB
  retained all 4 (`603,459,296` store bytes). Both arms completed without a
  crash and returned generated tokens. This establishes bounded capacity,
  eviction, and restart behavior for the probe; it does not establish
  baseline/SSD equality of generated token IDs or output under pressure.

## Final Apple M4 receipts

Run root:
`artifacts/ssd-redesign/20260901T122807Z-84982163c750`

Hardware: Apple M4 Pro, 24 GiB. Each latency scenario used six paired
iterations and 10,000 bootstrap resamples.

| Scenario | Baseline mean | SSD mean | Baseline / SSD | Latency reduction | 95% bootstrap interval | `cache_n` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | 3.767778 s | 0.755110 s | 4.9897x | 79.9588% | [4.968050, 5.019255] | 1000 |
| Partial | 3.767428 s | 2.380663 s | 1.5825x | 36.8093% | [1.578717, 1.587294] | 503 |
| Novel | 3.767945 s | 3.769903 s | 0.999481 | -0.0519% | [0.998510, 1.000445] | 0 |

Qualification receipt:
`live/ssd-l2-qualification-v19-final/result.json`

- Schema: `pcss.ssd-l2-prompt-cache-qualification/v6`
- Receipt SHA-256:
  `88754009441449411316c1aa23d076a9947bd16914d02116085f1f922e193862`
- Claim boundary: `final caller-attested synthetic M4 cache-specific qualification`
- Measured `learning_request_wall_s`: `3.791445`
- A clean learning-overhead component was not identified.
- Store size: `150,864,824` bytes
- Exact output text and generated token IDs: `true`

Pressure receipt: `live/pressure-v16-final/result.json`

- Schema: `pcss.ssd-l2-pressure/v4`
- Receipt SHA-256:
  `b073e6571c5434efeabc07ec1cd05f610faab1e8478164cc460c5d293e34e950`
- No crash: `true`
- Generated tokens present: `true`
- 512 MiB: 3 hits in 4 probes, `452,594,440` store bytes
- 1,024 MiB: 4 hits in 4 probes, `603,459,296` store bytes

Both final receipts identify:

- Model SHA-256:
  `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- llama.cpp SHA-256:
  `e4bcc69e760c1f8ce8c640a968841fa91757169334b9b6d48e9522558d0b5986`
- Server implementation SHA-256:
  `782e518e85d7cbdc8cd844469114715be573e59856dd50de8b11d4c322d2c0ef`
- Compatibility key:
  `sha256:e75eb4785c0efe328a26544cd05169c77e035d2b8100d3985391d34b9a761457`

## Limitations and retained historical evidence

- The workload is synthetic and cache-specific. It does not establish latency,
  hit rate, capacity, or economics for representative owner traffic.
- The compatibility key is supplied by the caller; caller-independent identity
  derivation and enforcement are not demonstrated.
- The receipts bind claims to the recorded M4 hardware and exact model,
  llama.cpp, server implementation, key, workload, and run configuration. They
  do not support a transferable general speedup claim.
- The pressure receipt establishes bounded capacity, eviction, and restart
  behavior for its four-prefix probe. It does not establish generated-token-ID
  or output equality under pressure, sustained production pressure, thermal
  behavior, or arbitrary working sets.
- Earlier v14/v15 receipts and earlier PCSS measurements remain useful historical
  provenance, but they are not current authority for SSD L2 qualification.
- Positive SSD L2 results do not reverse the sequence-datastore verdict:
  `NO_GO_CURRENT_REDESIGN`.
- The representative-workload question this synthetic evidence left open is
  still open at the product level: the write-path trigger gap has been
  patched and the forced single-tier config measured (provisional `NO_GO`
  for that config); the multi-level configuration is measured by Phase 1c /
  Gate G1c per `NOW.md`.
