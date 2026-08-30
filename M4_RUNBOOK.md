# M4 verifier runbook — Qwen3-8B Q4_K_M

This is the target correctness, isolated verifier-profile, controlled
live-decode, and edit-protocol procedure for one fixed Qwen3-8B GGUF
configuration. Live measurements below are bounded research harnesses, not a
production decode integration or a general PCSS speedup claim.

## Current status

- Host discovery found mac2-llm.local as Mac16,8, machine name MacBook Pro,
  with an Apple M4 Pro and 24 GiB unified memory. That discovery is not a
  verifier receipt.
- The target already has the public Qwen/Qwen3-8B-GGUF
  Qwen3-8B-Q4_K_M.gguf artifact at
  /Users/ai-server/Projects/pcss-artifacts/qwen3-8b/Qwen3-8B-Q4_K_M.gguf.
  Its source revision is
  7c41481f57cb95916b40956ab2f0b139b296d974 and its SHA-256 is
  d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785.
- The target Homebrew llama.cpp installation is build 10520, commit
  cd644c395. The commands below bind the actual loaded library and bridge
  hashes instead of trusting this note.
- The isolated run at
  /Users/ai-server/Projects/pcss-artifacts/runs/qwen3-8b-q4km-e71fa7e951be
  used committed revision e71fa7e951be2ea062689c999002030dd0b328e5:
  - raw/00-preflight.json is an outer NOT_RUN preflight receipt with a nested
    observed target;
  - raw/01-manifest.json produced manifest/observed.json;
  - raw/02-seed.json and raw/03-fixtures.json are LOCAL_MEASURED seed and
    fixture-build receipts;
  - raw/10-k-sweep.json is LOCAL_MEASURED and FAILED. Its k=2 case verified;
    its k=4 case reported serial_batched_state_mismatch. Its SHA-256 is
    8c90980fce21105202fa41781ceb0616ac26f28deb2ca0e982db43fffa3e896d.
- The k-sweep kill gate stopped that historical run. Suites 11 through 14 were
  deliberately NOT RUN, and no live PCSS decode, performance, or other
  follow-on target action may run under this revision/configuration.
- A later immutable run at
  /Users/ai-server/Projects/pcss-artifacts/runs/qwen3-8b-q4km-8c82b52 used
  committed revision 8c82b52aa8731f2705b2eadb2871eec0ed823e4d. All five
  correctness suites are LOCAL_MEASURED VERIFIED, including K={2,4}, perfect
  draft with bonus, provider floor, rejection at every K=4 position, and the
  long-mixed case. Their raw receipts are 10-k-sweep.json through
  14-long-mixed.json.
- That revision then produced raw/20-verifier-profile.json, SHA-256
  1a260a739f6454147ac3be9407c277298a34ab07374715168e32851807d7a8e5.
  It is a LOCAL_MEASURED PROFILE_VALID isolated perfect-draft profile for
  K={1,2,4,8,16,32}, with two warmups and eight paired measurement clusters.
  Every paired serial baseline passed the predeclared 5% repeatability gate.
  The decode-only ratio is not a live-PCSS or end-to-end speedup result.
- A separate provider-ABI-only run at
  /Users/ai-server/Projects/pcss-artifacts/runs/provider-abi-121c139 used
  revision 121c139b5a9aadddf73031183ee69787861cc6b0. Its raw receipt SHA-256 is
  5688cd4c96a0c991d2fabd3fad8003b1299f3e2049c3934efc976fb4eb465cf1.
  Across 20 whole-block clusters of 1,000 calls, the net per-call 95% CIs were
  1.150–1.274 microseconds for no-op, 0.882–0.896 microseconds for always-miss,
  and 2.462–2.472 microseconds for deterministic shadow-hit. This is a Python
  ABI microprofile, not a live-decode overhead result.
- The derived economics receipt at
  `artifacts/m4/economics-3522274.raw.json`, SHA-256
  `c5b8118ada5b48dde9da404358fd93416723c3d22bf69f70ec8d1366b4f73cae`,
  combines only those immutable measured inputs. Its decision is
  `NO_GO_CURRENT_SERIAL_CANONICAL_COMMIT`: for every measured K={1,2,4,8,16,32},
  the upper 95% confidence bound for the current exact-state path is below 1.0.
  Batch-only K=16 and K=32 rooflines do not override this gate because the
  current bridge must serially reconstruct all committed tokens after batch
  verification.
- The serial-canonical no-go does not apply to the later pending-token protocol.
  Native tail trimming was byte-exact for K=2, but the raw K>=3 state path was
  not. Revision `aa78ec1528af6560a722e101196df335137e32b0` therefore restores
  the checkpoint and serially replays the committed anchor/prefix after every
  wide batch; this reconstruction is included in the candidate timer.
- The isolated upstream llama.cpp `0b5be7e4a25862bc2777d0c47eae18788a8c963a`
  run at
  `/Users/ai-server/Projects/pcss-artifacts/runs/pending-upstream-wide-fallback-descriptorfix-0b5be7e4-aa78ec1`
  is `LOCAL_MEASURED VERIFIED` for all five suites: K={2,3,4} sweep
  (`5ae39d5ef79291d0efc46669961693d116f69f1baa5d0ac2cad0f4e8a0c23fa9`),
  perfect draft, provider floor, K=3 rejection at every position, and
  long-mixed trajectory. The bridge and library are isolated under the PCSS
  artifact root; neither the existing sidecar nor global Homebrew was changed.
- Its separate K=3 controlled exact-repeat decode receipt is
  `CONTROLLED_LIVE_DECODE_INCONCLUSIVE`, not a speedup: all three output/state
  comparisons are exact, but serial/candidate decode ratio is `0.439614`, 95%
  clustered CI `[0.435783, 0.441683]` (`raw/20-controlled-live-k3.json`,
  SHA-256 `39105d3df84b72fe7cd50dbda4d79b4c4fa3ab11a869c6e6b5a0af84e64da5c9`).
  Each 48-token case performed 12 timed serial-state reconstructions. This
  fallback is therefore correctness evidence and a performance kill gate, not
  a viable K=3 PCSS runtime path.
- The same isolated upstream configuration also rejects K=2 as a runtime
  speedup: `raw/21-controlled-live-k2.json` is output/state exact but
  `CONTROLLED_LIVE_DECODE_INCONCLUSIVE`, with serial/candidate ratio `0.742307`
  and clustered 95% CI `[0.717225, 0.759173]` (SHA-256
  `64e6b6b945cc9b3f639ab3f23f0dd9b45a64810f55f01e2ba22b3678d2602c7c`).
  It supersedes neither the old K=2 receipt nor its narrow claim, but proves
  that the earlier positive decode-only result does not reproduce across the
  isolated llama.cpp build; neither receipt supports a portable K=2 claim.
- Causal inspection found that this negative K=2 path serialized the full
  10--12 MiB llama state before every two-token verification round even though
  native tail trimming made that checkpoint unreachable. Revision
  `587a628bf8a46847148863e9fd012951961727e6` removes only that unused work and
  records checkpoint counts in the bridge receipt. The fresh immutable run at
  `/Users/ai-server/Projects/pcss-artifacts/runs/k2-checkpoint-elision-587a628-20260830`
  is output/state exact for all three cases, has zero checkpoint and zero serial
  reconstruction rounds, and reports `CONTROLLED_LIVE_DECODE_POSITIVE`: paired
  serial-over-candidate decode ratio `1.130962`, clustered 95% CI
  `[1.128441, 1.133302]`. Raw SHA-256:
  `25d222491717d7db077034b0787ce51b0a370361086b4aff34168145e2072949`.
  This causally supersedes the negative K=2 implementation result, but its claim
  remains controlled exact-repeat decode-only: provider lookup, prefill/restore,
  misses/rejections, and thermal sustainability are excluded.
- Revision `62901edd6b1fbd4f57860476a468245508514e3d` adds a deterministic,
  trace-token-only forced-rejection profile: it replaces exactly one candidate
  token while retaining the recorded target as the authority. On the corrected
  K=2 backend, all three-case profiles remained exact output/state with zero
  checkpoints and reconstructions. A rejection at position 0 was a clear loss,
  `0.970760`, clustered 95% CI `[0.968429, 0.972161]` (raw SHA-256
  `bd492104379c592514913c54220d00511859716fa340138150b0a3c6dc922917`);
  positions 1 and 3 were also negative, respectively `0.990725`
  `[0.988004, 0.993993]` and `0.976623` `[0.974464, 0.978489]` (raw SHA-256
  `379983968c308a02d5f9fb53e0ae961956702faaf7fd55f6909172a5239453bb` and
  `cb25f03c54268fc857e7785643c6e57c6c38e3bd410d2e2622447f9f0edde34c`).
  Positions 7 and 31 were locally positive, respectively `1.005929`
  `[1.004200, 1.007160]` and `1.008789` `[1.007192, 1.010979]` (raw SHA-256
  `d0f699a1ebd1b8f49c5effd41a66e7918a34d63089db118434da4614930f26be` and
  `f7d4ac7c856cb55928f75efcb69a1a78396ad44f726e9717f89686a22604de2c`).
  Different profile runs had materially different paired baseline times, so
  this is not a monotonic threshold or production-K selection. It does establish
  the kill gate that zero-to-three-token accepted prefixes are unprofitable in
  the measured decode-only regime; a shared-process repeatability profile is
  still required before comparing positions or making policy claims.
- Current-source edit/re-anchor is a distinct but also negative controlled
  path on that exact backend. `raw/22-source-reanchor-live-k2.json` is
  output/state exact, but reports serial/candidate `0.903863`, clustered 95%
  CI `[0.806391, 0.995205]`, and status
  `CONTROLLED_LIVE_DECODE_INCONCLUSIVE` (SHA-256
  `944b2f2399d716a407170e8fb314a9f2e57b347d2e5f41b98e3943f562a2bdc5`).
  Across eight repetitions per three edit cases it attempted 108 re-anchors
  and succeeded twice. The result does not justify tuning the re-anchor
  threshold or claiming benefit from source reuse.
- The checkpoint-elision run repeated that edited-source comparison on the
  corrected K=2 path. Four-token re-anchor remained inconclusive at `1.018267`,
  clustered 95% CI `[0.991027, 1.049227]`, despite exact output/state; its raw
  SHA-256 is `54d95c61d1bdab280df16417ef6474021a64072f2ebbd741d06eb06214d68dd5`.
  The predeclared no-re-anchor control on the same cases was exact and measured
  `1.031866`, clustered 95% CI `[1.013632, 1.051853]` (raw SHA-256
  `8acaaf2fb9fe3c5deda3dfaa55828918324767c450ba10926d768eb90e124f2a`).
  Therefore bounded source reuse until the first mismatch is the current
  controlled edit-path result; repeated re-anchor is disabled as an
  unprofitable widening step until independent trace families justify revisiting
  it. Both receipts exclude provider lookup, prefill/restore, and thermal claims.
- The controlled exact-repeat live run at
  `/Users/ai-server/Projects/pcss-artifacts/runs/pending-live-7a1a721` used
  revision `7a1a721fff9a1debd5c33c393a759ac6aac85b2a`. Its
  `raw/10-controlled-live.json` (SHA-256
  `83ecbfeb605234601b9cdbca6ad2438d36b0c2cd4c72edaf870d2e12d4cac3a0`)
  is `LOCAL_MEASURED CONTROLLED_LIVE_DECODE_POSITIVE`: three independent
  48-token exact-repeat clusters, K=2, all output/state exact, paired
  serial-over-candidate decode ratio 1.130338 with clustered 95% CI
  [1.128577, 1.132274]. Prefill, restore, and provider lookup are excluded.
- The edited-source trace receipt at
  `/Users/ai-server/Projects/pcss-artifacts/runs/source-edit-traces-ee95c02/raw/00-trace-generation.json`
  has SHA-256 `6a7345a06f6d538dee75103a1c2b45131bec358b066d3c3a13d2674527d39810`.
  It contains three source/edited-target pairs from the same Qwen3-8B protocol
  with hidden reasoning excluded.
- Source reuse without re-anchor at revision
  `ee95c0203eee2545d19fcea322ac482339a2de08` preserved exact output/state but
  was inconclusive: ratio 1.018248, 95% CI [0.985840, 1.049419]. Raw SHA-256:
  `56abcc507d4bbf284be81cfc7ca5a03d0f7e2106277f5ec4e11e455014e26bc5`.
- Monotonic four-token re-anchor at revision
  `b2ae25b629eae34febba03a1f6abaad59cc19dc2` also preserved exact output/state
  but was inconclusive: ratio 1.017299, 95% CI [0.989590, 1.049455]. It found a
  later source cursor in JSON and TypeScript but did not establish aggregate
  wall-clock benefit. Raw SHA-256:
  `8a837909706fe20686cfcc90f1e203a960f749392939a6feb3765213f042f5fa`.
- The offline protocol bakeoff at revision
  `987c19ec532036abf2cf72c448b85795a31d2ff4` is `SIMULATOR VERIFIED_OFFLINE`.
  The exact patch representation covered 42.36% of output tokens, current-source
  reuse/re-anchor covered 31.25%, and persistent cross-session PCSS received
  zero unique credit after the stronger source tier. This is an oracle protocol
  comparison, not online patch generation or speedup. Raw SHA-256:
  `75e7bb0215b023d4cf0d3cb33f5e0fd82a503f0da2a4dcf7ff4061026d3bf08b`.
- The distinct causal capacity-policy replay at revision
  `e7806196d0cc8e6cfcc02cac57a391b01dd273fd` is `SIMULATOR VERIFIED_OFFLINE`.
  It evaluates one six-record Qwen3-8B trace stream (288 chronological
  observations, no future-token retrieval) over the same immutable archive.
  At 384 logical bytes, historical value-density captured 993/1682 (59.04%)
  oracle-valued unique persistent tokens in its active view, versus 1/1682
  (0.06%) for the single fixed one-epoch recency control. At 1536 bytes,
  value-density captured 1682/1682 while that control remained at 993/1682.
  Inputs, raw receipt, and compact report are sealed under
  `/Users/ai-server/Projects/pcss-artifacts/runs/capacity-policy-e780619-20260830`
  (raw SHA-256 `455daeb6a5b6649a2a80a64366bd8bc09e1972d1c4a73887f12cfca55af91609`).
  The distinct six-record source-edit trace family reproduced the ordering:
  at 384 bytes, value-density captured 288/329 (87.54%) versus 1/329 (0.30%);
  at 1536 bytes it captured 329/329 versus 292/329 (88.75%). Its sealed root is
  `/Users/ai-server/Projects/pcss-artifacts/runs/capacity-policy-source-edit-e780619-20260830`
  and raw SHA-256 is `daa83b22201f8c2d5f40f13a90943d86df7004e38c3fc28af13cb60832eaae73`.
  This selects neither a production eviction policy nor a runtime capacity; it
  is a causal offline cache-membership result only.
- The same fixed-byte/probe scope test was then emitted in compact, inspectable
  receipts at revision `615163ffd44e4410d62b61f5848efeb74d38b5c1`. With 4096
  source bytes and two probes, global/hierarchical widening gained only one
  selected oracle token over repository/block isolation on each trace family
  (2377 versus 2376 synthetic; 354 versus 353 source-edit), while selecting
  substantially more zero-accept candidates (22 versus 1; 63 versus 25).
  The offline decision is therefore repository/block isolation as the next
  read-only default; widening has no demonstrated incremental value yet. The
  sealed raw receipts are
  `/Users/ai-server/Projects/pcss-artifacts/runs/strategy-synthetic-615163f-20260830/raw/strategy.raw.json`
  (SHA-256 `b06c650f57aa93bc0b6fffc56b48068a31f0c06bff34b89cf5a2ee3989bb22e6`)
  and
  `/Users/ai-server/Projects/pcss-artifacts/runs/strategy-source-edit-615163f-20260830/raw/strategy.raw.json`
  (SHA-256 `b4fe4b6ee689c85294cc20a8ca33833fa5286dbb1a7b20218b53e32f9ae75710`).
  This is not a claim about target verification cost or end-to-end speed.
- Consensus and negative-evidence diagnostics at revision
  `3245a81ec1a3dfe7cf44fdfa5478e3b8e9be523e` produce a no-go rather than a
  manufactured policy: neither the 165 synthetic nor 108 source-edit
  persistent-candidate opportunities contained two independent provenance
  families. The exact candidate-class anti-cache also had zero prior
  suppressions (22 and 63 zero-accept observations respectively), because
  no class recurred before its later target observation. Consensus thresholds
  and negative-evidence suppression are therefore not authorized for a future
  runtime from these traces. Raw receipts are
  `/Users/ai-server/Projects/pcss-artifacts/runs/consensus-synthetic-3245a81-20260830/raw/consensus.raw.json`
  (SHA-256 `a184eddc0a87ddc74eb0a4f1720fac680f94bc58c2054a1d17535055eaf34af1`)
  and
  `/Users/ai-server/Projects/pcss-artifacts/runs/consensus-source-edit-3245a81-20260830/raw/consensus.raw.json`
  (SHA-256 `6e9f2e8979309a2d0a658ee8c829cbac828ae8bf90d211790bfc8b7c65159727`).
- The request-clustered capacity curve at revision
  `89ec9f82da5e2a998e40702e926387794fda3c26` limits the earlier descriptive
  policy comparison. Each trace has only four nonzero-value request clusters:
  at 384 and 768 bytes the 95% clustered request-mean active-capture intervals
  are broad (including zero and one) for both policies. The 1536-byte
  value-density point is 100% in all four observed clusters, but is the full
  active-view endpoint rather than evidence of a smaller capacity knee.
  No capacity, growth rule, or production policy is selected from these tiny
  traces. The sealed receipts are
  `/Users/ai-server/Projects/pcss-artifacts/runs/capacity-ci-synthetic-89ec9f8-20260830/raw/capacity-ci.raw.json`
  (SHA-256 `ac3500235db76f0f285270189aecb5c569ef8cffa0d493c1d1664943e09d33d6`)
  and
  `/Users/ai-server/Projects/pcss-artifacts/runs/capacity-ci-source-edit-89ec9f8-20260830/raw/capacity-ci.raw.json`
  (SHA-256 `a39e3d77050bab6c592fa89239f6553d2af9e0f5f29f2f0b6fc8eefea200e378`).
- A larger, distinct Qwen3-8B edit corpus at revision
  `71c72509895ca1b595f8fe6cdf0ed1aeddd0e7f4` now supplies 24 real target
  traces (12 chronological source-to-edited-target pairs, 1,152 proposal
  positions) with hidden reasoning excluded. Its generation receipt is
  `LOCAL_MEASURED TRACE_WORKLOAD_GENERATED` (SHA-256
  `f359dd41491cddfd04dfbbbb2153f3c0669316a5458e71a66eee73fd70f678b5`), while its two analyses are
  `SIMULATOR VERIFIED_OFFLINE`: `raw/10-offline-replay.json` (SHA-256
  `c71a693ac146d8506ce8b1e80fb7e703212c19428674429412310438d2cfdb00`) and
  `raw/11-protocol-bakeoff.json` (SHA-256
  `afd808cbfe3fb5589a1b4e59f33817d1ae52622631e081b29fdb0dcb0325cdc8`), all
  under
  `/Users/ai-server/Projects/pcss-artifacts/runs/diverse-edit-traces-71c7250-20260830`.
  Its 12 clustered edit pairs give direct-output coverage of 56.60% for the
  patch representation (95% CI 39.93–71.70%), 51.74% for current-source
  reuse/re-anchor (34.03–67.71%), and 0% for persistent cross-session PCSS.
  Persistent history therefore receives no incremental credit on this corpus.
  This remains offline scoring, not an online patch proposer or a speedup
  result.
- The same larger corpus rules out selecting an eviction policy or capacity
  from the simulator. At 384 and 768 logical bytes the fixed one-epoch recency
  control captured more oracle-valued persistent tokens than historical
  value-density (24.30% versus 0.05%, then 46.77% versus 32.28%); at 3072
  bytes value-density was higher (89.51% versus 82.72%), but their 20-request
  clustered intervals overlap. At identical 4096-byte/two-probe scope budgets,
  bounded widening added 64 selected oracle tokens over repository isolation
  (4133 versus 3981) but selected 362 zero-accept candidates versus 84; global
  selection likewise had 365 zero-accept candidates. No policy, capacity knee,
  widening rule, consensus threshold, or negative-evidence runtime suppression
  is authorized from it. In particular, 0 of 722 candidate opportunities had
  two independent provenance families; the simulator observed just one prior
  suppression and performed no runtime suppression.
- A separate cross-repository shared-primitive positive control at revision
  `6622227b0a0d073dd173356691b3cde1454f4e2f` establishes the boundary of that
  negative result without disguising an artificial workload as a production
  trace. It contains 12 public code prompts generated first in synthetic
  library repositories and then identically in disjoint synthetic consumer
  repositories. The target trace is `LOCAL_MEASURED`, reasoning-excluding, and
  sealed under
  `/Users/ai-server/Projects/pcss-artifacts/runs/cross-repo-positive-6622227-20260830`:
  generation SHA-256 `0211e936a8eeeae923bc7f640455a8d71960a940ebd7c914b6d2f01580d3e22c`,
  trace SHA-256 `f57757664cf9b9bd99befab812d3515e6a24b8dc6eb73461040e113ea48ef655`,
  and simulator replay SHA-256
  `fd311a18b953a3b751d4f06a286f82961bf9d4b2327e54e31622729a29c3778c`.
  Here, and only because no stronger same-repository source exists,
  persistent history earns 10,957 of 13,411 oracle accepted-prefix tokens
  (48.09% persistent-unique ORR). This is a positive-control capability
  result—not a natural-workload estimate, privacy-policy choice, online
  retrieval policy, verifier result, or speedup claim. Its capacity and scope
  figures may not select an eviction policy: the control deliberately requires
  cross-repository widening and its policy conditions do not represent the
  edit corpus above.
- Preserved roots `pending-live-648cd7d` and `pending-live-7c0318e` have no
  benchmark receipt. They failed closed on response-contract validation and
  must not be reported as performance results.
- Do not stop, restart, reconfigure, or share resources with the existing
  persona-rag sidecar. PCSS runs only from the dedicated paths below.

Qwen3-8B Q4_K_M is a small, fixed correctness target. Any result is limited to
this model artifact, raw-token protocol, runtime configuration, and exercised
cases. It cannot select a profitable K or transfer a throughput, acceptance,
memory, Metal, thermal, or speedup conclusion to Qwen3.8-27B or another
backend/quantization.

## Controlled pending-token commands

These command shapes produced the immutable receipts above. They require a
fresh config/manifest bound to the exact bridge binary and a fresh output root.
Do not overwrite an existing root.

~~~sh
PYTHONPATH="$PCSS_REPO/src" "$PCSS_PYTHON" -m pcss.llamacpp_live_benchmark \
  --config "$PCSS_CONFIG" \
  --manifest "$PCSS_RUN_ROOT/manifest/observed.json" \
  --trace "$PCSS_TRACE" \
  --pair clamp-history:clamp-repeat-target \
  --pair json-history:json-repeat-target \
  --pair typescript-history:typescript-repeat-target \
  --pair-mode exact_repeat \
  --k 2 --output-tokens 48 \
  --warmup-repetitions 2 --measurement-repetitions 8 \
  --output "$PCSS_RUN_ROOT/raw/10-controlled-live.json" \
  --report "$PCSS_RUN_ROOT/reports/10-controlled-live.md" \
  --repository-root "$PCSS_REPO"

PYTHONPATH="$PCSS_REPO/src" "$PCSS_PYTHON" -m pcss.llamacpp_live_benchmark \
  --config "$PCSS_CONFIG" \
  --manifest "$PCSS_RUN_ROOT/manifest/observed.json" \
  --trace "$PCSS_EDIT_TRACE" \
  --pair clamp-source:clamp-edited-target \
  --pair json-source:json-edited-target \
  --pair typescript-source:typescript-edited-target \
  --pair-mode source_reuse --reanchor-min-tokens 4 \
  --k 2 --output-tokens 48 \
  --warmup-repetitions 2 --measurement-repetitions 8 \
  --output "$PCSS_RUN_ROOT/raw/10-source-reanchor-live.json" \
  --report "$PCSS_RUN_ROOT/reports/10-source-reanchor-live.md" \
  --repository-root "$PCSS_REPO"

PYTHONPATH="$PCSS_REPO/src" "$PCSS_PYTHON" -m pcss.protocol_bakeoff \
  --trace "$PCSS_EDIT_TRACE" \
  --pair clamp-source:clamp-edited-target \
  --pair json-source:json-edited-target \
  --pair typescript-source:typescript-edited-target \
  --max-continuation 32 \
  --output "$PCSS_RUN_ROOT/raw/20-protocol-bakeoff.json" \
  --report "$PCSS_RUN_ROOT/reports/20-protocol-bakeoff.md" \
  --repository-root "$PCSS_REPO"
~~~

Kill gates remain fail-closed: any output/state mismatch, a confidence interval
crossing the declared benefit margin,
or changed model/runtime/trace fingerprints blocks a runtime claim. Do not
infer end-to-end gain until provider lookup is inside the timed live path and a
representative workload clears its clustered confidence gate.

## Boundary and generated contracts

The bridge accepts caller-supplied raw token IDs only. It applies neither a
chat template nor a thinking policy; the configured template label is
caller-pretokenized/v1 and thinking_mode is disabled. No hidden reasoning is
an input or output.

Target fixtures generated by the current runbook are only
pcss.m4-verifier-cases/v4. Every case has a non-empty initial context,
provenance tied to the exact GGUF and runtime configuration, and an explicit
seed_id. The seed artifact is pcss.llamacpp-fixture-seeds/v1 and is derived
from the bridge capability probe token, never from a manually guessed BOS or
tokenizer fallback:

~~~text
seed_id = qwen3-8b-q4km-capability-probe-token/v1
initial_context_tokens = [<bridge-observed probe_token>]
~~~

The long-mixed fixture has a reconstructed, independent raw-token context. It
does not claim that one live context survives across multiple adapter calls.

The preflight records six input fingerprints when observation succeeds:
adapter_spec, adapter_source, backend_binary, model_artifact,
llama_cpp_build, and runtime_configuration. It must observe Mac16,8 together
with machine name MacBook Pro; a checkout with the former prefix-only host
gate must fail closed and must not be used. The observed manifest is generated
only by pcss.llamacpp_target_artifacts manifest from that preflight receipt;
do not hand-copy a verifier_gate or hand-write a target manifest.

## Isolated artifact layout

Run everything below on mac2-llm.local. The output tree is outside the Git
checkout, so receipts can require a clean revision throughout:

~~~text
/Users/ai-server/Projects/pcss-artifacts/runs/<run-id>/
  build/pcss_llama_bridge
  inputs/{adapter-config,seed-plan,model-descriptor,fixture-seeds}.json
  manifest/observed.json
  fixtures/{k-sweep,perfect-draft,provider-floor,forced-rejection,long-mixed}.v4.json
  raw/{00-preflight,01-manifest,02-seed,03-fixtures,10-k-sweep,
       11-perfect-draft,12-provider-floor,13-forced-rejection,14-long-mixed,
       20-verifier-profile}.json
  reports/<matching names>.md
~~~

The fresh-run-root guard prevents every output path from colliding. Raw,
seed, fixture, and manifest artifacts also reject overwrite directly. A failed
or NOT_RUN raw receipt is evidence: preserve the whole run directory and do
not rerun unchanged inputs under the same run ID.

## One bounded target run

The e71fa7e951be2ea062689c999002030dd0b328e5 run above is complete as a
failed kill-gate result. Do not resume it, overwrite its artifacts, alter its
fixture/configuration, or continue with suites 11–14. The commands below are
the exact procedure only for a future, independently committed revision and
fresh run root; they are not authorization to retry the failed inputs.

Use a committed checkout transferred to the target without a push. If the
checkout is absent, transfer a Git bundle or clone the same committed revision
first; do not run from a dirty working tree.

~~~sh
set -eu

export PCSS_REPO="/Users/ai-server/Projects/llm-caching"
export PCSS_ARTIFACTS="/Users/ai-server/Projects/pcss-artifacts"
export PCSS_MODEL="$PCSS_ARTIFACTS/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
export PCSS_LLAMA_ROOT="/opt/homebrew/Cellar/llama.cpp/10520"
export PCSS_LLAMA_LIB="$PCSS_LLAMA_ROOT/lib/libllama.0.1.2.dylib"
export PCSS_LLAMA_REV="0.1.2-dev (build 10520, commit cd644c395)"
export PCSS_ADAPTER="pcss.llamacpp_adapter:create_adapter"
export PCSS_GIT_HEAD="$(git -C "$PCSS_REPO" rev-parse --verify HEAD)"
export PCSS_RUN_ID="qwen3-8b-q4km-$(printf '%.12s' "$PCSS_GIT_HEAD")"
export PCSS_RUN_ROOT="$PCSS_ARTIFACTS/runs/$PCSS_RUN_ID"

test -z "$(git -C "$PCSS_REPO" status --porcelain=v1 --untracked-files=all)" \
  || { echo "refuse dirty, staged, or untracked checkout" >&2; exit 65; }
test -f "$PCSS_MODEL"
test -f "$PCSS_LLAMA_LIB"
test "$(shasum -a 256 "$PCSS_MODEL" | awk '{print $1}')" = \
  "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785" \
  || { echo "Qwen3-8B Q4_K_M artifact hash mismatch" >&2; exit 66; }
test ! -e "$PCSS_RUN_ROOT" \
  || { echo "run directory already exists; preserve it and choose a new committed input revision" >&2; exit 67; }

mkdir -p "$PCSS_RUN_ROOT"/{inputs,manifest,fixtures,raw,reports}
export PCSS_PYTHON="$PCSS_REPO/.venv/bin/python"
if ! test -x "$PCSS_PYTHON"; then
  python3 -m venv "$PCSS_REPO/.venv"
  "$PCSS_PYTHON" -m pip install --upgrade pip
  "$PCSS_PYTHON" -m pip install -r "$PCSS_REPO/requirements-dev.txt"
  "$PCSS_PYTHON" -m pip install -e "$PCSS_REPO"
fi
test -z "$(git -C "$PCSS_REPO" status --porcelain=v1 --untracked-files=all)" \
  || { echo "environment setup dirtied checkout" >&2; exit 65; }

cmake -S "$PCSS_REPO/native" -B "$PCSS_RUN_ROOT/build" \
  -DLLAMA_CPP_ROOT="$PCSS_LLAMA_ROOT" \
  -DCMAKE_PREFIX_PATH="$PCSS_LLAMA_ROOT"
cmake --build "$PCSS_RUN_ROOT/build" --parallel 2
export PCSS_BRIDGE="$PCSS_RUN_ROOT/build/pcss_llama_bridge"
test -x "$PCSS_BRIDGE"
~~~

The fixed validation configuration below is deliberately not tuned. Its small
K values are mechanical coverage points, not a K recommendation.

~~~sh
export PCSS_CONFIG="$PCSS_RUN_ROOT/inputs/adapter-config.json"
export PCSS_SEED_PLAN="$PCSS_RUN_ROOT/inputs/seed-plan.json"

"$PCSS_PYTHON" - "$PCSS_CONFIG" "$PCSS_SEED_PLAN" <<'PY'
import os
import sys
from pathlib import Path

from pcss.canonical import canonical_json

config_path, plan_path = map(Path, sys.argv[1:])
for path in (config_path, plan_path):
    if path.exists():
        raise SystemExit(f"refuse to overwrite input: {path}")

config = {
    "schema_version": "pcss.llamacpp-adapter-config/v1",
    "bridge_protocol": "pcss.llama-bridge/v1",
    "bridge_executable": os.environ["PCSS_BRIDGE"],
    "model_artifact": os.environ["PCSS_MODEL"],
    "llama_cpp_library": os.environ["PCSS_LLAMA_LIB"],
    "llama_cpp_revision": os.environ["PCSS_LLAMA_REV"],
    "runtime": {
        "n_ctx": 512,
        "n_batch": 32,
        "n_gpu_layers": 99,
        "flash_attn": "auto",
        "seed": 0,
        "n_threads": 4,
        "n_threads_batch": 4,
        "chat_template": "caller-pretokenized/v1",
        "thinking_mode": "disabled",
    },
    "request_timeout_seconds": 600,
}
plan = {
    "schema_version": "pcss.llamacpp-capability-seed-plan/v1",
    "seed_id": "qwen3-8b-q4km-capability-probe-token/v1",
    "k_values": [2, 4],
    "provider_floor_k": 2,
    "forced_rejection_k": 4,
    "long_mixed": {"context_steps": 12, "k": 4, "rejection_position": 2},
}
for path, payload in ((config_path, config), (plan_path, plan)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
PY

export PCSS_LLAMACPP_CONFIG="$PCSS_CONFIG"
~~~

First make the runner-owned preflight. It loads the exact bridge/model and
executes its full-state capability round trip, but it never runs a verifier
case. Its outer receipt intentionally remains NOT_RUN even when its nested
target observation is observed.

~~~sh
export PCSS_PREFLIGHT="$PCSS_RUN_ROOT/raw/00-preflight.json"

"$PCSS_PYTHON" -m pcss.verifier_preflight \
  --adapter "$PCSS_ADAPTER" \
  --output "$PCSS_PREFLIGHT" \
  --report "$PCSS_RUN_ROOT/reports/00-preflight.md" \
  --repository-root "$PCSS_REPO"
~~~

The next command derives the descriptor from the actual preflight binding; it
does not type an adapter binding, model hash, or runtime fingerprint by hand.

~~~sh
export PCSS_DESCRIPTOR="$PCSS_RUN_ROOT/inputs/model-descriptor.json"

"$PCSS_PYTHON" - "$PCSS_CONFIG" "$PCSS_PREFLIGHT" "$PCSS_DESCRIPTOR" <<'PY'
import json
import os
import sys
from pathlib import Path

from pcss.canonical import canonical_json, sha256
from pcss.llamacpp_fixture_builder import load_bridge_config
from pcss.results import file_sha256

config_path, preflight_path, output_path = map(Path, sys.argv[1:])
if output_path.exists():
    raise SystemExit(f"refuse to overwrite descriptor: {output_path}")
config = load_bridge_config(config_path)
receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
payload = receipt.get("payload", {})
if receipt.get("git_revision") != os.environ["PCSS_GIT_HEAD"]:
    raise SystemExit("preflight Git revision differs from this run")
if payload.get("observation_status") != "observed":
    raise SystemExit(f"preflight is not observed: {payload.get('reason')}")
binding = payload.get("adapter_binding")
if not isinstance(binding, dict):
    raise SystemExit("preflight has no adapter binding")
model_sha256 = file_sha256(config.model_artifact)
if model_sha256 != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785":
    raise SystemExit("configured GGUF hash differs from Qwen3-8B Q4_K_M")
if binding.get("model_artifact_sha256") != model_sha256:
    raise SystemExit("preflight model binding differs from configured GGUF")
if binding.get("runtime_configuration_fingerprint") != config.runtime_configuration_fingerprint:
    raise SystemExit("preflight runtime binding differs from explicit config")

def protocol(parameters):
    return {"fingerprint": sha256(parameters), "parameters": parameters}

descriptor = {
    "schema_version": "pcss.target-model-descriptor/v1",
    "run_id": os.environ["PCSS_RUN_ID"],
    "model_artifact": {
        "artifact_uri": (
            "hf://Qwen/Qwen3-8B-GGUF@"
            "7c41481f57cb95916b40956ab2f0b139b296d974/"
            "Qwen3-8B-Q4_K_M.gguf"
        ),
        "sha256": model_sha256,
        "source": {
            "source_class": "huggingface",
            "locator": "Qwen/Qwen3-8B-GGUF",
            "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
            "artifact_name": "Qwen3-8B-Q4_K_M.gguf",
        },
    },
    "quantization": protocol({"format": "gguf", "quantization": "Q4_K_M"}),
    "tokenizer": protocol({"source": "model-artifact"}),
    "chat_template": protocol({"template": config.runtime["chat_template"]}),
    "generation_protocol": protocol({
        "generation_mode": "greedy_argmax",
        "token_input": "caller-supplied-raw-token-ids/v1",
        "thinking_mode": "disabled",
    }),
    "llama_cpp": protocol({
        "llama_cpp_revision": config.llama_cpp_revision,
        "llama_cpp_build_fingerprint": binding["llama_cpp_build_fingerprint"],
        "llama_cpp_library_sha256": file_sha256(config.llama_cpp_library),
    }),
    "context_kv": protocol({
        "runtime_configuration_fingerprint": config.runtime_configuration_fingerprint,
        "runtime": dict(config.runtime),
    }),
    "benchmark": protocol({
        "measurement_status": "not_run",
        "parameters": {"purpose": "verifier-correctness-boundary/v1"},
    }),
}
with output_path.open("x", encoding="utf-8") as handle:
    handle.write(canonical_json(descriptor) + "\n")
PY

export PCSS_MANIFEST="$PCSS_RUN_ROOT/manifest/observed.json"
"$PCSS_PYTHON" -m pcss.llamacpp_target_artifacts manifest \
  --config "$PCSS_CONFIG" \
  --preflight "$PCSS_PREFLIGHT" \
  --descriptor "$PCSS_DESCRIPTOR" \
  --output "$PCSS_MANIFEST" \
  --raw-result "$PCSS_RUN_ROOT/raw/01-manifest.json" \
  --report "$PCSS_RUN_ROOT/reports/01-manifest.md" \
  --repository-root "$PCSS_REPO"
~~~

Create the only valid seed context and all five V4 fixtures. The seed command
is a target capability observation; the fixture command calls the serial
greedy model-memory reference bridge. Neither is a verifier acceptance or a
benchmark.

~~~sh
export PCSS_SEEDS="$PCSS_RUN_ROOT/inputs/fixture-seeds.json"

"$PCSS_PYTHON" -m pcss.llamacpp_target_artifacts seed \
  --config "$PCSS_CONFIG" \
  --plan "$PCSS_SEED_PLAN" \
  --output "$PCSS_SEEDS" \
  --raw-result "$PCSS_RUN_ROOT/raw/02-seed.json" \
  --report "$PCSS_RUN_ROOT/reports/02-seed.md" \
  --repository-root "$PCSS_REPO"

"$PCSS_PYTHON" -m pcss.llamacpp_fixture_builder \
  --config "$PCSS_CONFIG" \
  --seeds "$PCSS_SEEDS" \
  --fixture-directory "$PCSS_RUN_ROOT/fixtures" \
  --raw-result "$PCSS_RUN_ROOT/raw/03-fixtures.json" \
  --report "$PCSS_RUN_ROOT/reports/03-fixtures.md" \
  --repository-root "$PCSS_REPO"

"$PCSS_PYTHON" - "$PCSS_RUN_ROOT/fixtures" <<'PY'
import json
import sys
from pathlib import Path

from pcss.verifier_runner import load_cases

root = Path(sys.argv[1])
base_seed = "qwen3-8b-q4km-capability-probe-token/v1"
names = (
    "k-sweep.v4.json",
    "perfect-draft.v4.json",
    "provider-floor.v4.json",
    "forced-rejection.v4.json",
    "long-mixed.v4.json",
)
for name in names:
    path = root / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "pcss.m4-verifier-cases/v4":
        raise SystemExit(f"not a V4 fixture: {path}")
    cases = load_cases(path)
    if not cases or any(not case.initial_context_tokens or not case.seed_id for case in cases):
        raise SystemExit(f"fixture lacks required V4 context/seed identity: {path}")
    if any(case.seed_id != base_seed for case in cases):
        raise SystemExit(f"unexpected seed identity: {path}")
print("five target-bound V4 fixtures: OK")
PY
~~~

Run each suite once, in this order. Each invocation separately reloads the
adapter, re-observes the target, recomputes bindings, rejects stale V4
provenance before verify(), and records the command, Git revision, fixture
SHA-256, and manifest SHA-256 in its raw receipt.

~~~sh
run_suite() {
  suite="$1"
  fixture="$2"
  stem="$3"
  "$PCSS_PYTHON" -m pcss.verifier_runner \
    --adapter "$PCSS_ADAPTER" \
    --suite "$suite" \
    --fixture "$PCSS_RUN_ROOT/fixtures/$fixture" \
    --manifest "$PCSS_MANIFEST" \
    --output "$PCSS_RUN_ROOT/raw/$stem.json" \
    --report "$PCSS_RUN_ROOT/reports/$stem.md" \
    --repository-root "$PCSS_REPO"
}

run_suite k-sweep k-sweep.v4.json 10-k-sweep
run_suite perfect-draft perfect-draft.v4.json 11-perfect-draft
run_suite provider-floor provider-floor.v4.json 12-provider-floor
run_suite forced-rejection forced-rejection.v4.json 13-forced-rejection
run_suite long-mixed long-mixed.v4.json 14-long-mixed
~~~

Only after all five receipts are LOCAL_MEASURED VERIFIED for the same committed
bridge and manifest, run the isolated profile once:

~~~sh
"$PCSS_PYTHON" -m pcss.llamacpp_profiler \
  --config "$PCSS_CONFIG" \
  --seeds "$PCSS_SEEDS" \
  --manifest "$PCSS_MANIFEST" \
  --output "$PCSS_RUN_ROOT/raw/20-verifier-profile.json" \
  --report "$PCSS_RUN_ROOT/reports/20-verifier-profile.md" \
  --repository-root "$PCSS_REPO" \
  --k 1 --k 2 --k 4 --k 8 --k 16 --k 32 \
  --warmup-repetitions 2 \
  --measurement-repetitions 8 \
  --baseline-relative-tolerance 0.05
~~~

The raw response records every paired serial-before, batch-verify, and
serial-after duration, plus restore and snapshot timings. Context creation and
prompt prefill are outside the timed decode region. Treat
BASELINE_REPEATABILITY_FAILED as a kill gate. PROFILE_VALID only validates the
receipt and repeatability; it does not select K or establish profitability.

The provider ABI microprofile is a separate committed run because it does not
load the model or modify the verifier bridge:

~~~sh
"$PCSS_PYTHON" -m pcss.provider_profiler \
  --manifest "$PCSS_MANIFEST" \
  --output "$PCSS_RUN_ROOT/raw/30-provider-abi-profile.json" \
  --report "$PCSS_RUN_ROOT/reports/30-provider-abi-profile.md" \
  --repository-root "$PCSS_REPO" \
  --blocks 20 \
  --calls-per-block 1000 \
  --warmup-calls 100
~~~

It records empty-call controls both before and after each provider block and
bootstraps whole blocks. Do not substitute its microsecond result for a shadow
provider attached to real target generation.

Derive the fail-closed economics decision from the three immutable receipts:

~~~sh
PYTHONPATH=src python3 -m pcss.economics \
  --profile artifacts/m4/qwen3-8b-q4km-8c82b52/raw/20-verifier-profile.json \
  --verifier artifacts/m4/qwen3-8b-q4km-8c82b52/raw/10-k-sweep.json \
  --provider artifacts/m4/provider-abi-121c139/raw/00-provider-abi-profile.json \
  --output artifacts/m4/economics-3522274.raw.json \
  --report artifacts/m4/economics-3522274.report.md \
  --repository-root .
~~~

The command rejects dirty checkouts, failed or non-local receipts, mismatched
verifier revisions, different target hosts, and non-target correctness
evidence. `NO_GO_CURRENT_SERIAL_CANONICAL_COMMIT` is a kill gate for attaching
the present provider/verifier path to live decoding. It is not a negative claim
about a future verifier that can preserve exact canonical state without serial
reconstruction.

The actual e71fa7e951be run has exactly these durable artifacts; reports exist
only after their matching raw artifact:

~~~text
raw/00-preflight.json            outer NOT_RUN target observation
raw/01-manifest.json             manifest-build boundary, not verifier correctness
raw/02-seed.json                 capability-derived seed observation
raw/03-fixtures.json             serial-reference fixture build
raw/10-k-sweep.json              LOCAL_MEASURED FAILED kill-gate receipt
manifest/observed.json
inputs/{adapter-config,seed-plan,model-descriptor,fixture-seeds}.json
fixtures/{k-sweep,perfect-draft,provider-floor,forced-rejection,long-mixed}.v3.json
~~~

raw/11-perfect-draft.json through raw/14-long-mixed.json and their reports are
intentionally absent from that run. For a future, separately committed run,
the commands retain those four paths as the expected continuation only after
the preceding suite verifies; a failure stops the sequence at its own receipt.

## Kill gates and interpretation

Stop immediately and preserve the run directory if any of these happens:

- the checkout is dirty, the Git revision changes, the model SHA-256 is not
  the pinned value above, or the formula library/bridge/config file changes;
- CMake cannot build the bridge against the stated llama.cpp installation;
- preflight does not produce an observed TARGET_BACKEND payload with all six
  fingerprints, or the bridge cannot prove a byte-exact model-memory
  llama_state_get_data/llama_state_set_data round trip;
- manifest generation rejects the preflight binding, descriptor, host identity,
  model hash, loaded library hash, llama.cpp build fingerprint, or runtime
  configuration;
- seed derivation lacks a valid capability probe token, any fixture is not V4,
  lacks the named seed identity/non-empty context, or provenance changes;
- any runner receipt is NOT_RUN, FAILED, not LOCAL_MEASURED, or has a case
  commit/state mismatch. Do not proceed to the next suite;
- the derived economics receipt is `NO_GO_CURRENT_SERIAL_CANONICAL_COMMIT`.
  Do not select K or attach the current verifier/provider path to live decoding;
- the recorded e71 k=4 serial_batched_state_mismatch is already such a kill
  gate. It is not a debugging instruction or a basis for a PCSS conclusion;
  preserve the receipt and perform no further target action for that
  revision/configuration;
- the sidecar would need to be touched, inputs would need to be overwritten,
  or an unchanged failed input set would need a retry.

A verified individual case establishes only its tested raw-token commit and
model-memory state equivalence for that case. The isolated profile and derived
economics receipt establish a no-go only for the measured current serial
canonical-commit path. They do not establish live speculative decode
correctness, an optimal K, speedup, mmap-versus-pread, unified-memory behavior,
swap/page-fault behavior, Metal behavior, thermals, online storage behavior, or
a production PCSS controller. Those remain separate gates and must not be run
through the current path after its economics kill gate.
