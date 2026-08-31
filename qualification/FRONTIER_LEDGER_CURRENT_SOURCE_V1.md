# PCSS current-source exact-anchor v1 frontier ledger

This is a new retrieval family after the frozen suffix-ladder study
`synthetic-suffix-ladder-v1-2617641b`. It neither replaces that study nor
reopens unsafe draft depth above K=2.

## Causal hypothesis

The suffix-ladder control activated 72 exact persistent 32-token proposals but
only reached a baseline/PCSS estimate of 1.0268 (95% interval
`[1.0095, 1.0409]`). The prior visible prompt does not always preserve the
developer's stable source anchor. An explicit caller-supplied, request-scoped
visible source buffer can supply a stronger exact anchor without inferring
filesystem paths, repositories, or hidden future tokens.

## Frozen v1 contract

- Policy specification: `policies/pcss-current-source-exact-anchor-v1.json`.
- Proprietary opt-in request field: `pcss_current_source` with exact schema,
  target-vocabulary token IDs, opaque SHA-256 source id, and compatibility
  fingerprint equal to the configured PCSS fingerprint.
- Default off server gate: `--pcss-enable-current-source`.
- Source data is request-scoped only: no source text/id/tokens in persistent
  records, logs, metrics labels, capture receipts, or later slots.
- At most 16,384 supplied tokens; source anchor is an exact 32-token suffix.
- At most 64 candidate positions for one anchor; overflow or conflicting
  two-token continuations abstains.
- Source input plus its conservative index reservation is globally admitted
  across active slots. The source tier owns at most one eighth of configured
  PCSS RAM while the durable store is capped to the remaining seven eighths;
  exhausted admission abstains without disabling target generation.
- Target verification remains K=2. No K>2 variant is permitted by this study.
- Priority is current source, then exact persistent, then current-context
  fallback. Missing, disabled, incompatible, malformed, or ambiguous source
  must preserve ordinary generation by abstaining.

## Required gates before measurement

1. Native exact hit/miss/ambiguity/cap/lifecycle tests pass.
2. Raw `/metrics` shows source-tier lookup/hit/ambiguity/draft/accepted/index
   accounting with no source identity label.
3. OpenAI-compatible source extension validation, disabled behavior, mismatch
   abstention, and slot reuse are tested.
4. A fresh immutable source-bearing synthetic capability study is frozen before
   its three arm results are opened. It must bind its policy file SHA and live
   server policy info metric.

## Single allowed calibration

One source-bearing synthetic capability control may measure baseline, upstream
ngram K=2 reporting-only, and this PCSS policy once. It must include source
hit and no-source/mismatch/ambiguity abstention requests, preserve all raw
samples, and use a predeclared counterbalanced or explicitly single-order
claim boundary. No prompt/source redesign after its arm results is allowed.

This can establish only exact local source-tier capability unless an
independent owner-authorized source-bearing holdout is later supplied.

## Resource-admitted synthetic capability result

The original `16 MiB` total-PCSS configuration is closed as an admission
failure: its one-eighth source share could not admit the frozen 2,583–3,199
token source buffers. The distinct resource-admission ledger therefore froze a
single `64 MiB` total-PCSS variant, retaining the bounded `8 MiB` source share
and every other policy parameter.

The final one-ambiguity-control confirmation,
`synthetic-current-source-long-v4-9c7de25f`, passed every preregistered raw,
privacy, source-activation, resource, exact-output, and effect gate. Its raw
result SHA-256 is
`d804b0ae424757ae24bce8f6577285283d1e7cb008db5c9398c9710aeba00d18`; its
analysis SHA-256 is
`f4fc2be70a4622e225e1d34f0683c7f43db7f4f3c47ca3af9de6e6e086f50d13`.

- Baseline-over-PCSS estimate: `1.0666367292290648`.
- Clustered 95% interval: `[1.064369322375595, 1.0692194794528975]`, clearing
  the frozen `1.05` LCB gate.
- Miss-heavy overhead UCB95: `-0.033505602235602616`, clearing the `<= 0.03`
  gate.
- Source tier: 20 matching lookup/hit requests across four episodes, 1,680
  target-accepted draft tokens, four bounded ambiguity abstentions, four
  compatibility-mismatch abstentions, and zero source rejections.

This is a measured local synthetic capability result only. It does not qualify
persistent history, an owner-representative workload, a general speedup,
default enablement, or best-in-class performance.

## Prospective owner-holdout path

The capture path now has an opt-in v2 current-source sidecar: a public,
hash-chained pre-forward ledger retains only source commitments while the raw
source envelopes remain in a new `0600` local sidecar. Sealing requires an
exact ledger/sidecar bijection and emits a `0600` source-bearing workload plus
an immutable public receipt. An ordinary owner source-bearing holdout manifest
must bind that receipt, ledger digest, sidecar digest, attachment count, and
per-request commitments before its one-shot opening.

No ordinary owner trace has been captured and no owner holdout has been opened.
This implementation removes a capture/provenance product gap; it is not
evidence about performance or representativeness.
