# PCSS current-source resource-admission v1 frontier ledger

## Triggering evidence

`synthetic-current-source-long-v2-9c7de25f` is a valid local synthetic
capability receipt after the benchmark harness repaired PCSS arm-store
permissions. Its raw result SHA-256 is
`8f9ebb21ac2d45eacdc9afa48b3839049fcb8d569e1f119b60eea9843a795d7d` and
its immutable analysis SHA-256 is
`dc972f869d57d8195dd5063925ac4bf1430fcbc22a41c1304bd433cfd2be3e52`.

The run preserved exact output, redaction, policy binding, resource ceilings,
and process cleanup. It also proved that all 20 matching current-source
attachments abstained before lookup: the `16 MiB` PCSS budget reserves only
`2 MiB` for source input plus the conservative per-anchor collision envelope.
The four compatibility-mismatch controls independently recorded their required
mismatch abstentions. This closes the `16 MiB` source-admission configuration;
it is not a source-tier performance result.

## New causal hypothesis

For the already frozen byte-identical 28-request hand-authored source control,
a `64 MiB` total PCSS budget gives the source tier its still-bounded one-eighth
share (`8 MiB`). That is sufficient for the same 2,583–3,199-token inputs and
their conservative exact-anchor index reservation, while retaining no learning,
no current-context fallback, K=2 target verification, and the same RAM share
denominator.

## Predeclared resource-admission study

- New study only: `synthetic-current-source-long-v3-9c7de25f`.
- Workload SHA-256 remains
  `76babc940e16364e56850b19a544d3f66f6043d9ed94454b00a3a74c0d0a7a89`.
- The only changed performance configuration is `pcss.ram_mib: 64`.
- The finite resource grid is exactly `[64]`; no post-result RAM sweep is
  allowed.
- The source policy, model, server binary, sampling, 28 request order,
  clusters, arms, effect thresholds, source/mismatch/ambiguity controls, and
  claim boundary remain unchanged.
- Before the three-arm measurement, a one-request local activation preflight
  must show nonzero source lookup/hit/accepted metrics at the same 64 MiB
  budget. It is not a latency or speedup measurement.
- One arm run only; any repeat requires an independently evidenced
  infrastructure or causal implementation defect.

The study can establish only synthetic source-tier capability. It cannot make a
representative, general, default-enable, persistent-history, or best-in-class
claim.

## Outcome

The 64 MiB activation preflight passed on the largest 3,199-token source
fixture (lookup/hit/drafted/accepted = `1/1/2/2`, peak index 608,248 bytes,
zero abstention). The subsequent V3 measurement proved source admission and a
primary-gate speedup but exposed an ambiguity-control construction defect;
that receipt remains preserved. The corrected V4 control then passed every
gate and is the valid synthetic current-source result recorded in
`FRONTIER_LEDGER_CURRENT_SOURCE_V1.md`.
