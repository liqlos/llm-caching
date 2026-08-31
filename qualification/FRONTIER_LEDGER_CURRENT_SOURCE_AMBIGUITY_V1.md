# PCSS current-source one-ambiguity-control v1 frontier ledger

## Triggering evidence

The valid resource-admitted study
`synthetic-current-source-long-v3-9c7de25f` recorded a real exact source-tier
effect: all 20 matching rows looked up and hit, 1,676 source draft tokens were
target-accepted, no source draft was rejected, and the paired
baseline/PCSS LCB95 was `1.0623566146490762`. Its immutable raw/analysis
SHA-256 values are respectively
`2ad1ecd17ea6240190a87f927955c907cd7781797ae74b5b65ba0a8723b65b31`
and `7ed8b04247e8d504cd64baed1e32f102311504fed36d82c448d97d8909c4120b`.

V3 deliberately remains failed as a full gate result: each of four ambiguity
control rows safely produced two ambiguity abstentions, for eight total, while
the frozen control cap allowed four. This is not a rejection, mismatch, or
privacy failure; it is an under-specified negative-control construction.

## New causal hypothesis

The ambiguity source can be constructed to create exactly one deterministic
conflicting exact anchor on the ordinary continuation path while retaining
unique exact source anchors before and after it. The alternate pair must
diverge at its first token, so the next shifted anchor cannot repeat the
conflict. That proves bounded ambiguity abstention without imposing two such
events per episode.

For each episode, before any V4 arm result:

1. Take the hand-authored base source and locally tokenize only the already
   visible ambiguity-control prompt.
2. Find the unique base-source occurrence of the prompt's final 32 tokens.
3. Choose the source-only deterministic trigger at offset 64 tokens after that
   occurrence, append one duplicate of its 32-token anchor, and append one
   deterministic target-vocabulary continuation pair whose first token differs
   from the base continuation's first token.
4. Do not use a model response, target completion, latency, or V3 arm output
   to choose that source vector.

The resulting candidate workload SHA-256 is
`e0eba146ce36fbfc39f9c0f0a599ec4ba474eead940c8cfc4f9c0701604e6be7`.
The local 64 MiB source activation preflight passed for every control and is
recorded as `synthetic-current-source-long-v4/preflight-receipt.json`
(SHA-256 `90c4a08bb28c9f513c7a891c38aa5618b38a49f7299f72b2d5c10cca1b21fd28`):
each row had exactly one ambiguity and one source abstention, 42 hits, 84
accepted source tokens, and zero source rejection. No raw source material is
retained in that receipt.

## Predeclared V4 study

- Study: `synthetic-current-source-long-v4-9c7de25f`.
- Preserve all 16 positive, four mismatch, and four no-source rows byte for
  byte from V3.
- Replace only the four ambiguity source vectors using the source-only
  deterministic offset-64 construction above.
- Retain `pcss.ram_mib=64`, the policy, K=2 cap, model/server, sampling,
  arm order, clusters, thresholds, source RAM share, and claim boundary.
- Add exactly one finite control parameter: `ambiguity_trigger_offset_tokens:
  [64]`. No source or RAM grid sweep is allowed.
- One three-arm run after preflight; a repeat requires a distinct evidenced
  infrastructure or causal implementation defect.

V4 remains a synthetic capability study only and cannot support a
representative, general, default-enable, persistent-history, or best-in-class
claim.

## V4 result

`synthetic-current-source-long-v4-9c7de25f` completed all 84 frozen arm
requests with exact PCSS/baseline output, clean process-group receipts,
redacted source-bearing artifacts, and private `0700` PCSS stores. The one
ambiguity construction behaved exactly as preflighted: four total ambiguity
abstentions and four total mismatch abstentions, with 20 hit requests, 1,680
accepted source draft tokens, and zero rejections.

All preregistered effect gates passed. Baseline-over-PCSS was
`1.0666367292290648`, with clustered 95% interval
`[1.064369322375595, 1.0692194794528975]`; the frozen primary LCB threshold
was `1.05`. This confirms a genuine local synthetic current-source gain, while
preserving the stated non-representative claim boundary.
