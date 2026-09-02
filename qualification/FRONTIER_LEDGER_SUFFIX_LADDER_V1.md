# PCSS exact-suffix-ladder v1 frontier ledger

> **HISTORICAL / ARCHIVE-ONLY:** This file belongs to the 2026-08 PCSS sequence line on llama.cpp, which is closed (see `docs/CLOSED_LINES.md`). Its rules, gates, thresholds, budgets and stop rules bind nothing now. The current line is defined only by `NOW.md`.

This is a new, bounded calibration study. It does not modify, replace, or
reinterpret any prior frozen study or receipt.

## Causal hypothesis

The 64-token-only policy did not encounter a persistent hit in the sealed V4
agent-assisted development stream. A deterministic exact suffix ladder can
recover useful local reuse after edits before the final 32 visible tokens while
preserving a longer 64-token match when it exists. The target verifier remains
the sole acceptance authority.

## Frozen candidate

- Policy specification: `policies/pcss-exact-suffix-ladder-v1.json`.
- Persistent keys: 64 then 32 visible tokens, longest exact match first.
- Learning: 16 evenly distributed successful-end anchors; at most two records
  per anchor and therefore at most 32 records per completed request.
- Continuation payload: at most 8 tokens; server proposal is capped at 2.
- Negative evidence: after three zero-token 32-token-prefix rejections in one
  request, suppress short-prefix lookup for that request and continue with a
  64-token-only lookup/current-context fallback.

## Allowed comparison

One newly captured, sealed agent-assisted calibration workload may be measured
once with baseline, upstream ngram-cache K=2 (reporting-only if it diverges),
and this PCSS policy. The workload must be captured before any arm result is
opened, run in chronological order, and use the immutable manifest binding the
policy file hash.

The study is project-local calibration only. It cannot support a
representative, general, default-enable, or terminal frontier claim.

## Stop rule

Stop on an exact-output, lifecycle, resource, or process-health kill gate; or
after each frozen arm consumes the sealed workload once. A result that fails
the practical calibration gate closes this suffix-ladder policy for that
workload and does not authorize another constants-only retry.

## Frozen result

The one permitted calibration was completed as
`synthetic-suffix-ladder-v1-2617641b` on patch
`2617641b8033107e84a90fe2dd1ea0acad5446e1ef867b678a13e95c0af8166a`.
Manifest SHA-256: `9d662c214c09ea4f9d6cc7f40621ab599b889698e497a597ed70a4c6816981a1`.
Raw result SHA-256: `d47b6fe1c660d21292adea14984fb2e8988e245ac66a212ae440f5ebd7c78d5d`.
Independent analysis SHA-256:
`1726ca2a7ba5f9dd4a3a8d6b359e6eba9cf08ecd1252c4b750d7512ace54f132`.

The raw run was complete and exact baseline--PCSS equality passed 12/12. The
32-token tier activated in four episodes, with six hit requests and 72
target-accepted source-tier tokens; there were no 64-token hits, short-tier
suppressions, or fallback disables. Baseline/PCSS end-to-end estimate was
`1.02678`, but clustered 95% interval `[1.00952, 1.04089]` did not meet the
frozen 1.05 lower-bound gate. Miss-heavy overhead UCB95 was `0.00649`, within
the 3% ceiling. This closes this synthetic suffix-ladder policy/workload
combination without a retry. It establishes only exact synthetic capability,
not a representative or general gain.
