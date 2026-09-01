# SSD L2 current handoff

`docs/SSD_L2_CACHE.md` is the sole current-state authority.
All other PCSS and sequence documents are historical evidence only.

## Gate G1 verdict: `NO_GO_SSD_L2_PRODUCT` (Phase 1 complete)

Phase 1 ran. The persistent SSD tier never engaged for representative
solo-developer traffic (`--parallel 1`), on either a generous or a minimal
`--cache-ram` budget: 0 bytes ever written to the disk store, in two
independent attempts. Root cause is architectural, not a tuning miss: the
only save-to-cache trigger fires for a slot going idle while a *different*
slot is busy — a multi-slot mechanism that structurally cannot activate with
one sequential slot. Full receipts and both attempts:
`artifacts/ssd-redesign/20260901T152546Z-4c0f7eab23ef/RUN_SUMMARY.md`;
reflected in `docs/SSD_L2_CACHE.md`. **Phase 2 is not entered.** The
learning-write overhead control (previously open) is now closed — see the
same run summary.

## Proven scope and results

- Native RAM plus optional SSD L2 prompt/KV caching is implemented.
- SSD L2 remains disabled by default.
- The caller-attested synthetic M4 exact-reuse ratio is `4.9897x`.
- The caller-attested synthetic M4 partial-reuse ratio is `1.5825x`.
- Exact and partial qualification outputs and generated token IDs matched.
- Novel input establishes only no detected latency difference.
- Pressure proves bounded capacity, eviction, and restart for the four-prefix probe.
- Pressure does not prove output or generated-token equality.
- The compatibility key remains caller-attested, not caller-derived.
- Learning-write overhead: closed by the Phase 1 run (no reproducible net
  cost beyond noise once the tier has nothing to write); see verdict above.
- Representative persistent reuse: `NO_GO_SSD_L2_PRODUCT` (Phase 1, above).
  The multi-slot-only save trigger is the concrete gap for an upstream
  contribution to close.

## Claim and execution boundary

- Do not generalize synthetic results to representative owner traffic.
- Do not claim universal speedup, production economics, or default enablement.
- Do not repeat unchanged qualification, novel, learning, or pressure experiments.

## Goal invariant (does not change)

Deliver honest, receipt-backed evidence of whether persistent SSD L2 caching
produces real value on representative owner traffic. The evidence bar never
drops: raw receipts, explicit claim boundaries, paired controls where a cost
is claimed, no synthetic-to-representative generalization. A clean,
receipt-backed negative result satisfies the goal; an unsupported positive
claim does not.

## Ordered next plan (gated)

The decisive unknown is representative persistent reuse, not mechanism
hardening. Answer it first; harden only if it pays.

### Phase 0 — Define the target workload (cheap, do first)

Write one short section (in the Phase 1 run directory) that defines, before
any measurement: who the owner is, what traffic repeats (agent system
prompts, project context, multi-turn sessions), and how often the server
restarts with a cold RAM tier. This definition is frozen before Phase 1 runs
and is the standard "representative" is judged against.

### Phase 1 — Decisive representative experiment — **complete, see verdict above**

Run a representative long multi-session workload against the Phase 0
definition. Measure persistent (post-restart, cold-RAM) exact and partial
hit rates and paired wall-clock. Attach the paired learning-write overhead
control to the same run. The caller-attested compatibility key is acceptable
here: the experiment controls its own environment, and identity derivation
is a productization concern, not a go/no-go evidence concern.

A trimmed replay of real traces measuring hit rate only is a valid first
iteration if it is cheaper than the full paired-latency design.

### Gate G1 — decide from Phase 1 receipts

- Persistent hit rate is materially nonzero and paired wall-clock benefit is
  plausible: `GO_HARDENING` — proceed to Phase 2.
- Persistent hit rate is ~zero or benefit is not plausible:
  `NO_GO_SSD_L2_PRODUCT` — record the receipt-backed negative, keep SSD L2 as
  a qualified disabled-by-default feature, and evaluate an upstream llama.cpp
  contribution as the exit path. Do not enter Phase 2.

Record the gate verdict and its receipts in the run directory and reflect it
in `docs/SSD_L2_CACHE.md`. Either verdict is a successful outcome of Phase 1.

### Phase 2 — Hardening (only after `GO_HARDENING`) — **not entered: verdict was `NO_GO_SSD_L2_PRODUCT`**

1. Implement caller-derived fail-closed compatibility identity.
2. Establish restart-persistent recency.
3. Test larger-cardinality pressure with raw output and token receipts.

## Anti-stall rules (binding on the executing agent)

- **Two-strike loop-breaker.** If the same experiment design fails or stalls
  twice without a new hypothesis, do not run it a third time. Either change
  the design under a written new hypothesis, or record the blocker in the
  run journal and move to the next runnable item. Repetition without a new
  hypothesis is the failure mode that produced the v19/v16 rerun churn —
  treat it as prohibited, not merely discouraged.
- **Approach freedom within the evidence bar.** The agent is authorized to
  substitute an alternative experimental design, harness, or measurement
  approach for any planned item without asking, provided the Goal invariant
  holds and the substitution is recorded with its rationale. The plan binds
  the questions and the gate order, not the specific mechanism of answering.
- **Negative results are progress.** Declaring a receipt-backed NO-GO at
  Gate G1 (or a blocker on any item) is a completed deliverable, not a
  failure. Do not keep iterating in search of a positive result.
- **Timebox before pivot.** If a planned item consumes a full session
  without producing either a receipt or a concrete blocker, stop, journal
  the state honestly, and pivot to the cheapest design that can still
  answer the same question.
- **No silent scope changes.** Reordering within a phase and substituting
  designs are allowed; skipping Gate G1 or entering Phase 2 without a
  `GO_HARDENING` verdict is not.

The sequence datastore remains `NO_GO_CURRENT_REDESIGN`.
Historical PCSS documents do not define or authorize next work.
