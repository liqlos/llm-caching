# SSD L2 current handoff

`docs/SSD_L2_CACHE.md` is the sole current-state authority.
All other PCSS and sequence documents are historical evidence only.

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
- No clean learning-write overhead component has been identified.

## Claim and execution boundary

- Do not generalize synthetic results to representative owner traffic.
- Do not claim universal speedup, production economics, or default enablement.
- Do not repeat unchanged qualification, novel, learning, or pressure experiments.
- Current state is paused; no follow-on run is authorized by historical docs.

## Ordered next plan

1. Implement caller-derived fail-closed compatibility identity.
2. Establish restart-persistent recency.
3. Test larger-cardinality pressure with raw output and token receipts.
4. Run a paired learning-write overhead control.
5. Run a representative long multi-session workload.

The sequence datastore remains `NO_GO_CURRENT_REDESIGN`.
Historical PCSS documents do not define or authorize next work.
