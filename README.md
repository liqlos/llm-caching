# llm-caching — datastore speculative decoding on MLX (closed 2026-09-03)

**Line D is closed as NO_GO.** A persistent, cross-session datastore of
previously generated and seen token sequences was built and measured
against the per-request lookup mlx-dspark already ships, on the owner's
real agentic-coding traffic. It does not beat per-request lookup by the
pre-registered margin (1.014x vs. a 1.20x threshold) — the owner's long
coding sessions self-saturate a per-request drafter before persistence
across sessions can add much. See
[docs/CLOSED_LINES.md](docs/CLOSED_LINES.md) for the full record and
[NOW.md](NOW.md) for the decision trail. `GlobalDatastore` stays in the
fork (`liqlos/mlx-dspark`) as a working, correctness-checked, opt-in
feature; no further engineering is planned here.

Prefix / prefill KV caching is **out of scope**: oMLX already provides it
for Apple Silicon. See [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md) for
the two prefix-cache lines this project built, measured, and retired.

- Plan, gates, and execution rules: [NOW.md](NOW.md)
- Closed lines and their verdicts: [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md)

## Layout

All of the below is now historical: it was written for Line D while it was
open, and Line D closed 2026-09-03 (see `docs/CLOSED_LINES.md`). Kept for
reference and because the underlying techniques (offline replay, receipts,
admission/eviction simulation) may be reusable by a future, separately
scoped line — none of it is a live plan.

- `src/pcss/` — the laboratory (Python), Line D prior art: datastore
  admission/eviction simulation (`adaptive_datastore*`), a
  crash-recoverable checksummed disk store (`persistent_store`), byte
  accounting (`capacity`), model-free chronological replay and offline
  metrics (`replay`, `metrics`, `experiments`, `state_machine_simulator`,
  `offline_runner`), sealed token archives and receipts (`archive`,
  `importer`, `trace`, `manifest`, `canonical`, `results`), and workload
  sealing (`project_workload_capture`, `project_capture_proxy`).
- `tests/`, `scripts/check.sh` — lab checks (`ruff` + `pytest`).
- `docs/research/` — research memos behind Line D's now-closed strategy.
- `qualification/`, `fixtures/`, `schemas/`, `PCSS_PRODUCT.md`,
  `M4_RUNBOOK.md` — historical PCSS evidence and protocols; bind nothing.
- `artifacts/` — run roots and receipts (mostly untracked, on the owner's
  machines).

## Checks

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
bash scripts/check.sh
```

Live model runs happen only on the dedicated M4 host (see the execution
host rule in `NOW.md`).
