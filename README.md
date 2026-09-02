# llm-caching — datastore speculative decoding on MLX

Decode-side acceleration for local LLM serving on the owner's Apple
Silicon machines: a persistent datastore of previously generated and seen
token sequences proposes multi-token drafts during generation, which the
target model verifies exactly in one forward pass. Output stays
bit-identical; the model emits more tokens per weight pass. Built on MLX
into the existing Apple Silicon speculation stack (mlx-dspark, then oMLX),
measured on the owner's real agentic-coding traffic.

Prefix / prefill KV caching is **out of scope**: oMLX already provides it
for Apple Silicon. See [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md) for
the two prefix-cache lines this project built, measured, and retired.

- Plan, gates, and execution rules: [NOW.md](NOW.md)
- Closed lines and their verdicts: [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md)

## Layout

- `src/pcss/` — the laboratory (Python), prior art kept for Line D:
  datastore admission/eviction simulation (`adaptive_datastore*`), a
  crash-recoverable checksummed disk store (`persistent_store`), byte
  accounting (`capacity`), model-free chronological replay and offline
  metrics (`replay`, `metrics`, `experiments`, `state_machine_simulator`,
  `offline_runner`), sealed token archives and receipts (`archive`,
  `importer`, `trace`, `manifest`, `canonical`, `results`), and workload
  sealing (`project_workload_capture`, `project_capture_proxy`).
- `tests/`, `scripts/check.sh` — lab checks (`ruff` + `pytest`).
- `docs/research/` — research memos behind `NOW.md`'s strategy.
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
