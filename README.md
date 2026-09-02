# llm-caching — datastore speculative decoding on MLX

Decode-side acceleration for local LLM serving on the owner's Apple
Silicon machines: a persistent datastore of previously generated and seen
token sequences proposes multi-token drafts during generation, which the
target model verifies exactly in one forward pass. Output stays
bit-identical; the model emits more tokens per weight pass. Built on
MLX / mlx-lm, measured on the owner's real agentic-coding traffic.

Prefix / prefill KV caching is **out of scope**: oMLX already provides it
for Apple Silicon. See [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md) for
the two prefix-cache lines this project built, measured, and retired.

- Plan, gates, and execution rules: [NOW.md](NOW.md)
- Closed lines and their verdicts: [docs/CLOSED_LINES.md](docs/CLOSED_LINES.md)

## Layout

- `src/pcss/` — the laboratory (Python): corpus builders, paired benchmark
  harnesses, receipts. Modules named `llamacpp_*`, `exact_prompt_*`,
  `live_server_*`, `verifier*`, `provider*` are historical prior art from
  the 2026-08 PCSS line.
- `tests/`, `scripts/check.sh` — lab checks (`ruff` + `pytest`).
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
