# PCSS host-independent foundation

This repository contains the offline contracts needed to falsify or prepare
Persistent Chained Speculation Store (PCSS) experiments without asserting a
runtime result. It deliberately does not attach a provider to target decoding,
measure an Apple M4, or claim a speedup.

## Reproducible environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Runtime modules use only the standard library. `requirements-dev.txt` pins the
test runner stack; `.python-version` declares the minimum development runtime.

## Data contracts

- `schemas/manifest-v1.schema.json` defines an evidence manifest. Unknown M4,
  model, and backend values remain explicitly `not_observed`; they must not be
  guessed.
- `schemas/trace-v1.schema.json` defines visible, committed token traces. The
  ordinary schema rejects hidden-reasoning fields.
- Python validators live in `pcss.manifest` and `pcss.trace`. They are
  fail-closed and generate stable fingerprints from canonical JSON.

All artifacts produced by later replay commands must identify their exact
command, Git revision, input fingerprint, and evidence class. Hardware-specific
results remain `NOT_RUN` until the target machine and backend gates are run.
