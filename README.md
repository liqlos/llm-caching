# PCSS host-independent foundation

This repository contains the offline contracts needed to falsify or prepare
Persistent Chained Speculation Store (PCSS) experiments. It also contains an
isolated llama.cpp correctness bridge and perfect-draft verifier profiler. It
does not attach a provider to live target decoding or claim an end-to-end
speedup.

## Reproducible environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
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
receipts are valid only for their exact observed target, model, bridge,
llama.cpp build, and runtime configuration.

## Target evidence boundary

`M4_RUNBOOK.md` records the immutable Qwen3-8B Q4_K_M verifier and isolated
profile procedure. The current target receipt covers exact raw-token
correctness suites, paired decode-only perfect-draft timings, and deterministic
single-rejection controls. It excludes live PCSS/provider integration,
representative accepted-prefix workload distributions, sustained thermal
qualification, and any transferable speedup claim.

## Offline fixture receipt

The following command runs only the model-free chronological replay, sealed
temporary archive readback, and logical capacity simulator. It writes a
`SIMULATOR` receipt; it does not invoke a model, live decode, target verifier,
or timing measurement.

```sh
PYTHONPATH=src python3 -m pcss.offline_runner \
  --fixture fixtures/replay/chronological-oracle-v1.json \
  --rfc /Users/aleksei/Downloads/PCSS_MASTER_RFC_EN.md \
  --output artifacts/offline/chronological-oracle.raw.json \
  --report artifacts/offline/chronological-oracle.report.md
```

The command resolves Git `HEAD` and records its generated invocation itself;
the future verifier command uses the same receipt boundary. Callers cannot
supply a different revision or command string. The source RFC stays outside
this repository and is fail-closed against the pinned
`sha256:1dfdc69b055af5059021d130e216ed702a9636fb585cc5c597be783382d5e357`.
