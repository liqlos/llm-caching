# PCSS native llama.cpp product

This repository contains the original offline evidence contracts and an active,
pinned llama.cpp product path. `pcss-cache` runs inside llama-server's native
speculative loop, uses only visible-token candidates, and can persist accepted
completed request sequences in an SSD-backed store. The legacy bridge remains a correctness oracle,
not the production architecture. Persistent PCSS is experimental and disabled
by default: no end-to-end speedup is claimed without qualification evidence.

## Working policy

Optimize for the requested measured result, not activity. Plans, documents,
commits, green tests, running processes, and receipts are supporting evidence;
none is completion unless it directly satisfies the requested outcome. Work on
the highest-impact unresolved uncertainty first.

Experimental gates are scoped to the run, implementation path, or claim they
validate; they do not stop the broader work session. On failure, preserve the
immutable evidence, change the method or input, and continue toward the original
objective. Keep process minimal: add formal contracts, reviews, documents, or
harnesses only when they protect a concrete evidence boundary or acceptance
criterion. Resolve ambiguity in favor of safe, reversible local progress. Ask
the owner only for a genuine external authority or input boundary. See
`M4_RUNBOOK.md` for the full autonomous continuation policy.

## Reproducible environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -e .
bash scripts/check.sh
bash scripts/native-check.sh
```

For a local OpenAI-compatible server, set `PCSS_MODEL` to an existing GGUF and
run `bash scripts/run_server.sh`. See [PCSS_PRODUCT.md](PCSS_PRODUCT.md) for
store controls and the current qualification boundary.

The run script verifies the model SHA-256, keeps the store private by default,
uses a required single-principal namespace (`local-uid-<uid>` unless explicitly
overridden), and enables live `/metrics`. If any extra server option is passed,
the script requires an explicit `PCSS_FINGERPRINT`; this prevents template or
thinking-policy changes from silently reusing a store.

Store and benchmark operations:

```sh
PCSS_STORE=/absolute/store bash scripts/store-inspect.sh
PCSS_MODEL=/absolute/model.gguf PCSS_STORE=/absolute/store bash scripts/store-compact.sh
PCSS_STORE=/absolute/store PCSS_CONFIRM_RESET=move-store-to-backup bash scripts/store-reset.sh
PCSS_SERVER=/absolute/llama-server PCSS_MODEL=/absolute/model.gguf \
  PCSS_MODEL_SHA256=<sha256> PCSS_PATCH_SHA256=<sha256> \
  PCSS_RESULT=/absolute/fresh-result.json bash scripts/real-model-check.sh
bash scripts/benchmark.sh MANIFEST.json WORKLOAD.json MODEL.gguf FRESH_OUTPUT_DIRECTORY
```

Reset is recoverable: it moves the exact store to a timestamped sibling rather
than deleting it. The benchmark refuses unfrozen inputs, future-output fields,
hash mismatches, unpreregistered arm changes, and output-directory reuse.

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
profile procedure. The current native product receipt additionally covers the
standard OpenAI API, successful-end learning, restart persistence, cancellation
suppression, same-slot long-to-short equality, bounded store inspection, and
live PCSS metrics. It still excludes representative accepted-prefix workload
distributions, sustained thermal qualification, and any transferable speedup
claim.

The overlapped v2 synthetic timing remains excluded. The later exclusive v4
three-arm raw control passes exact output and exercises PCSS capability; its
frozen analysis finds a positive baseline-over-PCSS interval on the synthetic
repeat pairs, while upstream ngram remains faster and the preregistered
miss-overhead gate has no qualifying sample. The final native patch adds
separately tested writer-lock recovery, crash-consistent compaction, peak disk
quota enforcement, bounded index checkpoints, and complete fallback telemetry
after v4. A later K=2 persistent-only exact-reuse calibration passed raw
correctness/resource cleanup, but its frozen parameter grid declared K=8 while
the arm executed K=2, so its performance/effect-gate verdict is not
qualification evidence. K=8 itself produced an immutable output-mismatch
receipt and is not used by default. Persistent PCSS remains disabled by default: no owner
representative holdout has been supplied, so the terminal status is
`IMPLEMENTED_NOT_QUALIFIED`.

The separate opt-in request-scoped current-source tier now has a valid local
synthetic capability result: `synthetic-current-source-long-v4-9c7de25f`
passed all frozen gates with baseline-over-PCSS LCB95 `1.0644` (estimate
`1.0666`), exact output, bounded source admission, zero source rejection, and
safe ambiguity/mismatch abstention controls. It is not representative or
general speedup evidence and does not change the disabled-by-default persistent
policy.

The later captured agent-assisted local calibration is also negative for its
own frozen stream: baseline--PCSS output equality holds, but zero persistent
hits and the missed confidence/overhead gates prevent a gain claim. It is not
the separate ordinary-user project-naturalistic protocol and does not establish
a general technical limit.

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
