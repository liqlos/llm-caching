# Future M4 verifier procedure

This is a prepared, fail-closed procedure for the target MacBook Pro M4 with
24 GB unified memory. It is not a result, a hardware receipt, or evidence of
correctness, speedup, provider economics, memory safety, Metal behavior, or
thermals. No command in this file has been run against that target.

The current harness is a correctness boundary only. It runs a named suite
through an external VerifierAdapter; it does not attach a provider to live
decode, measure time, choose a beneficial K, or measure resource interference.
Preserve every raw result, including FAILED and NOT_RUN. Do not rerun unchanged
inputs merely to obtain a passing result.

## Current interface facts

The current entry point is:

~~~sh
.venv/bin/python -m pcss.verifier_runner \
  --adapter MODULE:CALLABLE \
  --suite k-sweep|perfect-draft|provider-floor|forced-rejection|long-mixed \
  --fixture PATH --manifest PATH --output PATH \
  [--report PATH] [--repository-root PATH]
~~~

MODULE:CALLABLE must construct an object with name, availability(), and
verify(case). Only an adapter whose availability reports
execution_class == "TARGET_BACKEND" can produce a LOCAL_MEASURED receipt.
ScriptedFixtureAdapter is intentionally OFFLINE_FIXTURE; UnavailableAdapter is
intentionally NOT_RUN. Neither is an M4 substitute.

The repository's current fixtures/verifier/offline_token_state_cases.json is
not a runner input: it has schema pcss.verifier-fixture/v1, while the runner
requires pcss.m4-verifier-cases/v1. Do not rename, coerce, or pass that offline
fixture to the target CLI.

## Required inputs

Before any target execution:

1. The physical target is the intended 24 GB M4 MacBook Pro. Record observed
   hardware and OS facts; do not copy values from another Mac.
2. Use a clean, committed repository revision. Record its real Git revision;
   do not type a guessed hash.
3. Create one actual, complete observed manifest for this run. The unfilled
   baseline manifest is invalid for a TARGET_BACKEND receipt.
4. Provide an external adapter for the exact target model/runtime. It must
   refuse to claim TARGET_BACKEND when it cannot make the required target-state
   comparison.
5. Create distinct M4 fixtures for the five suites below. They are run inputs,
   not placeholder process documents; their SHA-256 fingerprints are recorded
   in raw receipts.

### Observed manifest gate

The manifest must validate as pcss.manifest/v1 and contain actual observed
values for:

- target.hardware and target.operating_system;
- model_artifact, including artifact URI and SHA-256;
- quantization, tokenizer, chat_template, and generation_protocol;
- llama_cpp, including the actual revision and build flags in parameters;
- context_kv; and
- benchmark, describing the concrete harness configuration but not an invented
  speed result.

For every item above, observation_status is observed only after target
observation. The M4 manifest evidence_class is LOCAL_MEASURED. The runner
rejects a target adapter if any required field remains not_observed.

### Target fixture gate

Each fixture passed to the runner is exactly:

~~~text
{
  "schema_version": "pcss.m4-verifier-cases/v1",
  "cases": [ ... ]
}
~~~

Every case has exactly these fields:

~~~text
case_id
draft_tokens
target_tokens
expected_committed_tokens
expected_state_digest
k
include_bonus_on_full_accept
~~~

k equals the number of draft tokens. Every token ID, target continuation,
committed-token expectation, and state digest comes from the exact observed
model/tokenizer/template/protocol in the manifest. Do not put hidden reasoning
in a case. Generate expected_state_digest with an exact non-speculative
reference path for the same initial state, never with the offline fixture
digest helper.

The adapter state digest covers every mutable component that can change later
target behavior: token position/history, KV or recurrent state, and any
backend-specific decode state relevant to the next target token. Correction and
full-accept bonus tokens must affect both committed_tokens and the post-commit
state digest.

Use separate fixtures because the contracts differ:

- k-sweep contains at least two actual k values selected for the future run.
  It proves only state/commit behavior at those values; it does not select a
  profitable K.
- perfect-draft contains only drafts equal to the first k target tokens.
  Include a full-accept bonus case: target has one further token and expected
  committed tokens include it.
- provider-floor contains provider-boundary correctness cases. The current CLI
  has no deadline, latency, cost, or provider-floor metric, so its receipt is
  not verifier-economics evidence.
- forced-rejection contains, for every represented k, a first mismatch at every
  position from 0 through k - 1. Expected committed tokens are the accepted
  draft prefix plus the target correction token; a rejection has no bonus.
- long-mixed contains the adapter's declared long mixed accept/reject behavior
  and its final reference-state digest.

VerifierCase currently has no prompt or initial-state field, and
long_mixed_trajectories invokes independent cases. It cannot by itself prove
non-empty-context restoration or multi-cycle shared-state continuity. If the
external adapter cannot encode and compare those semantics inside each case,
leave that aspect NOT_RUN and extend the harness before making a general
hybrid-state correctness claim.

## Future target preflight

Run this block only after creating real target inputs. Its variables prevent
this runbook from fabricating a run ID, revision, adapter, fixture, model, or
hardware value.

~~~sh
cd /path/to/llm-caching
export PCSS_REPO="$PWD"
export PCSS_PYTHON="$PCSS_REPO/.venv/bin/python"

test -n "$PCSS_RUN_ID" || { echo "set an actual PCSS_RUN_ID first" >&2; exit 64; }
test -n "$PCSS_ADAPTER" || { echo "set external MODULE:CALLABLE first" >&2; exit 64; }

export PCSS_RUN_ROOT="$PCSS_REPO/artifacts/m4/$PCSS_RUN_ID"
export PCSS_OBSERVED_MANIFEST="$PCSS_RUN_ROOT/manifest-observed.json"
export PCSS_K_SWEEP_FIXTURE="$PCSS_RUN_ROOT/fixtures/k-sweep.json"
export PCSS_PERFECT_DRAFT_FIXTURE="$PCSS_RUN_ROOT/fixtures/perfect-draft.json"
export PCSS_PROVIDER_FLOOR_FIXTURE="$PCSS_RUN_ROOT/fixtures/provider-floor.json"
export PCSS_FORCED_REJECTION_FIXTURE="$PCSS_RUN_ROOT/fixtures/forced-rejection.json"
export PCSS_LONG_MIXED_FIXTURE="$PCSS_RUN_ROOT/fixtures/long-mixed.json"

test -x "$PCSS_PYTHON" || { echo "missing reproducible Python environment" >&2; exit 64; }
git -C "$PCSS_REPO" diff --quiet || { echo "refuse dirty repository" >&2; exit 65; }
export PCSS_GIT_REVISION="$(git -C "$PCSS_REPO" rev-parse HEAD)"
mkdir -p "$PCSS_RUN_ROOT/raw" "$PCSS_RUN_ROOT/reports" "$PCSS_RUN_ROOT/fixtures"

"$PCSS_PYTHON" -c 'import pcss.verifier_runner; print("verifier_runner import: OK")'
"$PCSS_PYTHON" -m pytest tests/test_verifier.py tests/test_verifier_runner.py
~~~

The manifest run_id must exactly equal PCSS_RUN_ID. Validate it before loading
the adapter or running a suite:

~~~sh
"$PCSS_PYTHON" - <<'PY'
import json
import os
from pcss.manifest import validate_manifest

with open(os.environ["PCSS_OBSERVED_MANIFEST"], encoding="utf-8") as handle:
    manifest = json.load(handle)
validate_manifest(manifest)
if manifest["run_id"] != os.environ["PCSS_RUN_ID"]:
    raise SystemExit("manifest run_id does not match PCSS_RUN_ID")
if manifest["evidence_class"] != "LOCAL_MEASURED":
    raise SystemExit("target manifest evidence_class must be LOCAL_MEASURED")
fields = (
    ("target.hardware", manifest["target"]["hardware"]),
    ("target.operating_system", manifest["target"]["operating_system"]),
    ("model_artifact", manifest["model_artifact"]),
    *((name, manifest[name]) for name in (
        "quantization", "tokenizer", "chat_template", "generation_protocol",
        "llama_cpp", "context_kv", "benchmark",
    )),
)
for name, value in fields:
    if value["observation_status"] != "observed":
        raise SystemExit(f"unobserved target manifest field: {name}")
print("observed manifest: OK")
PY
~~~

Validate runner schema and the declared correction/bonus coverage. This is
stricter than merely opening JSON:

~~~sh
"$PCSS_PYTHON" - <<'PY'
import os
from pathlib import Path
from pcss.verifier_runner import load_cases

paths = {
    "k-sweep": os.environ["PCSS_K_SWEEP_FIXTURE"],
    "perfect-draft": os.environ["PCSS_PERFECT_DRAFT_FIXTURE"],
    "provider-floor": os.environ["PCSS_PROVIDER_FLOOR_FIXTURE"],
    "forced-rejection": os.environ["PCSS_FORCED_REJECTION_FIXTURE"],
    "long-mixed": os.environ["PCSS_LONG_MIXED_FIXTURE"],
}
cases = {name: load_cases(Path(path)) for name, path in paths.items()}

if len({case.k for case in cases["k-sweep"]}) < 2:
    raise SystemExit("k-sweep requires at least two actual k values")

for case in cases["perfect-draft"]:
    if case.draft_tokens != case.target_tokens[:case.k]:
        raise SystemExit(f"perfect-draft case is not perfect: {case.case_id}")
    expected = case.draft_tokens
    if case.include_bonus_on_full_accept:
        if len(case.target_tokens) <= case.k:
            raise SystemExit(f"bonus token missing: {case.case_id}")
        expected += (case.target_tokens[case.k],)
    if case.expected_committed_tokens != expected:
        raise SystemExit(f"perfect-draft commit expectation invalid: {case.case_id}")

forced_by_k = {}
for case in cases["forced-rejection"]:
    if len(case.target_tokens) < case.k:
        raise SystemExit(f"short target sequence: {case.case_id}")
    mismatch = next((i for i in range(case.k)
                     if case.draft_tokens[i] != case.target_tokens[i]), None)
    if mismatch is None:
        raise SystemExit(f"forced-rejection case has no rejection: {case.case_id}")
    if case.include_bonus_on_full_accept:
        raise SystemExit(f"rejected case requests a bonus: {case.case_id}")
    expected = case.draft_tokens[:mismatch] + (case.target_tokens[mismatch],)
    if case.expected_committed_tokens != expected:
        raise SystemExit(f"correction expectation invalid: {case.case_id}")
    forced_by_k.setdefault(case.k, set()).add(mismatch)
for k, positions in forced_by_k.items():
    if positions != set(range(k)):
        raise SystemExit(f"rejection coverage incomplete for k={k}: {sorted(positions)}")

for name, suite_cases in cases.items():
    if not suite_cases:
        raise SystemExit(f"empty fixture: {name}")
print("runner fixtures and declared correction/bonus coverage: OK")
PY
~~~

## Target-suite sequence

The function resolves the repository's actual Git HEAD and records its parsed
invocation in the receipt command field. It accepts neither a caller-supplied
revision nor a caller-supplied command string.

~~~sh
run_suite() {
  suite="$1"
  fixture="$2"
  stem="$3"
  raw="$PCSS_RUN_ROOT/raw/$stem.json"
  report="$PCSS_RUN_ROOT/reports/$stem.md"

  "$PCSS_PYTHON" -m pcss.verifier_runner \
    --adapter "$PCSS_ADAPTER" \
    --suite "$suite" \
    --fixture "$fixture" \
    --manifest "$PCSS_OBSERVED_MANIFEST" \
    --output "$raw" \
    --report "$report" \
    --repository-root "$PCSS_REPO"
}

run_suite k-sweep "$PCSS_K_SWEEP_FIXTURE" 01-k-sweep
run_suite perfect-draft "$PCSS_PERFECT_DRAFT_FIXTURE" 02-perfect-draft
run_suite provider-floor "$PCSS_PROVIDER_FLOOR_FIXTURE" 03-provider-floor
run_suite forced-rejection "$PCSS_FORCED_REJECTION_FIXTURE" 04-forced-rejection
run_suite long-mixed "$PCSS_LONG_MIXED_FIXTURE" 05-long-mixed
~~~

Expected artifacts, after a future execution, are:

~~~text
artifacts/m4/<actual-run-id>/manifest-observed.json
artifacts/m4/<actual-run-id>/fixtures/{k-sweep,perfect-draft,provider-floor,forced-rejection,long-mixed}.json
artifacts/m4/<actual-run-id>/raw/01-k-sweep.json
artifacts/m4/<actual-run-id>/raw/02-perfect-draft.json
artifacts/m4/<actual-run-id>/raw/03-provider-floor.json
artifacts/m4/<actual-run-id>/raw/04-forced-rejection.json
artifacts/m4/<actual-run-id>/raw/05-long-mixed.json
artifacts/m4/<actual-run-id>/reports/01-k-sweep.md through 05-long-mixed.md
~~~

Every raw result is canonical pcss.result/v1 JSON: exact command, Git
revision, SHA-256 fixture/manifest fingerprints, evidence class, status, and
the harness receipt. Compact reports are derived only after their matching raw
result exists. A FAILED or NOT_RUN raw output remains evidence and must not be
overwritten.

After every suite, check the receipt rather than treating a zero shell exit as
acceptance:

~~~sh
check_receipt() {
  "$PCSS_PYTHON" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    receipt = json.load(handle)
payload = receipt.get("payload", {})
if receipt.get("schema_version") != "pcss.result/v1":
    raise SystemExit("wrong raw-result schema")
if receipt.get("evidence_class") != "LOCAL_MEASURED":
    raise SystemExit("receipt is not target-local measured evidence")
if receipt.get("status") != "VERIFIED":
    raise SystemExit(f"receipt status is not VERIFIED: {receipt.get('status')}")
if payload.get("execution_class") != "TARGET_BACKEND":
    raise SystemExit("adapter did not identify a target backend")
if payload.get("status") != "VERIFIED" or not payload.get("cases"):
    raise SystemExit("harness did not verify target cases")
for case in payload["cases"]:
    if case.get("status") != "VERIFIED" or case.get("state_matches") is not True or case.get("failures"):
        raise SystemExit(f"case failed state/commit check: {case.get('case_id')}")
print("target receipt gate: OK")
PY
}

check_receipt "$PCSS_RUN_ROOT/raw/01-k-sweep.json"
check_receipt "$PCSS_RUN_ROOT/raw/02-perfect-draft.json"
check_receipt "$PCSS_RUN_ROOT/raw/03-provider-floor.json"
check_receipt "$PCSS_RUN_ROOT/raw/04-forced-rejection.json"
check_receipt "$PCSS_RUN_ROOT/raw/05-long-mixed.json"
~~~

The harness checks committed tokens and state digests but does not separately
prove that an adapter's correction_token or bonus_token field is internally
consistent. Check those fields against the fixtures:

~~~sh
"$PCSS_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

fixture = json.loads(Path(os.environ["PCSS_FORCED_REJECTION_FIXTURE"]).read_text())
raw = json.loads(Path(os.environ["PCSS_RUN_ROOT"]).joinpath(
    "raw/04-forced-rejection.json").read_text())
cases = {case["case_id"]: case for case in fixture["cases"]}
for result in raw["payload"]["cases"]:
    case = cases[result["case_id"]]
    accepted = result["accepted_draft_tokens"]
    if result["correction_token"] is None or result["bonus_token"] is not None:
        raise SystemExit(f"invalid rejected-token fields: {result['case_id']}")
    committed = case["draft_tokens"][:accepted] + [result["correction_token"]]
    if committed != case["expected_committed_tokens"]:
        raise SystemExit(f"correction field disagrees: {result['case_id']}")

fixture = json.loads(Path(os.environ["PCSS_PERFECT_DRAFT_FIXTURE"]).read_text())
raw = json.loads(Path(os.environ["PCSS_RUN_ROOT"]).joinpath(
    "raw/02-perfect-draft.json").read_text())
cases = {case["case_id"]: case for case in fixture["cases"]}
for result in raw["payload"]["cases"]:
    case = cases[result["case_id"]]
    if result["correction_token"] is not None:
        raise SystemExit(f"perfect draft emitted correction: {result['case_id']}")
    if case["include_bonus_on_full_accept"]:
        if result["bonus_token"] is None:
            raise SystemExit(f"missing bonus token: {result['case_id']}")
        committed = case["draft_tokens"] + [result["bonus_token"]]
    else:
        if result["bonus_token"] is not None:
            raise SystemExit(f"unexpected bonus token: {result['case_id']}")
        committed = case["draft_tokens"]
    if committed != case["expected_committed_tokens"]:
        raise SystemExit(f"bonus field disagrees: {result['case_id']}")
print("correction/bonus field consistency: OK")
PY
~~~

## Kill gates and interpretation

Stop the target run and preserve raw artifacts if any of these occurs:

- the machine is not the intended M4 target, the manifest is incomplete, the
  worktree is dirty, or revision/input fingerprints do not match the run;
- the adapter reports NOT_RUN or OFFLINE_FIXTURE, cannot identify
  TARGET_BACKEND, cannot reset/compare the declared exact state, or any case
  reports a commit/state mismatch;
- a fixture fails schema/coverage checks, correction/bonus post-check, or does
  not come from the exact recorded model/tokenizer/template/protocol;
- the adapter cannot establish non-empty-context or multi-cycle long-mixed
  state equivalence under the present one-case API; that aspect is NOT_RUN
  until the contract is extended;
- a provider-floor receipt is used to infer a deadline, latency, cost, or
  verifier economics. The current CLI has no such fields, so the economic gate
  remains not run even if its correctness receipt verifies;
- anyone infers a good K, live speculative-decoding correctness, speedup,
  memory pressure, Metal behavior, swap/page-fault behavior, or thermals from
  these receipts.

Passing target receipts establish only the narrow state/commit checks exercised
by the actual adapter and fixtures. They do not unlock live speculation or
runtime claims. Those require their own M4-only gates and raw artifacts after
this verifier boundary is genuinely complete.
