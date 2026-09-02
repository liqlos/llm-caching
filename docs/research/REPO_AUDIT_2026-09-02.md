# Repo audit — 2026-09-02 (post-pivot to Line D)

Read-only audit of tracked files (`git ls-files`) plus the top-level layout,
against `NOW.md` (Line D: persistent cross-session datastore speculation on
MLX / mlx-dspark) and `docs/CLOSED_LINES.md`. Nothing was modified.

Baseline state: working tree clean, `git log` consistent (`adb1d18` closes the
prefix-cache lines, `89d15f8` sets the mlx-dspark strategy). `.github/workflows/ci.yml`
runs `requirements-dev.txt` + editable install + `scripts/check.sh`; that script
runs only `ruff check src tests` and `pytest`, both of which exist. Verified
locally: **346 passed, ruff clean**. No secrets, tokens, or injected/adversarial
instruction text found in any tracked file (JSON fixtures and schemas are pure
data).

## (a) Findings

| File / section | What is stale or misleading | Action |
|---|---|---|
| `artifacts/ssd-redesign/20260831T130127Z-84982163c750/continuous-journal/JOURNAL.md` | **Highest-risk item.** Tracked even though `.gitignore` excludes `artifacts/`. No HISTORICAL banner. Contains 20 imperative `Next action:` lines from the closed SSD L2 / PCSS lines ("define a bounded native aggregate-prefix plus TinyLFU experiment", "run a warmed, cache-pressure M4 benchmark"). L317 and L323 declare `docs/SSD_L2_CACHE.md` the "sole current-state authority" — that file **does not exist**. An agent reading only this file gets a closed line presented as the live plan. | `git rm --cached` (keep on disk, untracked like every other run root) — or, if kept, prepend a HISTORICAL banner and strike the `SSD_L2_CACHE.md` authority claim |
| `README.md:8` | "Built on MLX / mlx-lm". `NOW.md:44-48` says mlx-lm is **not a destination** ("Do not build on or target mlx-lm's speculative loop"); D1 builds into `mlx-dspark`. Direct contradiction in the first paragraph of the front door. | Rewrite: "Built on MLX, into `mlx-dspark`" |
| `README.md:19-27` (Layout) | Points to `src/pcss/` as "the laboratory" without saying that ~30 of its 48 modules are dead-line only, and does not mention `scripts/capture-project-proxy.sh` / `scripts/seal-project-workload.sh` (llama.cpp-era, still executable). | Rewrite Layout to name the live subset and mark the rest historical |
| `pyproject.toml:7-8` | `name = "pcss-foundation"`, description "Host-independent, falsification-first PCSS replay foundation". Project identity still advertises the closed PCSS line. | Rewrite metadata (cosmetic, but it is what `pip show` and tooling report) |
| `scripts/capture-project-proxy.sh` (whole) | Top-level executable with no historical marker; usage string requires `--origin {ordinary_local_llama_server\|agent_assisted_local_llama_server}`. Belongs to the retired llama.cpp proxy-capture protocol; `NOW.md:191-199` sources the corpus from `~/.claude/projects/` JSONL instead. Not referenced by CI, `check.sh`, or README. | Delete, or move under a `scripts/historical/` prefix |
| `scripts/seal-project-workload.sh` (whole) | Same: wraps `pcss.project_workload_capture` for the retired capture protocol, unreferenced. | Keep as prior art but move/mark — the sealing + hash-chain discipline is reusable for the D0 corpus |
| `qualification/FRONTIER_LEDGER.md` | **No banner.** Reads as live campaign law: "No fourth retrieval family is permitted in this campaign", a "Stop rule" section, eviction/storage/write-scheduling budgets ("At most two alternatives may be evaluated"). Would wrongly constrain D1's eviction and persistence choices. | Add HISTORICAL banner pointing to `NOW.md` / `docs/CLOSED_LINES.md` |
| `qualification/FRONTIER_LEDGER_CURRENT_SOURCE_V1.md` | No banner. "No K>2 variant is permitted by this study", "must bind its policy file SHA", frozen `1.05` LCB gate, `--pcss-enable-current-source` server flag (flag no longer exists). Line D's gates are 1.2x / 1.3x in `NOW.md`, and widths run to 32. | Add banner |
| `qualification/FRONTIER_LEDGER_CURRENT_SOURCE_RESOURCE_V1.md` | No banner. Frozen 16/64 MiB budgets, "must show nonzero source lookup/hit/accepted metrics at the same 64 MiB". Storage budgets bind nothing now (`NOW.md` uses byte quota + LRU, ~10 B/token). | Add banner |
| `qualification/FRONTIER_LEDGER_CURRENT_SOURCE_AMBIGUITY_V1.md` | No banner. Frozen LCB thresholds and control-cap rules stated as contract. | Add banner |
| `qualification/FRONTIER_LEDGER_SUFFIX_LADDER_V1.md` | No banner. "The workload must be captured before any arm result is opened", frozen 1.05 gate, K=2 cap. | Add banner |
| `qualification/AGENT_ASSISTED_CALIBRATION.md` | No banner. Claim-boundary rules ("cannot support a general, owner, default-enable, or technical-limit claim") stated in the present tense for `project-naturalistic-v1..v4`. | Add banner |
| `qualification/PROJECT_NATURALISTIC_CAPTURE.md:3-9` | **Has a banner, but the banner is itself stale**: it says the thresholds "do NOT bind SSD L2 work" — SSD L2 is now closed too (`CLOSED_LINES.md`). It also still points the reader at "The current owner-trace corpus rules are in `NOW.md`", which is correct, but the SSD L2 framing implies that line is live. | Rewrite banner to reference Line D, not SSD L2 |
| `PCSS_PRODUCT.md:17,24,39` | Banner is good (top-of-file, "entire file", names `NOW.md`). But the body gives runnable-looking commands for files that no longer exist: `bash scripts/native-check.sh`, `bash scripts/run_server.sh`, plus `third_party/llama.cpp` and `native/pcss_llama_bridge`. | Keep with banner; optionally annotate the removed paths inline |
| `M4_RUNBOOK.md` (1078 L; e.g. `:65,:117,:260,:672,:1042`) | Banner is good. Body is a long imperative procedure with concrete host paths (`PCSS_LLAMA_ROOT=/opt/homebrew/Cellar/llama.cpp/10520`), `native/CMakeLists.txt`, Homebrew build 10520. Length is the risk — an agent grepping mid-file may never see line 3. | Keep with banner; consider splitting or adding a repeated per-section marker |
| `schemas/llamacpp-fixture-seeds-v1.schema.json` | Schema for the removed `pcss_llama_bridge` fixture seeds. Undated, unmarked, sits beside the two still-generic schemas. | Delete with `llamacpp_fixture_builder.py`, or add a `"deprecated"` note |
| `fixtures/verifier/*.json`, `fixtures/replay/qwen3-8b-*.json` | Fixtures for the closed verifier / exact-prompt tests. Harmless (they only feed passing unit tests) but they are the only "qwen3-8b" artifacts in the tree and could be mistaken for Line D data. | Keep while their tests exist; delete together with them |
| `NOW.md:197` | References the untracked run root `artifacts/mlx/20260901T193508Z-phase-m0-baseline/` for corpus builders. Correct per the pinned host rule, but unreachable from a fresh clone. | Keep — flag only, and state the machine-local dependency |
| Repo root: no `CLAUDE.md` / `.claude/` is an empty dir | Nothing auto-loads `NOW.md`. An agent that opens `M4_RUNBOOK.md` or the tracked `JOURNAL.md` first has no pointer back to the current line. | Add a two-line project `CLAUDE.md` pointing at `NOW.md` + `docs/CLOSED_LINES.md` |
| Top-level `build/` (untracked, gitignored) | Contains `llama.cpp`, `llama.cpp-sanitize`, `native-current`, `native-current2` — leftover CMake trees for the removed native bridge. | Safe to delete locally; not a tracked-content issue |

## (b) Dead-module inventory (`src/pcss/`, 48 modules / `tests/` 49 files)

**Native llama.cpp bridge (binary `pcss_llama_bridge` removed in `027067b`) — cannot run:**

- `llamacpp_adapter.py` — fail-closed Python boundary for the removed bridge executable.
- `llamacpp_fixture_builder.py` — builds verifier fixtures by calling that bridge.
- `llamacpp_target_artifacts.py` — seed/manifest artifacts from the bridge capability probe.
- `llamacpp_profiler.py` — perfect-draft verifier profiler over the bridge.
- `llamacpp_live_benchmark.py` — best-case decode benchmark over the bridge pending protocol.
- `llamacpp_integrated_workload.py` — plan-bound integrated-workload executor for the bridge.
- `llamacpp_trajectory.py` — persistent exact-state pending-token trajectory on the bridge.
- `llamacpp_trace_workload.py` — token-only trace workload from a target llama.cpp model.

**`llama-server` harnesses (llama.cpp submodule removed):**

- `live_server_benchmark.py` (2601 L) — preregistered comparative runner against a live `llama-server`.
- `live_server_analysis.py` (1362 L) — strict offline analysis of that runner's receipt.
- `live_server_e2e.py` — correctness/lifecycle gates against a local `llama-server`.
- `live_current_source_e2e.py` — real-model lifecycle probe for the `pcss_current_source` request field.

**Exact-prompt / PCSS bridge-client line (whole-prompt-equality retrieval — subsumed by the closed prefix-cache verdict):**

- `exact_prompt_archive.py`, `exact_prompt_provider.py` — full-prompt-equality archive and its shadow provider.
- `exact_prompt_chain_runner.py`, `exact_prompt_roundtrip_harness.py`, `exact_prompt_live_harness.py`, `exact_prompt_external_trajectory.py` — four bridge-driven replay/decode harnesses.
- `exact_prompt_profiler.py`, `exact_prompt_economics.py`, `exact_prompt_policy_economics.py` — profiling and fail-closed economics composition over those receipts.
- `exact_repeat_trace.py` — cross-session exact-repeat trace control.

**Verifier / provider ABI (built for the absent llama.cpp adapter):**

- `verifier.py` — target-agnostic verification-harness contracts; ships only an offline fixture adapter.
- `verifier_runner.py` — fail-closed command boundary for a "future" target-backend verifier run.
- `verifier_preflight.py` — host/adapter observation that always records `NOT_RUN`.
- `provider.py`, `provider_profiler.py` — runtime-neutral provider lifecycle boundary and its microprofile.
- `economics.py` — verifier-economics verdict from receipts of the above.
- `integrated_run_plan.py`, `protocol_bakeoff.py`, `shadow_runner.py` — PCSS-specific run planning, protocol bakeoff, post-hoc shadow replay.

Each has a matching `tests/test_<name>.py`; the tests are pure-fixture and pass, so they add ~30 files of green-but-meaningless coverage.

**Genuine prior art for Line D (keep):**

- `adaptive_datastore.py` + `adaptive_datastore_study.py` — datastore admission/eviction policy simulation (LRU vs TinyLFU under a byte budget). Directly reusable for D1's suffix-tree quota + LRU and D2's tree-size-vs-gain curve.
- `persistent_store.py` — crash-recoverable checksummed-segment store with atomic manifest. Closest existing code to D1's on-disk datastore.
- `capacity.py` — byte-accounted logical active views; the accounting discipline behind "~10 B/token".
- `replay.py`, `metrics.py`, `experiments.py`, `state_machine_simulator.py`, `offline_runner.py` — model-free chronological replay and offline opportunity metrics. This is the shape D0a needs.
- `archive.py`, `importer.py`, `trace.py`, `manifest.py`, `canonical.py`, `results.py` — sealed token archives, strict JSONL import, versioned trace contracts, canonical JSON/SHA-256, receipt generation. The receipts/pre-registration machinery `NOW.md` still requires.
- `project_workload_capture.py`, `project_capture_proxy.py` — workload sealing (rejects responses, expected outputs, post-hoc labels) and hash-chained capture. The proxy's transport is llama.cpp-era, but the anti-tuning sealing rules transfer to the transcript corpus.

## (c) Safe to delete now / keep

**Safe to delete now** (nothing in `NOW.md`, CI, `check.sh`, or the live modules depends on them; all are unrunnable or line-specific):

- All eight `src/pcss/llamacpp_*.py` + their `tests/test_llamacpp_*.py`
- `src/pcss/live_server_benchmark.py`, `live_server_analysis.py`, `live_server_e2e.py`, `live_current_source_e2e.py` + their tests
- All nine `src/pcss/exact_prompt_*.py` and `exact_repeat_trace.py` + their tests
- `src/pcss/verifier.py`, `verifier_runner.py`, `verifier_preflight.py`, `provider.py`, `provider_profiler.py`, `economics.py`, `integrated_run_plan.py`, `protocol_bakeoff.py`, `shadow_runner.py` + their tests
- `schemas/llamacpp-fixture-seeds-v1.schema.json`; `fixtures/verifier/*`; `fixtures/replay/qwen3-8b-*.json`
- `scripts/capture-project-proxy.sh`
- Untrack `artifacts/.../continuous-journal/JOURNAL.md`
- Local only: the `build/` tree

Note: `exact_prompt_*` and the `llamacpp_*` harnesses import `provider.py` / `verifier*.py`, so delete the group as a unit or the imports break.

**Keep:**

- `NOW.md`, `docs/CLOSED_LINES.md`, `README.md` (after the mlx-lm fix)
- `PCSS_PRODUCT.md`, `M4_RUNBOOK.md`, `qualification/*` — historical, once every `.md` there carries a banner
- The prior-art modules listed above and their tests
- `schemas/manifest-v1.schema.json`, `schemas/trace-v1.schema.json`, `fixtures/archive/`, `fixtures/manifests/`, `fixtures/traces/`
- `scripts/check.sh`, `.github/workflows/ci.yml`, `pyproject.toml` (after metadata rewrite), `requirements-dev.txt`
