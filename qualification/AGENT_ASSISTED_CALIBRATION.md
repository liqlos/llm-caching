# Agent-assisted local calibration boundary

> **HISTORICAL / ARCHIVE-ONLY:** This file belongs to the 2026-08 PCSS sequence line on llama.cpp, which is closed (see `docs/CLOSED_LINES.md`). Its rules, gates, thresholds, budgets and stop rules bind nothing now. The current line is defined only by `NOW.md`.

The `project-naturalistic-v1` through `project-naturalistic-v4` artifacts use
operator-authorized requests generated while this PCSS development task used
the local Qwen server. Their proxy capture is technically trusted: visible
request bodies were fsynced before forwarding, ordered, hash-chained, and
sealed without responses.

They are nevertheless **agent-assisted calibration**, not the ordinary
user-initiated project-naturalistic protocol. In particular, they do not meet
the prospective user-origin and four-arm requirements in
`PROJECT_NATURALISTIC_CAPTURE.md`. They may prove exact local implementation
behaviour or a workload-scoped no-go, but cannot support a general, owner,
default-enable, or technical-limit claim outside their frozen patch/policy/
workload boundary.
