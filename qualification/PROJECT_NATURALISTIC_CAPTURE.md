# Project-naturalistic PCSS capture protocol

> **HISTORICAL / ARCHIVE-ONLY:** This protocol belongs to the 2026-08 PCSS
> sequence line on llama.cpp, which is closed (see `docs/CLOSED_LINES.md`).
> Its proxy-receipt requirement, holdout rules ("historical captures cannot
> be relabelled as an owner holdout", "cannot be supplied as an arbitrary
> local workload file"), and its numeric acceptance thresholds bind nothing
> now. The current owner-trace corpus rules are in `NOW.md` ("Owner-trace
> corpus"): the owner has authorized local session transcripts as workload
> data, with pre-registration as the safeguard.

This is the next evidence tier after the synthetic calibration receipts. It is
not a prompt generator: only ordinary, user-initiated local llama-server work
may enter the stream.

1. Before opening any benchmark arm, run the loopback forward proxy once per
   ordinary multi-turn `llm-caching` development episode. Keep one 0600 proxy
   ledger alive for the whole capture window and call its loopback
   `/pcss/capture/begin-episode` endpoint at each genuine boundary; the proxy
   generates opaque episode/event ids, timestamps, a hash chain, and tool
   version, fsyncs each visible request before forwarding, and never records
   responses. Do not record completions, candidates, token
   traces, hidden reasoning, logits/KV data, or post-response labels.
2. Seal the 32 request bodies and their SHA-256 list before launching a server.
   Every eligible request remains in the workload, including unique prompts,
   repeats, failures, and cancellations. Do not select for reuse rate or model
   behaviour.
3. Run baseline, upstream ngram, PCSS current-context-only, and full PCSS in
   a preregistered Latin-square order across episodes, with fresh isolated
   state per arm. The required harness extension must retain all 128 samples.
4. Require exact output, complete process cleanup, the existing resource
   ceilings, baseline/full clustered LCB95 >= 1.05, current-only/full LCB95 >=
   1.02, no-persistent-hit p95 overhead UCB95 <= 3% (all PCSS requests with
   zero persistent hit, including current-context-only drafts), persistent hits plus
   accepted drafts in at least four episodes/eight requests, and TTFT p95
   regression <= 3%.

The only permitted claim after a passing result is a project-local result for
this exact machine, model, runtime and capture protocol. It is not an owner
representative holdout, a general speedup claim, or a default-enable decision.

## Prospective current-source owner holdout

Historical captures cannot be relabelled as an owner holdout. For a future
ordinary owner episode that genuinely carries `pcss_current_source`, start the
proxy with a separate new sidecar:

```sh
scripts/capture-project-proxy.sh CAPTURE.jsonl \
  --origin ordinary_local_llama_server \
  --current-source-sidecar CURRENT-SOURCE.jsonl
```

The v2 public ledger holds visible request bodies and non-reversible source
commitments only. `CURRENT-SOURCE.jsonl` is a `0600` append-only sidecar; it
holds the raw token envelopes and is fsynced before its matching public ledger
event. A failed sidecar write therefore prevents forwarding the request.

Seal that exact pair before opening any arm:

```sh
python -m pcss.project_workload_capture \
  --input CAPTURE.jsonl \
  --source-sidecar CURRENT-SOURCE.jsonl \
  --workload OWNER-HOLDOUT.json \
  --receipt OWNER-HOLDOUT-receipt.json
```

The source-bearing sealed workload remains `0600`; the immutable v2 receipt is
`0444` and contains only the public-ledger hash, sidecar hash, attachment
count, and per-request commitments. The benchmark copies only that public
receipt and a separately redacted workload archive. Its frozen owner-holdout
manifest must bind those hashes/counts, declare the ordinary local origin,
place every request in the one-shot holdout split, and include no calibration
or tuning lane.

An owner-representative holdout without current-source input still requires a
trusted ordinary v1 proxy receipt; it cannot be supplied as an arbitrary local
workload file. The v2 sidecar contract is the additional requirement whenever
the holdout contains current-source attachments.
