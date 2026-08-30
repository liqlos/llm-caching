# PCSS llama.cpp bridge

pcss_llama_bridge is a small process-isolated C-API bridge for verifier
correctness. It is not a live PCSS provider and emits no timing, throughput,
memory, or speedup result.

The bridge receives exactly one JSON object per non-empty stdin line and writes
one JSON object per line. All requests use
pcss.llama-bridge-request/v1; every request must be run with its matching
operation guard:

~~~
cmake -S native -B build/native -DLLAMA_CPP_ROOT=/opt/homebrew/Cellar/llama.cpp/10520
cmake --build build/native --parallel 2

build/native/pcss_llama_bridge --pcss-capabilities
build/native/pcss_llama_bridge --pcss-reference
build/native/pcss_llama_bridge --pcss-verify-case
build/native/pcss_llama_bridge --pcss-profile
~~~

The runtime request shape is deliberately explicit:

~~~json
{
  "schema_version": "pcss.llama-bridge-request/v1",
  "operation": "capabilities",
  "model_artifact": "/absolute/path/to/model.gguf",
  "llama_cpp_library": "/absolute/path/to/libllama.dylib",
  "llama_cpp_revision": "b10520",
  "runtime": {
    "n_ctx": 512,
    "n_batch": 32,
    "n_gpu_layers": 99,
    "flash_attn": "auto",
    "seed": 0,
    "n_threads": 4,
    "n_threads_batch": 4,
    "chat_template": "caller-pretokenized/v1",
    "thinking_mode": "disabled"
  }
}
~~~

flash_attn is one of auto, enabled, or disabled. The bridge loads
model_artifact, sets n_gpu_layers, n_ctx, n_batch, thread counts, and
flash-attention mode through the C API. It only echoes llama_cpp_library and
llama_cpp_revision; caller-side binding/provenance code must validate those
paths and revisions independently. thinking_mode must be exactly disabled: the
bridge accepts only raw token IDs and will not apply a chat template or any
reasoning policy.

## Capabilities

--pcss-capabilities accepts the common request above. It does not report ok
until it has loaded the requested GGUF, created the requested context, decoded
the model BOS token (or token 0 if no BOS exists), exported the model-memory state
with llama_state_get_data, restored it with llama_state_set_data into a fresh
equivalent context, and compared the two serialized byte streams exactly.

At the pinned llama.cpp revision, this state contains the model architecture
identity plus the context memory/KV state. It does not contain logits,
embeddings, a sampler, or RNG state. The successful response uses
pcss.llama-bridge-capabilities/v2. Its required
top-level fields are:

~~~text
schema_version, operation, status, backend, state_export, generation_mode,
greedy_tie_break, raw_token_ids, llama_c_api_version, supports_gpu_offload,
loaded_llama_cpp_library_path, supports_mmap, supports_mlock, vocabulary_size, configuration,
observed_context, state_components, state_api, sampler_rng_state, probe_token,
probe_state_sha256, probe_state_bytes, restored_state_sha256,
restored_state_bytes, state_round_trip_byte_exact, operations,
declared_identity_fields, chat_template_applied, thinking_mode_applied
~~~

configuration is exactly the caller's
{model_artifact,llama_cpp_library,llama_cpp_revision,runtime} object.
loaded_llama_cpp_library_path is obtained from dladdr(&llama_backend_init) and
canonicalized with realpath when available. The harness must hash this observed
path and require it to match its configured library before treating capability
evidence as target-backend evidence.
observed_context contains the C-API-reported actual_n_ctx, actual_n_batch,
actual_n_threads, and actual_n_threads_batch. The bridge does not create a
sampler or RNG: successful capability output explicitly says
generation_mode: "greedy_argmax" and
sampler_rng_state: "not-applicable-no-sampler-or-rng-in-greedy-bridge/v1".

## Reference

--pcss-reference adds these fields to the common request:

~~~json
{
  "operation": "reference",
  "seed": {
    "seed_id": "fixture-seed-001",
    "initial_context_tokens": [1, 2, 3],
    "steps": 8
  },
  "reference": {
    "method": "serial-greedy-memory-state/v2"
  }
}
~~~

initial_context_tokens are raw token IDs. The bridge applies no tokenizer,
chat template, or thinking-mode policy. It serially chooses exactly steps
tokens by greedy argmax, breaking equal logits by lowest token ID. steps may be
zero, in which case target_tokens is empty and the completion state is the
prompt state.

The successful response uses pcss.llama-bridge-reference/v2 and contains:

~~~text
schema_version, operation, status, backend, generation_mode, greedy_tie_break,
seed_id, target_tokens, vocabulary_size, state_export, prompt_state_sha256,
prompt_state_bytes, completion_state_sha256, completion_state_bytes,
mutable_state_digest, configuration, observed_context,
tokenization_boundary, state_api, sampler_rng_state,
declared_identity_fields, chat_template_applied, thinking_mode_applied
~~~

target_tokens has exactly seed.steps entries. mutable_state_digest equals
completion_state_sha256, the SHA-256 of the model-memory state after that serial
continuation.

## Verification case

--pcss-verify-case adds these fields to the common request:

~~~json
{
  "operation": "verify",
  "seed": {
    "seed_id": "fixture-seed-001",
    "initial_context_tokens": [1, 2, 3]
  },
  "case": {
    "case_id": "reject-at-two",
    "draft_tokens": [4, 5, 6],
    "include_bonus_on_full_accept": false
  }
}
~~~

The bridge generates a serial reference and restores the prompt state into a
disposable verifier context. It scores every draft token in one batch with
logits enabled at every output position. The prompt logits plus those batch
outputs define the verifier target sequence and therefore the accepted prefix,
correction, or optional bonus token.

The disposable verifier context is never committed. The bridge restores the
prompt state into separate reference and committed contexts, then decodes the
serial-reference commit and verifier-derived commit one token at a time. This
serial canonical commit is deliberately a correctness protocol, not a speed
claim.

It requires the authoritative verifier targets to equal the serial reference,
compares the resulting model-memory states byte-for-byte, and compares the next
greedy token. A mismatch yields
status: "mismatch" and a non-zero process exit after its JSON response.

The successful or mismatch response uses pcss.llama-bridge-verify/v2 and
contains:

~~~text
schema_version, operation, status, backend, generation_mode, greedy_tie_break,
state_export, reconstruction, seed_id, case_id, target_tokens,
verification_target_tokens, verification_targets_match, vocabulary_size,
accepted_draft_tokens, reference_accepted_draft_tokens, correction_token,
bonus_token, committed_tokens, reference_committed_tokens,
prompt_state_sha256, prompt_state_bytes, reference_state_sha256,
reference_state_bytes, committed_state_sha256, committed_state_bytes,
verifier_draft_state_sha256, verifier_draft_state_bytes,
reference_mutable_state_digest, committed_mutable_state_digest,
mutable_state_digest, state_matches, reference_next_token,
committed_next_token, continuation_matches, configuration,
observed_context, tokenization_boundary, state_api, sampler_rng_state,
declared_identity_fields, chat_template_applied, thinking_mode_applied
~~~

mutable_state_digest is the canonical committed-state digest. The bridge never takes
caller-provided target tokens or expected state digests; the harness must
independently validate those values.

## Isolated verifier profile

`--pcss-profile` is available only after the correctness suites pass for the
same committed bridge, model, llama.cpp library, and runtime configuration. It
adds a capability-derived raw-token seed and this bounded profile request:

~~~json
{
  "operation": "profile",
  "profile": {
    "k_values": [1, 2, 4, 8, 16, 32],
    "warmup_repetitions": 2,
    "measurement_repetitions": 8
  }
}
~~~

For every K, the bridge derives a perfect continuation serially, then records
paired serial-before, one-batch verifier, and serial-after decode durations.
Each timed decode is synchronized. Context creation and model loading are
outside the measurement, while state restore and snapshot durations are
reported separately. The response schema is
`pcss.llama-bridge-profile/v1`.

This is an isolated decode profiler, not live PCSS. Its decode-only ratio omits
provider cost, canonical commit work, scheduling, sustained thermal behavior,
and end-to-end workload acceptance. A caller must preserve raw repetitions and
fail closed when the two serial baselines exceed its predeclared tolerance.

## Fail-closed behavior

Errors use pcss.llama-bridge-result/v1:

~~~json
{
  "schema_version": "pcss.llama-bridge-result/v1",
  "operation": "verify",
  "status": "error",
  "backend": "llama.cpp",
  "error": {"code": "state_restore_failed", "message": "…"}
}
~~~

Any error or verification mismatch makes the process exit non-zero after it
has emitted its JSON response. Neither a compile nor a capability response is
a performance or deployment claim.
