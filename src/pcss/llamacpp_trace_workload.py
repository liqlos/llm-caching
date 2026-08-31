"""Generate a token-only trace workload from a real target llama.cpp model."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, require_string, sha256
from .llamacpp_fixture_builder import _run_reference, load_bridge_config
from .manifest import validate_manifest
from .results import (
    ResultReceipt,
    file_sha256,
    make_receipt,
    recorded_module_command,
    resolve_git_revision,
    write_compact_report,
    write_raw_result,
)
from .trace import normalize_trace_record, trace_stream_fingerprint, validate_trace_sequence
from .verifier import TargetHostIdentity
from .verifier_runner import observe_target_host


PROMPTS_SCHEMA_VERSION = "pcss.target-trace-prompts/v1"
_PROMPTS_FIELDS = {"schema_version", "tokens_to_generate", "prompts"}
_PROMPT_FIELDS = {
    "record_id",
    "request_id",
    "session_id",
    "repository_id",
    "revision",
    "parent_revision",
    "workflow_stage",
    "semantic_block_id",
    "source_class",
    "provenance_family",
    "text",
}


def _load_prompts(path: Path) -> tuple[int, tuple[dict[str, Any], ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("unable to read target trace prompts") from error
    payload = require_mapping(value, "target trace prompts")
    require_exact_keys(payload, _PROMPTS_FIELDS, "target trace prompts")
    if payload["schema_version"] != PROMPTS_SCHEMA_VERSION:
        raise ContractError("unsupported target trace prompts schema")
    tokens_to_generate = require_int(payload["tokens_to_generate"], "target trace prompts.tokens_to_generate", minimum=1)
    prompts = payload["prompts"]
    if not isinstance(prompts, list) or len(prompts) < 2:
        raise ContractError("target trace prompts must contain at least two prompts")
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(prompts):
        prompt = require_mapping(value, f"target trace prompts.prompts[{index}]")
        require_exact_keys(prompt, _PROMPT_FIELDS, f"target trace prompts.prompts[{index}]")
        normalized_prompt = dict(prompt)
        for field in _PROMPT_FIELDS - {"parent_revision"}:
            require_string(normalized_prompt[field], f"target trace prompts.prompts[{index}].{field}")
        if normalized_prompt["parent_revision"] is not None:
            require_string(
                normalized_prompt["parent_revision"],
                f"target trace prompts.prompts[{index}].parent_revision",
            )
        normalized.append(normalized_prompt)
    if len({prompt["record_id"] for prompt in normalized}) != len(normalized):
        raise ContractError("target trace prompt record ids must be unique")
    return tokens_to_generate, tuple(normalized)


def _tokenize(tokenizer: Path, model: Path, text: str, *, timeout_seconds: int) -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            (str(tokenizer), "--model", str(model), "--ids", "--prompt", text),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ContractError("llama-tokenize did not complete") from error
    if completed.returncode != 0:
        raise ContractError("llama-tokenize returned a non-zero exit status")
    try:
        value = ast.literal_eval(completed.stdout.strip())
    except (SyntaxError, ValueError) as error:
        raise ContractError("llama-tokenize output is not a token-id list") from error
    if not isinstance(value, list) or not value or any(type(token) is not int or token < 0 for token in value):
        raise ContractError("llama-tokenize output contains invalid token ids")
    return tuple(value)


def _record(
    *,
    prompt: Mapping[str, Any],
    index: int,
    prompt_tokens: tuple[int, ...],
    output_tokens: tuple[int, ...],
    model_fingerprint: str,
    tokenizer_fingerprint: str,
    template_fingerprint: str,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    repository_id = prompt["repository_id"]
    return {
        "schema_version": "pcss.trace/v1",
        "record_id": prompt["record_id"],
        "timestamp_ns": (index + 1) * 1_000_000_000,
        "chronological_index": index + 1,
        "request_id": prompt["request_id"],
        "session_id": prompt["session_id"],
        "trusted_scope": {
            "scope_class": "personal_persistent",
            "principal_namespace": "synthetic/local-m4",
            "application_id": "pcss-target-trace-workload",
            "repository_id": repository_id,
            "workflow_stage": prompt["workflow_stage"],
            "semantic_block_id": prompt["semantic_block_id"],
        },
        "artifact_lineage": {
            "repository_id": repository_id,
            "revision_family": "synthetic-git",
            "revision": prompt["revision"],
            "parent_revision": prompt["parent_revision"],
            "path_hash": f"sha256:{sha256({'record_id': prompt['record_id'], 'kind': 'path'})}",
            "before_blob": f"sha256:{sha256({'record_id': prompt['record_id'], 'kind': 'before'})}",
            "after_blob": f"sha256:{sha256({'record_id': prompt['record_id'], 'kind': 'after'})}",
            "source_span": None,
        },
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "chat_template_fingerprint": template_fingerprint,
        "generation_protocol_fingerprint": protocol_fingerprint,
        "generation_protocol": {
            "decoding_mode": "greedy",
            "thinking_mode": "off",
            "reasoning_effort": None,
            "preserve_thinking": False,
            "schema_or_tool_version": None,
        },
        "outcome_class": "unknown",
        "source": {
            "source_class": prompt["source_class"],
            "provenance_family": prompt["provenance_family"],
            "eligible_for_persistent_store": True,
        },
        "privacy": {"privacy_class": "public", "retention_class": "ephemeral"},
        "reasoning_retention": "exclude",
        "record_state": "active",
        "content": {
            "prompt_token_ids": list(prompt_tokens),
            "committed_token_ids": list(output_tokens),
            "committed_token_status": "complete",
        },
    }


def generate_trace_workload(
    *,
    config_path: Path,
    manifest_path: Path,
    tokenizer_path: Path,
    prompts_path: Path,
    trace_path: Path,
    output_path: Path,
    report_path: Path,
    repository_root: Path,
    command_arguments: Sequence[str],
    target_observer: Callable[[], TargetHostIdentity] = observe_target_host,
) -> ResultReceipt:
    for path in (trace_path, output_path, report_path):
        if path.exists():
            raise ContractError(f"refuse to overwrite target trace workload artifact: {path}")
    config = load_bridge_config(config_path)
    tokenizer = tokenizer_path.expanduser().resolve()
    if not tokenizer.is_file():
        raise ContractError("target tokenizer executable is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("unable to read target trace workload manifest") from error
    validate_manifest(manifest)
    host = target_observer()
    model_hash = file_sha256(config.model_artifact)
    bridge_hash = file_sha256(config.bridge_executable)
    binding = manifest["verifier_gate"]["adapter_binding"]
    if (
        manifest["evidence_class"] != "LOCAL_MEASURED"
        or manifest["verifier_gate"]["host_identity"] != host.as_dict()
        or manifest["model_artifact"]["sha256"] != model_hash
        or binding["backend_binary_sha256"] != bridge_hash
        or binding["model_artifact_sha256"] != model_hash
        or binding["runtime_configuration_fingerprint"] != config.runtime_configuration_fingerprint
    ):
        raise ContractError("target trace workload inputs differ from observed manifest binding")
    tokens_to_generate, prompts = _load_prompts(prompts_path)
    model_fingerprint = f"sha256:{model_hash}"
    tokenizer_fingerprint = f"sha256:{sha256({'binary': file_sha256(tokenizer), 'model': model_hash, 'flags': ['--ids']})}"
    template_fingerprint = f"sha256:{sha256({'template': 'caller-pretokenized/v1', 'applied': False})}"
    protocol_fingerprint = f"sha256:{sha256({'mode': 'greedy', 'thinking': 'off', 'runtime': config.runtime_configuration_fingerprint})}"
    records: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        prompt_tokens = _tokenize(
            tokenizer,
            config.model_artifact,
            prompt["text"],
            timeout_seconds=config.request_timeout_seconds,
        )
        reference = _run_reference(
            config,
            seed_id=prompt["record_id"],
            context=prompt_tokens,
            steps=tokens_to_generate,
        )
        records.append(
            normalize_trace_record(
                _record(
                    prompt=prompt,
                    index=index,
                    prompt_tokens=prompt_tokens,
                    output_tokens=reference.target_tokens,
                    model_fingerprint=model_fingerprint,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    template_fingerprint=template_fingerprint,
                    protocol_fingerprint=protocol_fingerprint,
                )
            )
        )
    validate_trace_sequence(records, require_uniform_fingerprints=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    receipt = make_receipt(
        evidence_class="LOCAL_MEASURED",
        command=recorded_module_command("pcss.llamacpp_trace_workload", command_arguments, repository_root),
        git_revision=resolve_git_revision(repository_root),
        input_fingerprints={
            "bridge_binary": bridge_hash,
            "config": file_sha256(config.config_path),
            "llama_cpp_library": file_sha256(config.llama_cpp_library),
            "manifest": file_sha256(manifest_path),
            "model_artifact": model_hash,
            "prompts": file_sha256(prompts_path),
            "tokenizer_binary": file_sha256(tokenizer),
        },
        status="TRACE_WORKLOAD_GENERATED",
        payload={
            "kind": "target_generated_token_trace_workload",
            "host_identity": host.as_dict(),
            "record_count": len(records),
            "tokens_to_generate_per_record": tokens_to_generate,
            "trace_path": str(trace_path.resolve()),
            "trace_sha256": file_sha256(trace_path),
            "trace_stream_fingerprint": trace_stream_fingerprint(records),
            "reasoning_retention": "exclude",
            "runtime_claim": "trace generation only; no PCSS performance or speedup claim",
        },
    )
    write_raw_result(output_path, receipt)
    write_compact_report(report_path, raw_result=output_path, title="PCSS target-generated token trace workload")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    generate_trace_workload(
        config_path=args.config,
        manifest_path=args.manifest,
        tokenizer_path=args.tokenizer,
        prompts_path=args.prompts,
        trace_path=args.trace,
        output_path=args.output,
        report_path=args.report,
        repository_root=args.repository_root.resolve(),
        command_arguments=arguments,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
