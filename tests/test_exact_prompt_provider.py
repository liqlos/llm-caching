from __future__ import annotations

import json
from pathlib import Path

from pcss.exact_prompt_archive import build_exact_prompt_archive
from pcss.exact_prompt_provider import ExactPromptArchiveShadowProvider, ExactPromptDescriptor
from pcss.exact_repeat_trace import derive_exact_repeat_records
from pcss.provider import Fingerprints, Proposal, RequestContext, TrustedScope


ROOT = Path(__file__).parents[1]


def test_exact_prompt_provider_reads_sealed_cursor_and_abstains_on_unknown_prompt(tmp_path: Path) -> None:
    source = json.loads((ROOT / "fixtures/replay/chronological-oracle-v1.json").read_text(encoding="utf-8"))[0]
    source["content"]["prompt_token_ids"] = [90, 91, 92, 93]
    source, target = derive_exact_repeat_records([source])
    archive = build_exact_prompt_archive(tmp_path / "archive", [source])
    fingerprints = Fingerprints("model", "tokenizer", "template", "protocol")
    scope = TrustedScope("principal", "app", "repo", "git", "edit", "block", None, widening_level=1)
    descriptor = ExactPromptDescriptor(
        tail_tokens=tuple(target["content"]["prompt_token_ids"]),
        source_record_id=source["record_id"],
        source_class="same_repository",
        scope=scope,
        fingerprints=fingerprints,
        provenance_family="repository",
        matched_suffix_tokens=len(target["content"]["prompt_token_ids"]),
    )
    provider = ExactPromptArchiveShadowProvider(archive, (descriptor,), clock_ns=lambda: 10)
    request = RequestContext("request", fingerprints, scope, ("same_repository",), deadline_ns=100, probe_budget=1)
    provider.begin_request(request)

    result = provider.propose(
        request.request_id, tail_tokens=descriptor.tail_tokens, max_tokens=2, deadline_ns=100
    )
    assert isinstance(result, Proposal)
    assert result.tokens == tuple(target["content"]["committed_token_ids"][:2])
    miss = provider.propose(request.request_id, tail_tokens=(1, 2, 3, 4), max_tokens=2, deadline_ns=100)
    assert miss.reason == "no_exact_prompt_descriptor"
