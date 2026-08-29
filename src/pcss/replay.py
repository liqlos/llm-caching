"""Model-free, chronological PCSS opportunity replay.

This module deliberately evaluates *opportunity*, not target-runtime speed.  A
candidate may only be retrieved from a completed earlier record, an explicitly
provided current source, or the legal prefix of the current request.  Future
target tokens are used solely after retrieval to score an exact continuation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .canonical import ContractError, canonical_bytes, require_int, require_mapping, require_string


NGRAM_SIZES = (4, 8, 12, 16, 24, 32)


class ReplayError(ContractError):
    """Raised when replay input cannot support a fail-closed evaluation."""


class SourceTier(StrEnum):
    """Cheapest legal source classes used for span-level attribution."""

    CURRENT_CONTEXT = "current_context"
    CURRENT_SOURCE = "current_source"
    SAME_SESSION = "same_session"
    SAME_REPOSITORY = "same_repository"
    PERSONAL_PERSISTENT_HISTORY = "personal_persistent_history"
    IMPORTED_CORPUS = "imported_corpus"


SOURCE_TIER_PRIORITY = (
    SourceTier.CURRENT_CONTEXT,
    SourceTier.CURRENT_SOURCE,
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
)
_TIER_RANK = {tier: index for index, tier in enumerate(SOURCE_TIER_PRIORITY)}
_PERSISTENT_TIERS = frozenset(
    {
        SourceTier.SAME_SESSION,
        SourceTier.SAME_REPOSITORY,
        SourceTier.PERSONAL_PERSISTENT_HISTORY,
        SourceTier.IMPORTED_CORPUS,
    }
)


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return require_string(value, field)


def _token_tuple(value: Any, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReplayError(f"{field} must be an array of u32 token ids")
    result: list[int] = []
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF:
            raise ReplayError(f"{field}[{index}] must be a u32 token id")
        result.append(token)
    return tuple(result)


def _mapping_field(value: Mapping[str, Any], name: str, field: str) -> Mapping[str, Any]:
    try:
        return require_mapping(value[name], f"{field}.{name}")
    except KeyError as error:
        raise ReplayError(f"{field}.{name} is required") from error


@dataclass(frozen=True, order=True)
class Fingerprints:
    """Fingerprint tuple that makes cross-protocol retrieval fail closed."""

    model: str
    tokenizer: str
    chat_template: str
    generation_protocol: str

    @classmethod
    def from_mapping(cls, value: Any, field: str = "fingerprints") -> "Fingerprints":
        mapping = require_mapping(value, field)
        expected = {"model", "tokenizer", "chat_template", "generation_protocol"}
        actual = set(mapping)
        if actual != expected:
            raise ReplayError(f"{field} has invalid keys: expected={sorted(expected)}, actual={sorted(actual)}")
        return cls(
            model=require_string(mapping["model"], f"{field}.model"),
            tokenizer=require_string(mapping["tokenizer"], f"{field}.tokenizer"),
            chat_template=require_string(mapping["chat_template"], f"{field}.chat_template"),
            generation_protocol=require_string(
                mapping["generation_protocol"], f"{field}.generation_protocol"
            ),
        )

    @classmethod
    def from_trace_record(cls, value: Mapping[str, Any]) -> "Fingerprints":
        """Read the four explicit trace-v1 compatibility namespace fields."""

        return cls(
            model=require_string(value.get("model_fingerprint"), "record.model_fingerprint"),
            tokenizer=require_string(value.get("tokenizer_fingerprint"), "record.tokenizer_fingerprint"),
            chat_template=require_string(
                value.get("chat_template_fingerprint"), "record.chat_template_fingerprint"
            ),
            generation_protocol=require_string(
                value.get("generation_protocol_fingerprint"),
                "record.generation_protocol_fingerprint",
            ),
        )


@dataclass(frozen=True)
class ScopeIdentity:
    """Only fields needed for trustworthy logical scope selection."""

    principal_namespace: str | None
    application_id: str | None
    repository_id: str | None
    workflow_stage: str | None
    semantic_block_id: str | None
    session_id: str | None
    revision_family: str | None
    path_hash: str | None
    before_blob: str | None

    def same_repository(self, other: "ScopeIdentity") -> bool:
        return bool(self.repository_id and self.repository_id == other.repository_id)

    def same_session(self, other: "ScopeIdentity") -> bool:
        return bool(self.session_id and self.session_id == other.session_id)

    def same_block(self, other: "ScopeIdentity") -> bool:
        return bool(
            self.same_repository(other)
            and self.workflow_stage
            and self.workflow_stage == other.workflow_stage
            and self.semantic_block_id
            and self.semantic_block_id == other.semantic_block_id
        )

    def same_current_source(self, other: "ScopeIdentity") -> bool:
        if not self.same_repository(other) or not self.path_hash or self.path_hash != other.path_hash:
            return False
        if self.before_blob and self.before_blob == other.before_blob:
            return True
        return bool(self.revision_family and self.revision_family == other.revision_family)


@dataclass(frozen=True)
class ReplayRecord:
    """Visible committed tokens plus the minimum metadata for offline replay."""

    record_id: str
    timestamp_ns: int
    chronological_index: int
    prompt_tokens: tuple[int, ...]
    committed_tokens: tuple[int, ...]
    fingerprints: Fingerprints
    scope: ScopeIdentity
    source_class: str
    provenance_family: str
    eligible_for_persistent_store: bool
    current_source_tokens: tuple[int, ...] | None = None
    declared_source_bytes: int | None = None
    request_id: str | None = None

    @property
    def order_key(self) -> tuple[int, int, str]:
        return (self.timestamp_ns, self.chronological_index, self.record_id)

    @property
    def logical_source_bytes(self) -> int:
        """Exact bytes under the declared u32-token logical accounting model."""
        if self.declared_source_bytes is not None:
            return self.declared_source_bytes
        return 4 * len(self.committed_tokens)


def record_from_mapping(value: Mapping[str, Any]) -> ReplayRecord:
    """Coerce a trace-v1-shaped mapping without importing the trace package.

    The adapter accepts optional session and source-token fields which are not
    required by trace-v1.  Their absence means the related source tier simply
    has no evidence rather than being filled with an invented value.
    """

    raw = require_mapping(value, "record")
    content = _mapping_field(raw, "content", "record")
    scope_raw = _mapping_field(raw, "trusted_scope", "record")
    lineage_field = "artifact_lineage" if "artifact_lineage" in raw else "repository_lineage"
    lineage = _mapping_field(raw, lineage_field, "record")
    source = _mapping_field(raw, "source", "record")

    record_id = require_string(raw.get("record_id"), "record.record_id")
    timestamp_ns = require_int(raw.get("timestamp_ns"), "record.timestamp_ns")
    chronological_index = require_int(raw.get("chronological_index"), "record.chronological_index")
    fingerprints = (
        Fingerprints.from_trace_record(raw)
        if "model_fingerprint" in raw
        else Fingerprints.from_mapping(raw.get("fingerprints"), "record.fingerprints")
    )

    raw_session = raw.get("session_id", scope_raw.get("session_id", source.get("session_id")))
    scope = ScopeIdentity(
        principal_namespace=_optional_string(
            scope_raw.get("principal_namespace"), "record.trusted_scope.principal_namespace"
        ),
        application_id=_optional_string(scope_raw.get("application_id"), "record.trusted_scope.application_id"),
        repository_id=_optional_string(
            lineage.get("repository_id", scope_raw.get("repository_id")),
            f"record.{lineage_field}.repository_id",
        ),
        workflow_stage=_optional_string(scope_raw.get("workflow_stage"), "record.trusted_scope.workflow_stage"),
        semantic_block_id=_optional_string(
            scope_raw.get("semantic_block_id"), "record.trusted_scope.semantic_block_id"
        ),
        session_id=_optional_string(raw_session, "record.session_id"),
        revision_family=_optional_string(
            lineage.get("revision_family"), f"record.{lineage_field}.revision_family"
        ),
        path_hash=_optional_string(lineage.get("path_hash"), f"record.{lineage_field}.path_hash"),
        before_blob=_optional_string(lineage.get("before_blob"), f"record.{lineage_field}.before_blob"),
    )
    source_tokens_value = content.get("current_source_token_ids", content.get("source_token_ids"))
    if source_tokens_value is None:
        source_tokens_value = raw.get("current_source_token_ids", raw.get("source_token_ids"))
    current_source_tokens = (
        _token_tuple(source_tokens_value, "record.content.current_source_token_ids")
        if source_tokens_value is not None
        else None
    )
    declared_source_bytes = raw.get("declared_source_bytes", content.get("declared_source_bytes"))
    if declared_source_bytes is not None:
        declared_source_bytes = require_int(declared_source_bytes, "record.declared_source_bytes")
    eligible = source.get("eligible_for_persistent_store")
    if not isinstance(eligible, bool):
        raise ReplayError("record.source.eligible_for_persistent_store must be a boolean")
    return ReplayRecord(
        record_id=record_id,
        timestamp_ns=timestamp_ns,
        chronological_index=chronological_index,
        prompt_tokens=_token_tuple(content.get("prompt_token_ids"), "record.content.prompt_token_ids"),
        committed_tokens=_token_tuple(
            content.get("committed_token_ids"), "record.content.committed_token_ids"
        ),
        fingerprints=fingerprints,
        scope=scope,
        source_class=require_string(source.get("source_class"), "record.source.source_class"),
        provenance_family=require_string(source.get("provenance_family"), "record.source.provenance_family"),
        eligible_for_persistent_store=eligible,
        current_source_tokens=current_source_tokens,
        declared_source_bytes=declared_source_bytes,
        request_id=_optional_string(raw.get("request_id"), "record.request_id"),
    )


def coerce_records(records: Iterable[ReplayRecord | Mapping[str, Any]]) -> tuple[ReplayRecord, ...]:
    """Return a deterministic, strict chronological order for replay."""

    result = tuple(
        record
        if isinstance(record, ReplayRecord)
        else record_from_mapping(record)
        for record in records
        if not isinstance(record, Mapping) or record.get("record_state") != "redacted"
    )
    seen: set[tuple[int, int]] = set()
    for record in result:
        key = (record.timestamp_ns, record.chronological_index)
        if key in seen:
            raise ReplayError(f"duplicate chronological ordering key: {key}")
        seen.add(key)
    return tuple(sorted(result, key=lambda record: record.order_key))


def longest_exact_suffix(
    query_tokens: Sequence[int], source_tokens: Sequence[int], source_cursor: int
) -> int:
    """Length of the exact suffix ending at a legal source cursor."""

    if source_cursor < 0 or source_cursor > len(source_tokens):
        raise ReplayError("source_cursor is outside source_tokens")
    length = 0
    limit = min(len(query_tokens), source_cursor)
    while length < limit and query_tokens[-1 - length] == source_tokens[source_cursor - 1 - length]:
        length += 1
    return length


def continuation_prefix_length(target_tokens: Sequence[int], candidate_tokens: Sequence[int]) -> int:
    """Post-hoc exact target prefix length; never call this to select a source."""

    length = 0
    for expected, actual in zip(target_tokens, candidate_tokens):
        if expected != actual:
            break
        length += 1
    return length


@dataclass(frozen=True)
class SuffixMatch:
    """A dense, exact match at one cursor in a legal source sequence."""

    source_cursor: int
    matched_suffix_length: int
    continuation: tuple[int, ...]
    accepted_prefix_length: int


def dense_exact_suffix_matches(
    query_tokens: Sequence[int],
    target_tokens: Sequence[int],
    source_tokens: Sequence[int],
    *,
    min_suffix: int = 1,
    max_continuation: int | None = None,
) -> tuple[SuffixMatch, ...]:
    """Enumerate every dense exact-suffix candidate with a legal continuation.

    `target_tokens` is inspected only after a candidate has been constructed,
    which makes this suitable for post-hoc oracle scoring but not a ranker.
    """

    if min_suffix < 0:
        raise ReplayError("min_suffix must be non-negative")
    if max_continuation is not None and max_continuation < 1:
        raise ReplayError("max_continuation must be positive when set")
    matches: list[SuffixMatch] = []
    # The final source token cannot be a cursor because it has no continuation.
    for cursor in range(0, max(0, len(source_tokens) - 1) + 1):
        if cursor >= len(source_tokens):
            continue
        matched = longest_exact_suffix(query_tokens, source_tokens, cursor)
        if matched < min_suffix:
            continue
        continuation = tuple(
            source_tokens[cursor:] if max_continuation is None else source_tokens[cursor : cursor + max_continuation]
        )
        if not continuation:
            continue
        matches.append(
            SuffixMatch(
                source_cursor=cursor,
                matched_suffix_length=matched,
                continuation=continuation,
                accepted_prefix_length=continuation_prefix_length(target_tokens, continuation),
            )
        )
    return tuple(matches)


@dataclass(frozen=True)
class Candidate:
    """A source cursor plus post-hoc replay evidence."""

    tier: SourceTier
    source_record_id: str
    source_cursor: int
    matched_suffix_length: int
    continuation: tuple[int, ...]
    accepted_prefix_length: int
    source_token_bytes: int
    source_scope: ScopeIdentity
    provenance_family: str
    source_class: str

    @property
    def candidate_key(self) -> tuple[str, int, SourceTier]:
        return (self.source_record_id, self.source_cursor, self.tier)


@dataclass(frozen=True)
class Opportunity:
    """One target-output position evaluated against only legal evidence."""

    record_id: str
    position: int
    target_remaining_tokens: int
    candidates: tuple[Candidate, ...]
    request_id: str | None = None

    @property
    def request_cluster_id(self) -> str:
        """Trace request id when present, otherwise one record is one cluster."""
        return self.request_id or self.record_id

    def candidates_for(self, tier: SourceTier) -> tuple[Candidate, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.tier == tier)

    def prefix_for(self, tier: SourceTier) -> int:
        return max((candidate.accepted_prefix_length for candidate in self.candidates_for(tier)), default=0)

    @property
    def all_prefix(self) -> int:
        return max((candidate.accepted_prefix_length for candidate in self.candidates), default=0)

    @property
    def context_prefix(self) -> int:
        return self.prefix_for(SourceTier.CURRENT_CONTEXT)

    @property
    def current_source_prefix(self) -> int:
        return self.prefix_for(SourceTier.CURRENT_SOURCE)

    @property
    def persistent_prefix(self) -> int:
        return max((self.prefix_for(tier) for tier in _PERSISTENT_TIERS), default=0)

    @property
    def persistent_unique_prefix(self) -> int:
        """PCSS credit after cheaper context and current-source alternatives."""
        return max(0, self.persistent_prefix - max(self.context_prefix, self.current_source_prefix))

    @property
    def best_candidate(self) -> Candidate | None:
        if not self.candidates:
            return None
        return min(
            self.candidates,
            key=lambda candidate: (
                -candidate.accepted_prefix_length,
                -candidate.matched_suffix_length,
                _TIER_RANK[candidate.tier],
                candidate.source_record_id,
                candidate.source_cursor,
            ),
        )

    def source_attribution(self) -> dict[SourceTier, int]:
        """Attribute each accepted offset to the cheapest tier that can emit it."""
        counts: Counter[SourceTier] = Counter()
        for offset in range(1, self.all_prefix + 1):
            for tier in SOURCE_TIER_PRIORITY:
                if self.prefix_for(tier) >= offset:
                    counts[tier] += 1
                    break
        return dict(counts)


@dataclass(frozen=True)
class NgramCensus:
    n: int
    total_occurrences: int
    distinct_ngrams: int
    repeated_occurrences: int
    chronological_reused_occurrences: int
    max_posting_fanout: int
    mean_posting_fanout: float


@dataclass(frozen=True)
class PostingFanout:
    n: int
    postings: int
    keys: int
    max_fanout: int
    mean_fanout: float
    fanout_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FootprintAccounting:
    """Declared logical u32 archive bytes, not a physical storage benchmark."""

    source_tokens: int
    token_bytes: int
    metadata_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class CorpusGrowthPoint:
    record_id: str
    records_seen: int
    source_tokens: int
    footprint: FootprintAccounting


@dataclass(frozen=True)
class ReplayCensus:
    ngrams: tuple[NgramCensus, ...]
    fanout: tuple[PostingFanout, ...]
    growth: tuple[CorpusGrowthPoint, ...]
    footprint: FootprintAccounting


@dataclass(frozen=True)
class ReplayResult:
    """Offline replay output; no field represents a runtime-speed result."""

    records: tuple[ReplayRecord, ...]
    opportunities: tuple[Opportunity, ...]
    census: ReplayCensus
    future_tokens_used_for_retrieval: bool = False


def _record_metadata_bytes(record: ReplayRecord) -> int:
    metadata = {
        "record_id": record.record_id,
        "request_id": record.request_id,
        "timestamp_ns": record.timestamp_ns,
        "chronological_index": record.chronological_index,
        "fingerprints": {
            "model": record.fingerprints.model,
            "tokenizer": record.fingerprints.tokenizer,
            "chat_template": record.fingerprints.chat_template,
            "generation_protocol": record.fingerprints.generation_protocol,
        },
        "scope": record.scope.__dict__,
        "source_class": record.source_class,
        "provenance_family": record.provenance_family,
        "eligible_for_persistent_store": record.eligible_for_persistent_store,
    }
    return len(canonical_bytes(metadata))


def footprint_accounting(records: Iterable[ReplayRecord | Mapping[str, Any]]) -> FootprintAccounting:
    """Count source payloads as immutable u32 streams plus canonical metadata."""

    normalized = coerce_records(records)
    source_tokens = sum(len(record.committed_tokens) for record in normalized)
    token_bytes = sum(record.logical_source_bytes for record in normalized)
    metadata_bytes = sum(_record_metadata_bytes(record) for record in normalized)
    return FootprintAccounting(
        source_tokens=source_tokens,
        token_bytes=token_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=token_bytes + metadata_bytes,
    )


def corpus_growth(records: Iterable[ReplayRecord | Mapping[str, Any]]) -> tuple[CorpusGrowthPoint, ...]:
    """Growth curve over one ordered immutable archive representation."""

    normalized = coerce_records(records)
    token_count = 0
    token_bytes = 0
    metadata_bytes = 0
    result: list[CorpusGrowthPoint] = []
    for index, record in enumerate(normalized, start=1):
        token_count += len(record.committed_tokens)
        token_bytes += record.logical_source_bytes
        metadata_bytes += _record_metadata_bytes(record)
        footprint = FootprintAccounting(
            source_tokens=token_count,
            token_bytes=token_bytes,
            metadata_bytes=metadata_bytes,
            total_bytes=token_bytes + metadata_bytes,
        )
        result.append(CorpusGrowthPoint(record.record_id, index, token_count, footprint))
    return tuple(result)


def ngram_census(
    records: Iterable[ReplayRecord | Mapping[str, Any]], *, n_values: Sequence[int] = NGRAM_SIZES
) -> tuple[NgramCensus, ...]:
    """Census output-only n-grams without crossing independent record bounds."""

    normalized = coerce_records(records)
    result: list[NgramCensus] = []
    for n in n_values:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ReplayError("n_values must contain positive integers")
        postings: Counter[tuple[int, ...]] = Counter()
        chronological_reused = 0
        prior: set[tuple[int, ...]] = set()
        total = 0
        for record in normalized:
            tokens = record.committed_tokens
            for offset in range(0, max(0, len(tokens) - n + 1)):
                gram = tokens[offset : offset + n]
                if gram in prior:
                    chronological_reused += 1
                prior.add(gram)
                postings[gram] += 1
                total += 1
        fanouts = tuple(postings.values())
        result.append(
            NgramCensus(
                n=n,
                total_occurrences=total,
                distinct_ngrams=len(postings),
                repeated_occurrences=sum(count for count in fanouts if count > 1),
                chronological_reused_occurrences=chronological_reused,
                max_posting_fanout=max(fanouts, default=0),
                mean_posting_fanout=(sum(fanouts) / len(fanouts)) if fanouts else 0.0,
            )
        )
    return tuple(result)


def posting_fanout(
    records: Iterable[ReplayRecord | Mapping[str, Any]], *, n_values: Sequence[int] = NGRAM_SIZES
) -> tuple[PostingFanout, ...]:
    """Fanout histogram for dense fixed-key postings, not an index benchmark."""

    normalized = coerce_records(records)
    result: list[PostingFanout] = []
    for n in n_values:
        postings: Counter[tuple[int, ...]] = Counter()
        for record in normalized:
            tokens = record.committed_tokens
            postings.update(tokens[offset : offset + n] for offset in range(max(0, len(tokens) - n + 1)))
        fanouts = tuple(postings.values())
        histogram = Counter(fanouts)
        result.append(
            PostingFanout(
                n=n,
                postings=sum(fanouts),
                keys=len(postings),
                max_fanout=max(fanouts, default=0),
                mean_fanout=(sum(fanouts) / len(fanouts)) if fanouts else 0.0,
                fanout_histogram=tuple(sorted(histogram.items())),
            )
        )
    return tuple(result)


def build_census(records: Iterable[ReplayRecord | Mapping[str, Any]]) -> ReplayCensus:
    normalized = coerce_records(records)
    return ReplayCensus(
        ngrams=ngram_census(normalized),
        fanout=posting_fanout(normalized),
        growth=corpus_growth(normalized),
        footprint=footprint_accounting(normalized),
    )


def _authorized_for_persistent(target: ReplayRecord, source: ReplayRecord) -> bool:
    """Cross-namespace history is never silently admitted to a replay query."""

    return (
        source.eligible_for_persistent_store
        and target.scope.principal_namespace == source.scope.principal_namespace
        and target.scope.application_id == source.scope.application_id
    )


def _tier_for_history(target: ReplayRecord, source: ReplayRecord) -> SourceTier:
    if "import" in source.source_class.casefold():
        return SourceTier.IMPORTED_CORPUS
    if target.scope.same_current_source(source.scope):
        return SourceTier.CURRENT_SOURCE
    if target.scope.same_session(source.scope):
        return SourceTier.SAME_SESSION
    if target.scope.same_repository(source.scope):
        return SourceTier.SAME_REPOSITORY
    return SourceTier.PERSONAL_PERSISTENT_HISTORY


def _to_candidates(
    *,
    tier: SourceTier,
    source_record_id: str,
    source_tokens: Sequence[int],
    source_token_bytes: int,
    source_scope: ScopeIdentity,
    provenance_family: str,
    source_class: str,
    query_tokens: Sequence[int],
    target_tokens: Sequence[int],
    min_suffix: int,
    max_continuation: int | None,
) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            tier=tier,
            source_record_id=source_record_id,
            source_cursor=match.source_cursor,
            matched_suffix_length=match.matched_suffix_length,
            continuation=match.continuation,
            accepted_prefix_length=match.accepted_prefix_length,
            source_token_bytes=source_token_bytes,
            source_scope=source_scope,
            provenance_family=provenance_family,
            source_class=source_class,
        )
        for match in dense_exact_suffix_matches(
            query_tokens,
            target_tokens,
            source_tokens,
            min_suffix=min_suffix,
            max_continuation=max_continuation,
        )
    )


def replay_chronologically(
    records: Iterable[ReplayRecord | Mapping[str, Any]],
    *,
    min_suffix: int = 1,
    max_continuation: int | None = 32,
) -> ReplayResult:
    """Run a dense, model-free oracle without future-history access.

    Records enter ``completed_history`` only after all positions in that record
    have been scored.  At a current position, the only same-record material is
    ``prompt + committed[:position]``; therefore a later committed token cannot
    influence retrieval.
    """

    normalized = coerce_records(records)
    if min_suffix < 1:
        raise ReplayError("replay min_suffix must be at least one")
    if max_continuation is not None and max_continuation < 1:
        raise ReplayError("max_continuation must be positive when set")
    completed_history: list[ReplayRecord] = []
    opportunities: list[Opportunity] = []
    for record in normalized:
        for position in range(len(record.committed_tokens)):
            query_tokens = record.prompt_tokens + record.committed_tokens[:position]
            target_tokens = record.committed_tokens[position:]
            candidates: list[Candidate] = []
            # This source is a legal prefix of the current request, never its future.
            candidates.extend(
                _to_candidates(
                    tier=SourceTier.CURRENT_CONTEXT,
                    source_record_id=record.record_id,
                    source_tokens=query_tokens,
                    source_token_bytes=0,
                    source_scope=record.scope,
                    provenance_family=record.provenance_family,
                    source_class="current_context",
                    query_tokens=query_tokens,
                    target_tokens=target_tokens,
                    min_suffix=min_suffix,
                    max_continuation=max_continuation,
                )
            )
            if record.current_source_tokens is not None:
                candidates.extend(
                    _to_candidates(
                        tier=SourceTier.CURRENT_SOURCE,
                        source_record_id=f"{record.record_id}:current-source",
                        source_tokens=record.current_source_tokens,
                        source_token_bytes=4 * len(record.current_source_tokens),
                        source_scope=record.scope,
                        provenance_family=record.provenance_family,
                        source_class="current_source",
                        query_tokens=query_tokens,
                        target_tokens=target_tokens,
                        min_suffix=min_suffix,
                        max_continuation=max_continuation,
                    )
                )
            for source in completed_history:
                if source.fingerprints != record.fingerprints or not _authorized_for_persistent(record, source):
                    continue
                candidates.extend(
                    _to_candidates(
                        tier=_tier_for_history(record, source),
                        source_record_id=source.record_id,
                        source_tokens=source.committed_tokens,
                        source_token_bytes=source.logical_source_bytes,
                        source_scope=source.scope,
                        provenance_family=source.provenance_family,
                        source_class=source.source_class,
                        query_tokens=query_tokens,
                        target_tokens=target_tokens,
                        min_suffix=min_suffix,
                        max_continuation=max_continuation,
                    )
                )
            opportunities.append(
                Opportunity(
                    record_id=record.record_id,
                    position=position,
                    target_remaining_tokens=len(target_tokens),
                    candidates=tuple(candidates),
                    request_id=record.request_id,
                )
            )
        # The entire record becomes eligible only after its output is complete.
        completed_history.append(record)
    return ReplayResult(
        records=normalized,
        opportunities=tuple(opportunities),
        census=build_census(normalized),
        future_tokens_used_for_retrieval=False,
    )


def group_opportunities_by_request(result: ReplayResult) -> dict[str, tuple[Opportunity, ...]]:
    """Stable trace-request clusters for metrics that must not treat tokens independently."""

    grouped: defaultdict[str, list[Opportunity]] = defaultdict(list)
    for opportunity in result.opportunities:
        grouped[opportunity.request_cluster_id].append(opportunity)
    return {record_id: tuple(items) for record_id, items in grouped.items()}


def group_opportunities_by_record(result: ReplayResult) -> dict[str, tuple[Opportunity, ...]]:
    """Backward-compatible alias; clusters are keyed by trace request when known."""

    return group_opportunities_by_request(result)
