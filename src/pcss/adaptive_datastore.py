from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from math import ceil
from typing import Mapping, Sequence

from .replay import Fingerprints, ReplayRecord, ScopeIdentity, SourceTier


SUFFIX_LENGTHS = (16, 12, 8, 6, 4)
MAX_CONTINUATION = 32
MAX_POSTINGS_PER_KEY = 128
K_VALUES = (2, 4, 8, 16)
ON_DISK_INDEX_PAGE_BYTES = 4096
ON_DISK_INDEX_CACHE_CAP_BYTES = 64 * 1024 * 1024


class SequenceMode(str, Enum):
    COMPLETION_ONLY = "COMPLETION_ONLY"
    PROMPT_PLUS_COMPLETION = "PROMPT_PLUS_COMPLETION"


class IndexDensity(str, Enum):
    DENSE = "DENSE"
    STRIDE_4 = "STRIDE_4"
    STRIDE_16 = "STRIDE_16"
    LEGACY = "LEGACY"


class PolicyName(str, Enum):
    LEGACY_SPARSE = "legacy_sparse"
    DENSE_RECENT = "dense_recent"
    DENSE_MODAL = "dense_modal"
    DENSE_CONFIDENT = "dense_confident"
    STRIDE_4_CONFIDENT = "stride_4_confident"
    STRIDE_16_CONFIDENT = "stride_16_confident"


class SourceCondition(str, Enum):
    CURRENT_CONTEXT_ONLY = "CURRENT_CONTEXT_ONLY"
    CURRENT_SOURCE_ONLY = "CURRENT_SOURCE_ONLY"
    PERSISTENT_ONLY = "PERSISTENT_ONLY"
    CURRENT_CONTEXT_PLUS_PERSISTENT = "CURRENT_CONTEXT_PLUS_PERSISTENT"
    CURRENT_SOURCE_PLUS_PERSISTENT = "CURRENT_SOURCE_PLUS_PERSISTENT"
    ALL_LEGAL_TIERS = "ALL_LEGAL_TIERS"


@dataclass(frozen=True)
class Posting:
    key: tuple[int, ...]
    source_record_id: str
    source_cursor: int
    chronological_ordinal: int
    source_scope: ScopeIdentity
    source_fingerprints: Fingerprints
    source_class: str
    provenance_family: str
    eligible_for_persistent_store: bool
    continuation: tuple[int, ...]
    source_tier: SourceTier = SourceTier.PERSONAL_PERSISTENT_HISTORY


@dataclass(frozen=True)
class DepthSupport:
    depth: int
    token_id: int
    record_support: int
    occurrence_support: int
    top_share: float
    runner_up_share: float
    margin: float
    surviving_postings: int
    passed_confidence: bool


@dataclass(frozen=True)
class Proposal:
    policy: PolicyName
    suffix_length: int
    key: tuple[int, ...]
    continuation: tuple[int, ...]
    matching_postings: tuple[Posting, ...]
    surviving_postings: tuple[Posting, ...]
    supports: tuple[DepthSupport, ...]
    hit: bool
    abstention_reason: str | None = None

    @property
    def proposed_tokens(self) -> int:
        return len(self.continuation)


@dataclass(frozen=True)
class IngestStats:
    indexed_sequences: int
    indexed_tokens: int
    indexed_cursors: int
    new_keys: int
    postings_added: int
    dropped_postings: int


@dataclass(frozen=True)
class StoreAccounting:
    indexed_sequences: int
    indexed_tokens: int
    indexed_cursors: int
    distinct_keys_by_suffix_length: tuple[tuple[int, int], ...]
    postings: int
    dropped_postings: int
    fanout_p50: float
    fanout_p95: float
    fanout_p99: float
    fanout_max: int
    key_bytes: int
    posting_metadata_bytes: int
    token_payload_bytes: int
    packed_ram_bytes: int
    on_disk_index_bytes: int
    bounded_ram_index_bytes: int
    on_disk_index_page_bytes: int
    on_disk_index_cache_cap_bytes: int
    total_logical_bytes: int
    bytes_per_indexed_source_token: float


@dataclass(frozen=True)
class TrieNode:
    prefix: tuple[int, ...]
    support_records: int
    support_occurrences: int
    children: tuple[tuple[int, "TrieNode"], ...]


@dataclass(frozen=True)
class MultibranchScore:
    tree_hit: bool
    target_path_tokens_found: int
    persistent_unique_tree_coverage: int
    linear_accepted_tokens: int
    tree_accepted_tokens: int
    branching_loss_tokens: int
    tree_nodes: int


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


def _persistent_tier(target: ReplayRecord, posting: Posting) -> SourceTier:
    if "import" in posting.source_class.casefold():
        return SourceTier.IMPORTED_CORPUS
    if target.scope.same_session(posting.source_scope):
        return SourceTier.SAME_SESSION
    if target.scope.same_repository(posting.source_scope):
        return SourceTier.SAME_REPOSITORY
    return SourceTier.PERSONAL_PERSISTENT_HISTORY


def _authorized(target: ReplayRecord, posting: Posting) -> bool:
    return (
        posting.eligible_for_persistent_store
        and target.fingerprints == posting.source_fingerprints
        and target.scope.principal_namespace == posting.source_scope.principal_namespace
        and target.scope.application_id == posting.source_scope.application_id
    )


def _density_for_policy(policy: PolicyName) -> IndexDensity:
    if policy == PolicyName.LEGACY_SPARSE:
        return IndexDensity.LEGACY
    if policy == PolicyName.STRIDE_4_CONFIDENT:
        return IndexDensity.STRIDE_4
    if policy == PolicyName.STRIDE_16_CONFIDENT:
        return IndexDensity.STRIDE_16
    return IndexDensity.DENSE


def _is_confident(suffix_length: int, support: DepthSupport) -> bool:
    if suffix_length in (16, 12):
        return support.occurrence_support >= 1
    if suffix_length == 8:
        return support.occurrence_support >= 2 and support.top_share >= 0.67
    if suffix_length == 6:
        return (
            support.record_support >= 2
            and support.occurrence_support >= 3
            and support.top_share >= 0.75
            and support.margin >= 0.25
        )
    return (
        support.record_support >= 3
        and support.occurrence_support >= 4
        and support.top_share >= 0.80
        and support.margin >= 0.30
    )


class AdaptiveDatastore:
    def __init__(
        self,
        sequence_mode: SequenceMode,
        density: IndexDensity,
        *,
        max_postings_per_key: int = MAX_POSTINGS_PER_KEY,
    ) -> None:
        if max_postings_per_key < 1:
            raise ValueError("max_postings_per_key must be positive")
        self.sequence_mode = SequenceMode(sequence_mode)
        self.density = IndexDensity(density)
        self.max_postings_per_key = max_postings_per_key
        self._postings: dict[tuple[int, ...], list[Posting]] = defaultdict(list)
        self._indexed_records: dict[str, ReplayRecord] = {}
        self._dropped_postings = 0

    @property
    def postings(self) -> Mapping[tuple[int, ...], tuple[Posting, ...]]:
        return {key: tuple(value) for key, value in self._postings.items()}

    @property
    def dropped_postings(self) -> int:
        return self._dropped_postings

    def _sequence(self, record: ReplayRecord) -> tuple[int, ...]:
        if self.sequence_mode == SequenceMode.COMPLETION_ONLY:
            return record.committed_tokens
        return record.prompt_tokens + record.committed_tokens

    def _lengths(self, policy: PolicyName | None = None) -> tuple[int, ...]:
        if policy == PolicyName.LEGACY_SPARSE or self.density == IndexDensity.LEGACY:
            return (64, 32)
        return SUFFIX_LENGTHS

    def _cursors(self, sequence_length: int, suffix_length: int) -> tuple[int, ...]:
        last = sequence_length - 1
        eligible = range(suffix_length, last + 1)
        if self.density == IndexDensity.DENSE:
            return tuple(eligible)
        values = {suffix_length, last}
        divisor = 4 if self.density == IndexDensity.STRIDE_4 else 16
        values.update(cursor for cursor in eligible if cursor % divisor == 0)
        return tuple(sorted(cursor for cursor in values if suffix_length <= cursor <= last))

    def ingest(self, record: ReplayRecord) -> IngestStats:
        if record.record_id in self._indexed_records:
            raise ValueError(f"record already indexed: {record.record_id}")
        self._indexed_records[record.record_id] = record
        if not record.eligible_for_persistent_store:
            return IngestStats(0, 0, 0, 0, 0, 0)
        sequence = self._sequence(record)
        lengths = self._lengths()
        before_keys = len(self._postings)
        added = 0
        dropped_before = self._dropped_postings
        for length in lengths:
            for cursor in self._cursors(len(sequence), length):
                continuation = tuple(sequence[cursor : cursor + MAX_CONTINUATION])
                if not continuation:
                    continue
                key = tuple(sequence[cursor - length : cursor])
                stored_tier = (
                    SourceTier.IMPORTED_CORPUS
                    if "import" in record.source_class.casefold()
                    else SourceTier.PERSONAL_PERSISTENT_HISTORY
                )
                posting = Posting(
                    key=key,
                    source_record_id=record.record_id,
                    source_cursor=cursor,
                    chronological_ordinal=record.chronological_index,
                    source_scope=record.scope,
                    source_fingerprints=record.fingerprints,
                    source_class=record.source_class,
                    provenance_family=record.provenance_family,
                    eligible_for_persistent_store=record.eligible_for_persistent_store,
                    continuation=continuation,
                    source_tier=stored_tier,
                )
                bucket = self._postings[key]
                bucket.append(posting)
                bucket.sort(key=lambda item: (item.chronological_ordinal, item.source_record_id, item.source_cursor))
                if len(bucket) > self.max_postings_per_key:
                    removed = len(bucket) - self.max_postings_per_key
                    del bucket[:removed]
                    self._dropped_postings += removed
                added += 1
        return IngestStats(
            indexed_sequences=1 if added else 0,
            indexed_tokens=len(sequence) if added else 0,
            indexed_cursors=added,
            new_keys=len(self._postings) - before_keys,
            postings_added=added,
            dropped_postings=self._dropped_postings - dropped_before,
        )

    def _eligible_postings(self, target: ReplayRecord, key: tuple[int, ...]) -> tuple[Posting, ...]:
        result = []
        for posting in self._postings.get(key, ()):
            if posting.chronological_ordinal >= target.chronological_index or not _authorized(target, posting):
                continue
            result.append(replace(posting, source_tier=_persistent_tier(target, posting)))
        return tuple(result)

    @staticmethod
    def _support(postings: Sequence[Posting], depth: int, selected: int) -> DepthSupport:
        counts: Counter[int] = Counter(
            posting.continuation[depth]
            for posting in postings
            if depth < len(posting.continuation)
        )
        total = sum(counts.values())
        ordered = sorted(counts.values(), reverse=True)
        top = counts[selected]
        runner = ordered[1] if len(ordered) > 1 else 0
        record_support = len(
            {
                posting.source_record_id
                for posting in postings
                if depth < len(posting.continuation) and posting.continuation[depth] == selected
            }
        )
        top_share = top / total if total else 0.0
        runner_share = runner / total if total else 0.0
        return DepthSupport(
            depth=depth,
            token_id=selected,
            record_support=record_support,
            occurrence_support=top,
            top_share=top_share,
            runner_up_share=runner_share,
            margin=top_share - runner_share,
            surviving_postings=len(postings),
            passed_confidence=False,
        )

    @staticmethod
    def _modal(postings: Sequence[Posting], depth: int) -> int | None:
        available = [posting for posting in postings if depth < len(posting.continuation)]
        if not available:
            return None
        grouped: dict[int, list[Posting]] = defaultdict(list)
        for posting in available:
            grouped[posting.continuation[depth]].append(posting)
        return min(
            grouped,
            key=lambda token: (
                -len({posting.source_record_id for posting in grouped[token]}),
                -len(grouped[token]),
                -max(posting.chronological_ordinal for posting in grouped[token]),
                token,
            ),
        )

    def lookup(
        self,
        target: ReplayRecord,
        query_tokens: Sequence[int],
        *,
        k: int,
        policy: PolicyName,
    ) -> Proposal | None:
        policy = PolicyName(policy)
        if k < 1 or k > MAX_CONTINUATION:
            raise ValueError("k must be between 1 and 32")
        lengths = self._lengths(policy)
        confidence_enabled = policy in {
            PolicyName.DENSE_CONFIDENT,
            PolicyName.STRIDE_4_CONFIDENT,
            PolicyName.STRIDE_16_CONFIDENT,
        }
        for suffix_length in lengths:
            if len(query_tokens) < suffix_length:
                continue
            key = tuple(query_tokens[-suffix_length:])
            matching = self._eligible_postings(target, key)
            if not matching:
                continue
            if policy == PolicyName.LEGACY_SPARSE:
                continuations = {posting.continuation for posting in matching}
                if len(continuations) > 1:
                    return Proposal(policy, suffix_length, key, (), matching, matching, (), True, "ambiguity")
                chosen = max(
                    matching,
                    key=lambda posting: (posting.chronological_ordinal, posting.source_record_id, posting.source_cursor),
                )
                return Proposal(policy, suffix_length, key, chosen.continuation[:k], matching, (chosen,), (), True)
            if policy == PolicyName.DENSE_RECENT:
                chosen = max(
                    matching,
                    key=lambda posting: (posting.chronological_ordinal, posting.source_record_id, posting.source_cursor),
                )
                return Proposal(policy, suffix_length, key, chosen.continuation[:k], matching, (chosen,), (), True)
            surviving = matching
            selected_tokens: list[int] = []
            supports: list[DepthSupport] = []
            for depth in range(k):
                selected = self._modal(surviving, depth)
                if selected is None:
                    break
                support = self._support(surviving, depth, selected)
                passed = not confidence_enabled or _is_confident(suffix_length, support)
                support = replace(support, passed_confidence=passed)
                supports.append(support)
                if not passed:
                    if not selected_tokens:
                        return Proposal(policy, suffix_length, key, (), matching, tuple(surviving), tuple(supports), True, "confidence")
                    break
                selected_tokens.append(selected)
                surviving = tuple(
                    posting
                    for posting in surviving
                    if depth < len(posting.continuation) and posting.continuation[depth] == selected
                )
            return Proposal(
                policy,
                suffix_length,
                key,
                tuple(selected_tokens),
                matching,
                tuple(surviving),
                tuple(supports),
                True,
                None if selected_tokens else "empty",
            )
        return None

    def accounting(self) -> StoreAccounting:
        fanouts = [len(value) for value in self._postings.values()]
        distinct_by_length = Counter(len(key) for key in self._postings)
        indexed_tokens = sum(len(self._sequence(record)) for record in self._indexed_records.values() if record.eligible_for_persistent_store)
        indexed_sequences = sum(record.eligible_for_persistent_store for record in self._indexed_records.values())
        indexed_cursors = sum(fanouts)
        key_bytes = sum(len(key) * 4 for key in self._postings)
        posting_metadata_bytes = indexed_cursors * 48
        token_payload_bytes = sum(
            len(posting.continuation) * 4 for values in self._postings.values() for posting in values
        )
        packed_ram_bytes = len(self._postings) * 40 + indexed_cursors * 32
        on_disk_index_bytes = key_bytes + posting_metadata_bytes
        bounded_ram_index_bytes = min(on_disk_index_bytes, ON_DISK_INDEX_CACHE_CAP_BYTES)
        total_logical_bytes = on_disk_index_bytes + token_payload_bytes
        return StoreAccounting(
            indexed_sequences=indexed_sequences,
            indexed_tokens=indexed_tokens,
            indexed_cursors=indexed_cursors,
            distinct_keys_by_suffix_length=tuple(sorted(distinct_by_length.items())),
            postings=indexed_cursors,
            dropped_postings=self._dropped_postings,
            fanout_p50=_percentile(fanouts, 0.50),
            fanout_p95=_percentile(fanouts, 0.95),
            fanout_p99=_percentile(fanouts, 0.99),
            fanout_max=max(fanouts, default=0),
            key_bytes=key_bytes,
            posting_metadata_bytes=posting_metadata_bytes,
            token_payload_bytes=token_payload_bytes,
            packed_ram_bytes=packed_ram_bytes,
            on_disk_index_bytes=on_disk_index_bytes,
            bounded_ram_index_bytes=bounded_ram_index_bytes,
            on_disk_index_page_bytes=ON_DISK_INDEX_PAGE_BYTES,
            on_disk_index_cache_cap_bytes=ON_DISK_INDEX_CACHE_CAP_BYTES,
            total_logical_bytes=total_logical_bytes,
            bytes_per_indexed_source_token=total_logical_bytes / indexed_tokens if indexed_tokens else 0.0,
        )

    def build_multibranch_tree(self, proposal: Proposal) -> TrieNode:
        root = TrieNode((), 0, 0, ())
        paths: dict[tuple[int, ...], list[Posting]] = defaultdict(list)
        for posting in proposal.matching_postings:
            for depth in range(min(16, len(posting.continuation))):
                paths[posting.continuation[: depth + 1]].append(posting)
        selected: set[tuple[int, ...]] = set()
        frontier = [()]
        while frontier and len(selected) < 32:
            parent = frontier.pop(0)
            children = [path for path in paths if len(path) == len(parent) + 1 and path[:-1] == parent]
            children.sort(
                key=lambda path: (
                    -len({posting.source_record_id for posting in paths[path]}),
                    -len(paths[path]),
                    -max(posting.chronological_ordinal for posting in paths[path]),
                    path[-1],
                )
            )
            for child in children[:4]:
                if child in selected:
                    continue
                selected.add(child)
                frontier.append(child)
                if len(selected) >= 32:
                    break

        def build(prefix: tuple[int, ...]) -> TrieNode:
            child_paths = sorted(
                (path for path in selected if len(path) == len(prefix) + 1 and path[:-1] == prefix),
                key=lambda path: path[-1],
            )
            children = tuple(
                (
                    path[-1],
                    build(path),
                )
                for path in child_paths
            )
            postings = paths.get(prefix, [])
            return TrieNode(
                prefix=prefix,
                support_records=len({posting.source_record_id for posting in postings}),
                support_occurrences=len(postings),
                children=children,
            )

        return build(root.prefix)

    @staticmethod
    def score_multibranch(
        proposal: Proposal,
        tree: TrieNode,
        target_remaining: Sequence[int],
        *,
        linear_accepted_tokens: int,
    ) -> MultibranchScore:
        node = tree
        found = 0
        for token in target_remaining[:16]:
            child = next((child for child_token, child in node.children if child_token == token), None)
            if child is None:
                break
            found += 1
            node = child
        total_nodes = 0
        pending = [tree]
        while pending:
            current = pending.pop()
            total_nodes += len(current.children)
            pending.extend(child for _, child in current.children)
        return MultibranchScore(
            tree_hit=found > 0,
            target_path_tokens_found=found,
            persistent_unique_tree_coverage=found,
            linear_accepted_tokens=linear_accepted_tokens,
            tree_accepted_tokens=found,
            branching_loss_tokens=max(0, found - linear_accepted_tokens),
            tree_nodes=total_nodes,
        )


def policy_density(policy: PolicyName) -> IndexDensity:
    return _density_for_policy(PolicyName(policy))


def policy_names() -> tuple[PolicyName, ...]:
    return tuple(PolicyName)


def source_conditions() -> tuple[SourceCondition, ...]:
    return tuple(SourceCondition)
