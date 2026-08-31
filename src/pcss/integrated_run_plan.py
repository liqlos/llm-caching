"""Sealed, future-token-free qualification plans for integrated PCSS runs.

This module selects only from chronological replay evidence.  Target-output
comparison lives in the separately labelled post-hoc section and is used only
to decide whether a workload is worth an integrated measurement; it can never
affect selection, admission, or ranking.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .canonical import ContractError, canonical_json, require_exact_keys, require_int, require_mapping, require_string, sha256
from .offline_runner import load_trace_fixture
from .replay import Candidate, ReplayRecord, ReplayResult, SourceTier, longest_exact_suffix, replay_chronologically


PLAN_SCHEMA_VERSION = "pcss.integrated-run-plan/v1"
POLICY_SCHEMA_VERSION = "pcss.integrated-run-policy/v1"
_PERSISTENT_TIERS = frozenset({
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
})
_TIER_RANK = {tier: index for index, tier in enumerate((
    SourceTier.SAME_SESSION,
    SourceTier.SAME_REPOSITORY,
    SourceTier.PERSONAL_PERSISTENT_HISTORY,
    SourceTier.IMPORTED_CORPUS,
))}


@dataclass(frozen=True)
class QualificationPolicy:
    """Explicit selection policy; abstract simulator costs are intentionally absent."""

    k: int = 2
    max_history_bytes: int | None = None
    require_cross_session: bool = True
    allowed_persistent_tiers: tuple[SourceTier, ...] = (
        SourceTier.SAME_SESSION,
        SourceTier.SAME_REPOSITORY,
        SourceTier.PERSONAL_PERSISTENT_HISTORY,
        SourceTier.IMPORTED_CORPUS,
    )

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ContractError("qualification policy K must be positive")
        if self.max_history_bytes is not None and self.max_history_bytes < 0:
            raise ContractError("qualification policy max_history_bytes must be non-negative")
        if not self.allowed_persistent_tiers or any(tier not in _PERSISTENT_TIERS for tier in self.allowed_persistent_tiers):
            raise ContractError("qualification policy must contain only persistent source tiers")
        if len(set(self.allowed_persistent_tiers)) != len(self.allowed_persistent_tiers):
            raise ContractError("qualification policy source tiers must not repeat")

    def canonical(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "k": self.k,
            "max_history_bytes": self.max_history_bytes,
            "require_cross_session": self.require_cross_session,
            "allowed_persistent_tiers": [tier.value for tier in self.allowed_persistent_tiers],
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "QualificationPolicy":
        raw = require_mapping(value, "qualification policy")
        require_exact_keys(raw, {"schema_version", "k", "max_history_bytes", "require_cross_session", "allowed_persistent_tiers"}, "qualification policy")
        if require_string(raw["schema_version"], "qualification policy.schema_version") != POLICY_SCHEMA_VERSION:
            raise ContractError("qualification policy has unsupported schema_version")
        k = require_int(raw["k"], "qualification policy.k", minimum=1)
        history = raw["max_history_bytes"]
        if history is not None:
            history = require_int(history, "qualification policy.max_history_bytes")
        if not isinstance(raw["require_cross_session"], bool):
            raise ContractError("qualification policy.require_cross_session must be a boolean")
        tiers = raw["allowed_persistent_tiers"]
        if not isinstance(tiers, list) or not tiers:
            raise ContractError("qualification policy.allowed_persistent_tiers must be a non-empty array")
        try:
            parsed = tuple(SourceTier(require_string(item, "qualification policy.allowed_persistent_tiers[]")) for item in tiers)
        except ValueError as error:
            raise ContractError("qualification policy contains an unknown source tier") from error
        return cls(k=k, max_history_bytes=history, require_cross_session=raw["require_cross_session"], allowed_persistent_tiers=parsed)


def _candidate_rank(candidate: Candidate) -> tuple[int, int, int, str, int]:
    """Only source metadata available before the target output is read."""

    return (-candidate.matched_suffix_length, _TIER_RANK[candidate.tier], candidate.source_token_bytes, candidate.source_record_id, candidate.source_cursor)


def _cross_session(candidate: Candidate, target: ReplayRecord) -> bool:
    return not target.scope.same_session(candidate.source_scope)


def _legal_completed_candidate(
    candidate: Candidate,
    *,
    target: ReplayRecord,
    source: ReplayRecord | None,
    position: int,
) -> bool:
    """Reject replay evidence whose source metadata no longer binds to its record.

    ``Opportunity`` is a derived object, so the sealed-plan boundary must not
    trust a candidate merely because it was present in one.  All checks here
    use source and target material that existed before the target token being
    proposed is verified.
    """

    if source is None or source.order_key >= target.order_key:
        return False
    if not source.eligible_for_persistent_store or source.fingerprints != target.fingerprints:
        return False
    if (
        source.scope.principal_namespace != target.scope.principal_namespace
        or source.scope.application_id != target.scope.application_id
        or candidate.source_scope != source.scope
        or candidate.provenance_family != source.provenance_family
        or candidate.source_class != source.source_class
        or candidate.source_token_bytes != source.logical_source_bytes
    ):
        return False
    if candidate.source_cursor < 0 or candidate.source_cursor >= len(source.committed_tokens):
        return False
    if candidate.continuation != source.committed_tokens[
        candidate.source_cursor : candidate.source_cursor + len(candidate.continuation)
    ] or not candidate.continuation:
        return False
    if candidate.matched_suffix_length != longest_exact_suffix(
        target.prompt_tokens + target.committed_tokens[:position],
        source.committed_tokens,
        candidate.source_cursor,
    ):
        return False
    expected_tier = (
        SourceTier.IMPORTED_CORPUS if "import" in source.source_class.casefold()
        else SourceTier.CURRENT_SOURCE if target.scope.same_current_source(source.scope)
        else SourceTier.SAME_SESSION if target.scope.same_session(source.scope)
        else SourceTier.SAME_REPOSITORY if target.scope.same_repository(source.scope)
        else SourceTier.PERSONAL_PERSISTENT_HISTORY
    )
    return candidate.tier == expected_tier


def _decision(candidate: Candidate, target: ReplayRecord, *, position: int, proposed_k: int, admitted: bool) -> dict[str, Any]:
    return {
        "target_record_id": target.record_id,
        "target_request_id": target.request_id or target.record_id,
        "target_position": position,
        "source_record_id": candidate.source_record_id,
        "source_cursor": candidate.source_cursor,
        "source_tier": candidate.tier.value,
        "source_provenance_family": candidate.provenance_family,
        "source_class": candidate.source_class,
        "cross_session": _cross_session(candidate, target),
        "matched_suffix_length": candidate.matched_suffix_length,
        "proposed_k": proposed_k,
        "source_token_bytes": candidate.source_token_bytes,
        "admission_status": "admitted" if admitted else "not_admitted",
        "compatibility": {"fingerprints_match": True, "scope_authorized": True},
    }


def _posthoc(candidate: Candidate, *, proposed_k: int, cheaper_coverage: int) -> dict[str, Any]:
    accepted = min(candidate.accepted_prefix_length, proposed_k)
    return {
        "accepted_prefix_length": accepted,
        "full_accept": accepted == proposed_k,
        "correction": accepted < proposed_k,
        "unique_persistent_credit": max(0, accepted - min(cheaper_coverage, proposed_k)),
    }


def _status(*, future: bool, eligible_cross_session: int, selected: int, unique_credit: int) -> str:
    if future:
        return "NOT_QUALIFIED_FUTURE_RETRIEVAL"
    if eligible_cross_session == 0:
        return "NOT_QUALIFIED_NO_ELIGIBLE_CROSS_SESSION_DECISION"
    if selected == 0:
        return "NOT_QUALIFIED_NO_SELECTED_CASES"
    if unique_credit == 0:
        return "NOT_QUALIFIED_ZERO_UNIQUE_PERSISTENT_CREDIT"
    return "QUALIFIED"


def build_integrated_run_plan(result: ReplayResult, *, policy: QualificationPolicy = QualificationPolicy()) -> dict[str, Any]:
    """Return the sole admissible input shape for a future integrated run.

    The plan itself makes no performance claim.  It contains both a sealed
    pre-target decision list and separately labelled post-hoc qualification
    observations so a runner can consume only ``selected_cases``.
    """

    if result.future_tokens_used_for_retrieval:
        # Keep a sealed negative outcome rather than permitting any selected input.
        selected_cases: list[dict[str, Any]] = []
        abstentions: list[dict[str, Any]] = []
        eligible_cross_session = 0
        unique_credit = 0
    else:
        records = {record.record_id: record for record in result.records}
        opportunities = {(item.record_id, item.position): item for item in result.opportunities}
        active: set[str] = set()
        order: deque[tuple[str, int]] = deque()
        active_bytes = 0
        selected_cases = []
        abstentions = []
        eligible_cross_session = 0
        unique_credit = 0
        for target in result.records:
            for position in range(len(target.committed_tokens)):
                opportunity = opportunities[(target.record_id, position)]
                eligible = [
                    candidate for candidate in opportunity.candidates
                    if candidate.tier in policy.allowed_persistent_tiers
                    and candidate.source_record_id in active
                    and _legal_completed_candidate(
                        candidate,
                        target=target,
                        source=records.get(candidate.source_record_id),
                        position=position,
                    )
                ]
                if any(_cross_session(candidate, target) for candidate in eligible):
                    eligible_cross_session += 1
                if policy.require_cross_session:
                    eligible = [candidate for candidate in eligible if _cross_session(candidate, target)]
                candidate = min(eligible, key=_candidate_rank, default=None)
                if candidate is None:
                    abstentions.append({
                        "target_record_id": target.record_id,
                        "target_request_id": target.request_id or target.record_id,
                        "target_position": position,
                        "reason": "no_eligible_persistent_pre_target_candidate",
                    })
                    continue
                proposed_k = min(policy.k, len(candidate.continuation), opportunity.target_remaining_tokens)
                if proposed_k < 1:
                    raise ContractError("selected plan candidate has no usable continuation")
                decision = _decision(candidate, target, position=position, proposed_k=proposed_k, admitted=True)
                cheaper = max(opportunity.context_prefix, opportunity.current_source_prefix)
                posthoc = _posthoc(candidate, proposed_k=proposed_k, cheaper_coverage=cheaper)
                unique_credit += posthoc["unique_persistent_credit"]
                selected_cases.append({"decision": decision, "posthoc": posthoc})
            active.add(target.record_id)
            order.append((target.record_id, target.logical_source_bytes))
            active_bytes += target.logical_source_bytes
            while policy.max_history_bytes is not None and active_bytes > policy.max_history_bytes:
                evicted_id, evicted_bytes = order.popleft()
                active.remove(evicted_id)
                active_bytes -= evicted_bytes

    trace_payload = [_record_identity(record) for record in result.records]
    trace_sha256 = f"sha256:{sha256(trace_payload)}"
    policy_value = policy.canonical()
    policy_fingerprint = f"sha256:{sha256(policy_value)}"
    status = _status(future=result.future_tokens_used_for_retrieval, eligible_cross_session=eligible_cross_session, selected=len(selected_cases), unique_credit=unique_credit)
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "sealed_future_token_free_workload_qualification_plan",
        "runtime_claim": "workload qualification only; no latency, throughput, or speedup claim",
        "trace_sha256": trace_sha256,
        "policy": policy_value,
        "policy_fingerprint": policy_fingerprint,
        "future_tokens_used_for_retrieval": result.future_tokens_used_for_retrieval,
        "status": status,
        "qualification": {
            "eligible_cross_session_decisions": eligible_cross_session,
            "selected_case_count": len(selected_cases),
            "unique_persistent_credit": unique_credit,
            "default_requires_cross_session": True,
        },
        "selected_cases": selected_cases,
        "abstentions": abstentions,
    }
    plan = {**payload, "plan_fingerprint": f"sha256:{sha256(payload)}"}
    validate_integrated_run_plan(plan)
    return plan


def _record_identity(record: ReplayRecord) -> dict[str, Any]:
    """The exact replay inputs that determine legal eligibility and ordering."""

    return {
        "record_id": record.record_id,
        "order_key": record.order_key,
        "prompt_tokens": record.prompt_tokens,
        "committed_tokens": record.committed_tokens,
        "fingerprints": asdict(record.fingerprints),
        "scope": asdict(record.scope),
        "source_class": record.source_class,
        "provenance_family": record.provenance_family,
        "eligible_for_persistent_store": record.eligible_for_persistent_store,
        "current_source_tokens": record.current_source_tokens,
        "declared_source_bytes": record.declared_source_bytes,
        "request_id": record.request_id,
    }


def validate_integrated_run_plan(value: Any) -> None:
    """Fail closed on tampering or a plan that tries to carry a runtime claim."""

    plan = require_mapping(value, "integrated run plan")
    required = {"schema_version", "kind", "runtime_claim", "trace_sha256", "policy", "policy_fingerprint", "future_tokens_used_for_retrieval", "status", "qualification", "selected_cases", "abstentions", "plan_fingerprint"}
    require_exact_keys(plan, required, "integrated run plan")
    if require_string(plan["schema_version"], "integrated run plan.schema_version") != PLAN_SCHEMA_VERSION:
        raise ContractError("integrated run plan has unsupported schema_version")
    if require_string(plan["kind"], "integrated run plan.kind") != "sealed_future_token_free_workload_qualification_plan":
        raise ContractError("integrated run plan has invalid kind")
    if require_string(plan["runtime_claim"], "integrated run plan.runtime_claim") != "workload qualification only; no latency, throughput, or speedup claim":
        raise ContractError("integrated run plan may not make a runtime claim")
    if not require_string(plan["trace_sha256"], "integrated run plan.trace_sha256").startswith("sha256:"):
        raise ContractError("integrated run plan trace hash must be sha256")
    policy = QualificationPolicy.from_mapping(plan["policy"])
    if plan["policy_fingerprint"] != f"sha256:{sha256(policy.canonical())}":
        raise ContractError("integrated run plan policy fingerprint mismatch")
    if not isinstance(plan["future_tokens_used_for_retrieval"], bool):
        raise ContractError("integrated run plan future_tokens_used_for_retrieval must be a boolean")
    if not isinstance(plan["selected_cases"], list) or not isinstance(plan["abstentions"], list):
        raise ContractError("integrated run plan cases and abstentions must be arrays")
    qualification = require_mapping(plan["qualification"], "integrated run plan.qualification")
    require_exact_keys(qualification, {"eligible_cross_session_decisions", "selected_case_count", "unique_persistent_credit", "default_requires_cross_session"}, "integrated run plan.qualification")
    for key in ("eligible_cross_session_decisions", "selected_case_count", "unique_persistent_credit"):
        require_int(qualification[key], f"integrated run plan.qualification.{key}")
    if qualification["default_requires_cross_session"] is not True:
        raise ContractError("integrated run plan must retain the cross-session default")
    case_credit = 0
    decision_keys = {
        "target_record_id", "target_request_id", "target_position", "source_record_id", "source_cursor",
        "source_tier", "source_provenance_family", "source_class", "cross_session", "matched_suffix_length",
        "proposed_k", "source_token_bytes", "admission_status", "compatibility",
    }
    posthoc_keys = {"accepted_prefix_length", "full_accept", "correction", "unique_persistent_credit"}
    for index, case in enumerate(plan["selected_cases"]):
        case_value = require_mapping(case, f"integrated run plan.selected_cases[{index}]")
        require_exact_keys(case_value, {"decision", "posthoc"}, f"integrated run plan.selected_cases[{index}]")
        decision = require_mapping(case_value["decision"], f"integrated run plan.selected_cases[{index}].decision")
        require_exact_keys(decision, decision_keys, f"integrated run plan.selected_cases[{index}].decision")
        for field in ("target_record_id", "target_request_id", "source_record_id", "source_provenance_family", "source_class"):
            require_string(decision[field], f"integrated run plan.selected_cases[{index}].decision.{field}")
        for field in ("target_position", "source_cursor", "matched_suffix_length", "proposed_k", "source_token_bytes"):
            require_int(decision[field], f"integrated run plan.selected_cases[{index}].decision.{field}")
        if decision["proposed_k"] < 1 or decision["proposed_k"] > policy.k:
            raise ContractError("integrated run plan case K is outside policy")
        try:
            tier = SourceTier(require_string(decision["source_tier"], "integrated run plan case source tier"))
        except ValueError as error:
            raise ContractError("integrated run plan case source tier is invalid") from error
        if tier not in policy.allowed_persistent_tiers or decision["admission_status"] != "admitted":
            raise ContractError("integrated run plan case is not an admitted policy-eligible persistent candidate")
        if not isinstance(decision["cross_session"], bool) or (policy.require_cross_session and not decision["cross_session"]):
            raise ContractError("integrated run plan case violates cross-session policy")
        compatibility = require_mapping(decision["compatibility"], "integrated run plan case compatibility")
        require_exact_keys(compatibility, {"fingerprints_match", "scope_authorized"}, "integrated run plan case compatibility")
        if compatibility["fingerprints_match"] is not True or compatibility["scope_authorized"] is not True:
            raise ContractError("integrated run plan case lacks compatible authorized source scope")
        posthoc = require_mapping(case_value["posthoc"], f"integrated run plan.selected_cases[{index}].posthoc")
        require_exact_keys(posthoc, posthoc_keys, f"integrated run plan.selected_cases[{index}].posthoc")
        for field in ("accepted_prefix_length", "unique_persistent_credit"):
            require_int(posthoc[field], f"integrated run plan.selected_cases[{index}].posthoc.{field}")
        if posthoc["accepted_prefix_length"] > decision["proposed_k"]:
            raise ContractError("integrated run plan post-hoc acceptance exceeds proposed K")
        if posthoc["unique_persistent_credit"] > posthoc["accepted_prefix_length"]:
            raise ContractError("integrated run plan unique credit exceeds accepted prefix")
        if not isinstance(posthoc["full_accept"], bool) or not isinstance(posthoc["correction"], bool):
            raise ContractError("integrated run plan post-hoc fields must be booleans")
        if posthoc["full_accept"] != (posthoc["accepted_prefix_length"] == decision["proposed_k"]) or posthoc["correction"] == posthoc["full_accept"]:
            raise ContractError("integrated run plan post-hoc feedback is inconsistent")
        case_credit += posthoc["unique_persistent_credit"]
    if qualification["selected_case_count"] != len(plan["selected_cases"]) or qualification["unique_persistent_credit"] != case_credit:
        raise ContractError("integrated run plan qualification totals are inconsistent")
    abstention_keys = {"target_record_id", "target_request_id", "target_position", "reason"}
    for index, abstention in enumerate(plan["abstentions"]):
        abstention_value = require_mapping(abstention, f"integrated run plan.abstentions[{index}]")
        require_exact_keys(abstention_value, abstention_keys, f"integrated run plan.abstentions[{index}]")
        if abstention_value["reason"] != "no_eligible_persistent_pre_target_candidate":
            raise ContractError("integrated run plan abstention reason is invalid")
    expected_status = _status(
        future=plan["future_tokens_used_for_retrieval"],
        eligible_cross_session=qualification["eligible_cross_session_decisions"],
        selected=qualification["selected_case_count"],
        unique_credit=qualification["unique_persistent_credit"],
    )
    if plan["status"] != expected_status or (plan["status"] == "QUALIFIED" and not plan["selected_cases"]):
        raise ContractError("integrated run plan qualification status is inconsistent")
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    if plan["plan_fingerprint"] != f"sha256:{sha256(payload)}":
        raise ContractError("integrated run plan fingerprint mismatch")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = QualificationPolicy()
    if args.policy is not None:
        try:
            policy = QualificationPolicy.from_mapping(json.loads(args.policy.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"unable to load qualification policy: {args.policy}") from error
    plan = build_integrated_run_plan(replay_chronologically(load_trace_fixture(args.trace)), policy=policy)
    output = canonical_json(plan)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
