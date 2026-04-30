from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from kg_core.io import read_csv_records, read_json, read_jsonl, write_json, write_jsonl
from kg_core.mention_filters import classify_low_information_mention
from relation_extraction.config import resolve_target_relations
from relation_extraction.prepare import PreparedRelationBundle
from relation_extraction.rules import SUPPORTED_RELATION_NAMES, build_relation_rules, build_sentence_trigger_map, match_relation_triggers


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _build_claim_tuple_index(claims: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    claim_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        subject_id = str(claim.get("subject_id", "")).strip()
        predicate = str(claim.get("predicate", "")).strip().upper()
        object_id = str(claim.get("object_id", "")).strip()
        if not subject_id or not predicate or not object_id:
            continue
        claim_index[(subject_id, predicate, object_id)].append(dict(claim))
    return claim_index


def _candidate_token_window(candidate: dict[str, Any], *, margin: int = 4) -> tuple[int, int]:
    token_count = len(candidate.get("tokens", []))
    subject_start = int(candidate.get("subject_token_start", 0))
    subject_end = int(candidate.get("subject_token_end", subject_start))
    object_start = int(candidate.get("object_token_start", 0))
    object_end = int(candidate.get("object_token_end", object_start))
    window_start = max(0, min(subject_start, object_start) - margin)
    window_end = min(token_count, max(subject_end, object_end) + margin)
    return window_start, window_end


def _local_trigger_hits(
    *,
    candidate: dict[str, Any],
    relation_rules: dict[str, Any],
) -> tuple[list[str], list[int]]:
    predicate = str(candidate.get("predicate", "")).strip().upper()
    relation_rule = relation_rules.get(predicate)
    if relation_rule is None:
        return [], [0, 0]
    window_start, window_end = _candidate_token_window(candidate)
    window_tokens = list(candidate.get("tokens", []))[window_start:window_end]
    return match_relation_triggers(tokens=window_tokens, relation_rule=relation_rule), [window_start, window_end]


def _mention_review_reasons(candidate: dict[str, Any]) -> list[str]:
    review_reasons: list[str] = []
    for role in ("subject", "object"):
        mention_text = str(candidate.get(f"{role}_text", "")).strip()
        low_information_reason = classify_low_information_mention(mention_text)
        if low_information_reason:
            review_reasons.append(f"{role}_{low_information_reason.lower()}")
    return review_reasons


def _label_one_candidate(
    *,
    candidate: dict[str, Any],
    relation_rules: dict[str, Any],
    trigger_map: dict[str, list[str]],
    claim_tuple_index: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    subject_entity_id = str(candidate.get("subject_entity_id", "")).strip()
    object_entity_id = str(candidate.get("object_entity_id", "")).strip()
    candidate_predicates = [
        str(item).strip().upper()
        for item in candidate.get("candidate_predicates", [candidate.get("predicate", "")])
        if str(item).strip()
    ]
    allowed_predicates = [
        str(item).strip().upper()
        for item in candidate.get("allowed_predicates", candidate_predicates)
        if str(item).strip()
    ]
    existing_hard_negative_predicates = [
        str(item).strip().upper()
        for item in candidate.get("hard_negative_predicates", [])
        if str(item).strip()
    ]
    if existing_hard_negative_predicates and not candidate_predicates:
        labeled_candidate = dict(candidate)
        labeled_candidate["allowed_predicates"] = allowed_predicates
        labeled_candidate["candidate_predicates"] = []
        labeled_candidate["positive_predicates"] = []
        labeled_candidate["review_predicates"] = []
        labeled_candidate["unknown_predicates"] = []
        labeled_candidate["hard_negative_predicates"] = sorted(set(existing_hard_negative_predicates))
        labeled_candidate["weak_labels_by_predicate"] = {}
        labeled_candidate["weak_label_reasons_by_predicate"] = {}
        labeled_candidate["weak_label"] = "hard_negative"
        labeled_candidate["weak_label_reason"] = str(candidate.get("weak_label_reason") or "conservative_no_relation_evidence")
        labeled_candidate["weak_label_source"] = str(candidate.get("weak_label_source") or "prepare_conservative_sampler")
        labeled_candidate["supervision_tier"] = "hard_negative"
        labeled_candidate["supervision_tiers_by_predicate"] = {}
        labeled_candidate["needs_manual_review"] = False
        labeled_candidate["review_reasons"] = []
        return labeled_candidate
    sentence_trigger_hits_by_predicate = {
        str(key).strip().upper(): list(value)
        for key, value in dict(candidate.get("sentence_trigger_hits") or {}).items()
        if str(key).strip()
    }
    if not sentence_trigger_hits_by_predicate:
        sentence_trigger_hits_by_predicate = {
            relation_name: list(trigger_map.get(relation_name, []))
            for relation_name in candidate_predicates
            if trigger_map.get(relation_name)
        }
    local_trigger_hits_by_predicate = {
        str(key).strip().upper(): list(value)
        for key, value in dict(candidate.get("local_trigger_hits") or {}).items()
        if str(key).strip()
    }
    if not local_trigger_hits_by_predicate:
        for relation_name in candidate_predicates:
            relation_candidate = dict(candidate)
            relation_candidate["predicate"] = relation_name
            local_hits, evidence_token_window = _local_trigger_hits(
                candidate=relation_candidate,
                relation_rules=relation_rules,
            )
            if local_hits:
                local_trigger_hits_by_predicate[relation_name] = local_hits
    else:
        evidence_token_window = list(candidate.get("evidence_token_window", [0, 0]))
    exact_claim_matches = {
        str(key).strip().upper(): [str(item) for item in value if str(item)]
        for key, value in dict(candidate.get("exact_claim_matches") or {}).items()
        if str(key).strip()
    }
    if not exact_claim_matches:
        for relation_name in candidate_predicates:
            exact_rows = claim_tuple_index.get((subject_entity_id, relation_name, object_entity_id), [])
            if exact_rows:
                exact_claim_matches[relation_name] = [
                    str(row.get("claim_id") or "").strip()
                    for row in exact_rows
                    if str(row.get("claim_id") or "").strip()
                ]
    uses_bridge = (
        candidate.get("subject_resolution") == "claim_guided_alias_bridge"
        or candidate.get("object_resolution") == "claim_guided_alias_bridge"
    )
    review_reasons = _mention_review_reasons(candidate)
    bridge_predicates = {
        str(item).strip().upper()
        for item in candidate.get("bridge_predicates", [])
        if str(item).strip()
    }
    weak_labels_by_predicate: dict[str, str] = {}
    weak_label_reasons_by_predicate: dict[str, str] = {}
    positive_predicates: list[str] = []
    review_predicates: list[str] = []
    unknown_predicates: list[str] = []
    hard_negative_predicates: list[str] = []
    supervision_tiers_by_predicate: dict[str, str] = {}

    for predicate in candidate_predicates:
        local_hits = local_trigger_hits_by_predicate.get(predicate, [])
        sentence_hits = sentence_trigger_hits_by_predicate.get(predicate, [])
        exact_claim_ids = exact_claim_matches.get(predicate, [])
        if (exact_claim_ids or predicate in bridge_predicates) and local_hits and not review_reasons:
            weak_label = "ds_strict"
            weak_label_reason = "structured_support_with_local_trigger"
            supervision_tier = "bridge_supervised" if uses_bridge else "strict_text"
            positive_predicates.append(predicate)
        elif (exact_claim_ids or predicate in bridge_predicates) and local_hits:
            weak_label = "manual_review"
            weak_label_reason = "structured_support_with_local_trigger_but_low_information_mention"
            supervision_tier = "manual_review"
            review_predicates.append(predicate)
            review_reasons.append("low_information_mention_needs_manual_check")
        elif exact_claim_ids:
            weak_label = "ds_weak"
            weak_label_reason = "exact_claim_but_no_local_trigger"
            supervision_tier = "weak_text"
            unknown_predicates.append(predicate)
            review_predicates.append(predicate)
            review_reasons.append("exact_claim_without_local_text_evidence")
            if sentence_hits:
                review_reasons.append("sentence_trigger_not_local_to_pair")
        elif local_hits:
            weak_label = "manual_review"
            weak_label_reason = "local_trigger_without_exact_claim_support"
            supervision_tier = "manual_review"
            review_predicates.append(predicate)
            review_reasons.append("local_trigger_without_structured_claim")
        else:
            weak_label = "unknown"
            weak_label_reason = "no_confirmed_structured_support"
            supervision_tier = "unknown"
            unknown_predicates.append(predicate)
        weak_labels_by_predicate[predicate] = weak_label
        weak_label_reasons_by_predicate[predicate] = weak_label_reason
        supervision_tiers_by_predicate[predicate] = supervision_tier

    if positive_predicates:
        record_weak_label = "ds_strict"
        record_weak_label_reason = "has_strict_positive_predicate"
        record_supervision_tier = "bridge_supervised" if uses_bridge else "strict_text"
    elif review_predicates:
        record_weak_label = "manual_review"
        record_weak_label_reason = "has_review_predicate"
        record_supervision_tier = "manual_review"
    else:
        record_weak_label = "unknown"
        record_weak_label_reason = "only_unknown_predicates"
        record_supervision_tier = "unknown"

    labeled_candidate = dict(candidate)
    labeled_candidate["allowed_predicates"] = allowed_predicates
    labeled_candidate["candidate_predicates"] = candidate_predicates
    labeled_candidate["sentence_trigger_hits"] = sentence_trigger_hits_by_predicate
    labeled_candidate["trigger_hits"] = sorted({hit for hits in sentence_trigger_hits_by_predicate.values() for hit in hits})
    labeled_candidate["trigger_hit_count"] = sum(len(hits) for hits in sentence_trigger_hits_by_predicate.values())
    labeled_candidate["local_trigger_hits"] = local_trigger_hits_by_predicate
    labeled_candidate["local_trigger_hit_count"] = sum(len(hits) for hits in local_trigger_hits_by_predicate.values())
    labeled_candidate["evidence_token_window"] = evidence_token_window
    labeled_candidate["exact_claim_matches"] = exact_claim_matches
    labeled_candidate["exact_claim_match"] = bool(exact_claim_matches)
    labeled_candidate["matched_claim_ids"] = sorted({claim_id for ids in exact_claim_matches.values() for claim_id in ids})
    labeled_candidate["matched_claim_count"] = sum(len(ids) for ids in exact_claim_matches.values())
    labeled_candidate["positive_predicates"] = sorted(set(positive_predicates))
    labeled_candidate["weak_labels_by_predicate"] = weak_labels_by_predicate
    labeled_candidate["weak_label_reasons_by_predicate"] = weak_label_reasons_by_predicate
    labeled_candidate["review_predicates"] = sorted(set(review_predicates))
    labeled_candidate["unknown_predicates"] = sorted(set(unknown_predicates))
    labeled_candidate["hard_negative_predicates"] = hard_negative_predicates
    labeled_candidate["weak_label"] = record_weak_label
    labeled_candidate["weak_label_reason"] = record_weak_label_reason
    labeled_candidate["weak_label_source"] = "distant_supervision"
    labeled_candidate["supervision_tier"] = record_supervision_tier
    labeled_candidate["supervision_tiers_by_predicate"] = supervision_tiers_by_predicate
    labeled_candidate["uses_claim_guided_alias_bridge"] = uses_bridge
    labeled_candidate["needs_manual_review"] = bool(review_reasons)
    labeled_candidate["review_reasons"] = sorted(set(review_reasons))
    return labeled_candidate


def weak_label_relation_candidates(
    *,
    prepared_bundle: PreparedRelationBundle,
    relation_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inferred_relation_names = sorted(
        {
            str(candidate.get("predicate", "")).strip().upper()
            for candidate in prepared_bundle.relation_candidates
            if str(candidate.get("predicate", "")).strip()
        }
    )
    normalized_relation_names = tuple(
        name.strip().upper() for name in (relation_names or inferred_relation_names or SUPPORTED_RELATION_NAMES) if name and name.strip()
    )
    entity_index = {}
    for mention in prepared_bundle.resolved_mentions:
        entity_id = str(mention.get("entity_id") or "").strip()
        if not entity_id:
            continue
        entity_index.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": mention.get("linked_entity_type"),
            },
        )

    relation_rules = build_relation_rules(
        ontology=prepared_bundle.ontology,
        claims=prepared_bundle.claims,
        entity_index=entity_index,
        relation_names=normalized_relation_names,
    )
    claim_tuple_index = _build_claim_tuple_index(prepared_bundle.claims)
    sentence_trigger_cache: dict[str, dict[str, list[str]]] = {}
    weak_label_counts: Counter[str] = Counter()
    weak_label_reason_counts: Counter[str] = Counter()
    predicate_counts: Counter[str] = Counter()
    predicate_strict_counts: Counter[str] = Counter()
    predicate_weak_counts: Counter[str] = Counter()
    predicate_na_counts: Counter[str] = Counter()
    hard_negative_count = 0
    predicate_manual_review_counts: Counter[str] = Counter()
    pair_source_counts: Counter[str] = Counter()
    supervision_tier_counts: Counter[str] = Counter()
    bridge_usage_counts: Counter[str] = Counter()
    exact_claim_counts: Counter[str] = Counter()
    trigger_positive_counts: Counter[str] = Counter()
    labeled_candidates: list[dict[str, Any]] = []

    for candidate in prepared_bundle.relation_candidates:
        sentence_id = str(candidate.get("sentence_id", "")).strip()
        if sentence_id not in sentence_trigger_cache:
            sentence_trigger_cache[sentence_id] = build_sentence_trigger_map(
                tokens=list(candidate.get("tokens", [])),
                relation_rules=relation_rules,
            )
        labeled_candidate = _label_one_candidate(
            candidate=candidate,
            relation_rules=relation_rules,
            trigger_map=sentence_trigger_cache[sentence_id],
            claim_tuple_index=claim_tuple_index,
        )
        labeled_candidates.append(labeled_candidate)

        candidate_predicates = [
            str(item).strip().upper()
            for item in labeled_candidate.get("candidate_predicates", [labeled_candidate.get("predicate", "")])
            if str(item).strip()
        ]
        weak_label = str(labeled_candidate.get("weak_label", "NA")).strip()
        weak_label_counts[weak_label] += 1
        if weak_label == "hard_negative":
            hard_negative_count += 1
        weak_label_reason_counts[str(labeled_candidate.get("weak_label_reason", ""))] += 1
        supervision_tier_counts[str(labeled_candidate.get("supervision_tier", ""))] += 1
        pair_source_counts[str(labeled_candidate.get("pair_source", ""))] += 1
        for predicate in candidate_predicates:
            predicate_counts[predicate] += 1
            predicate_label = dict(labeled_candidate.get("weak_labels_by_predicate", {})).get(predicate, "unknown")
            if labeled_candidate.get("uses_claim_guided_alias_bridge"):
                bridge_usage_counts[predicate] += 1
            if predicate in dict(labeled_candidate.get("exact_claim_matches", {})):
                exact_claim_counts[predicate] += 1
            if predicate in dict(labeled_candidate.get("sentence_trigger_hits", {})):
                trigger_positive_counts[predicate] += 1
            if predicate_label == "ds_strict":
                predicate_strict_counts[predicate] += 1
            elif predicate_label == "ds_weak":
                predicate_weak_counts[predicate] += 1
            elif predicate_label == "manual_review":
                predicate_manual_review_counts[predicate] += 1
            elif predicate_label == "hard_negative":
                predicate_na_counts[predicate] += 1

    summary = {
        "relation_names": list(normalized_relation_names),
        "candidate_count": len(labeled_candidates),
        "weak_label_counts": _counter_to_sorted_dict(weak_label_counts),
        "weak_label_reason_counts": _counter_to_sorted_dict(weak_label_reason_counts),
        "supervision_tier_counts": _counter_to_sorted_dict(supervision_tier_counts),
        "predicate_counts": _counter_to_sorted_dict(predicate_counts),
        "predicate_ds_strict_counts": _counter_to_sorted_dict(predicate_strict_counts),
        "predicate_ds_weak_counts": _counter_to_sorted_dict(predicate_weak_counts),
        "predicate_manual_review_counts": _counter_to_sorted_dict(predicate_manual_review_counts),
        "predicate_na_counts": _counter_to_sorted_dict(predicate_na_counts),
        "hard_negative_count": hard_negative_count,
        "pair_source_counts": _counter_to_sorted_dict(pair_source_counts),
        "bridge_usage_counts": _counter_to_sorted_dict(bridge_usage_counts),
        "exact_claim_counts": _counter_to_sorted_dict(exact_claim_counts),
        "trigger_positive_counts": _counter_to_sorted_dict(trigger_positive_counts),
        "local_trigger_positive_counts": _counter_to_sorted_dict(
            Counter(
                predicate
                for record in labeled_candidates
                for predicate in dict(record.get("local_trigger_hits", {}))
            )
        ),
        "manual_review_count": sum(1 for record in labeled_candidates if record.get("needs_manual_review")),
        "prepare_summary": prepared_bundle.summary,
    }
    return labeled_candidates, summary


def _review_queue_records(labeled_candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    review_records: list[dict[str, Any]] = []
    for candidate in labeled_candidates:
        if not candidate.get("needs_manual_review"):
            continue
        review_records.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "sentence_id": candidate.get("sentence_id"),
                "doc_id": candidate.get("doc_id"),
                "source_id": candidate.get("source_id"),
                "predicate": candidate.get("predicate"),
                "candidate_predicates": candidate.get("candidate_predicates", []),
                "positive_predicates": candidate.get("positive_predicates", []),
                "review_predicates": candidate.get("review_predicates", []),
                "unknown_predicates": candidate.get("unknown_predicates", []),
                "weak_labels_by_predicate": candidate.get("weak_labels_by_predicate", {}),
                "weak_label": candidate.get("weak_label"),
                "weak_label_reason": candidate.get("weak_label_reason"),
                "review_reasons": candidate.get("review_reasons", []),
                "subject_text": candidate.get("subject_text"),
                "subject_entity_id": candidate.get("subject_entity_id"),
                "subject_canonical_name": candidate.get("subject_canonical_name"),
                "object_text": candidate.get("object_text"),
                "object_entity_id": candidate.get("object_entity_id"),
                "object_canonical_name": candidate.get("object_canonical_name"),
                "trigger_hits": candidate.get("trigger_hits", []),
                "local_trigger_hits": candidate.get("local_trigger_hits", []),
                "exact_claim_match": candidate.get("exact_claim_match", False),
                "matched_claim_ids": candidate.get("matched_claim_ids", []),
                "bridge_predicates": candidate.get("bridge_predicates", []),
                "text": candidate.get("text"),
            }
        )
    return review_records


def weak_label_relation_candidates_from_paths(
    *,
    pair_candidates_path: str,
    entities_csv_path: str,
    claims_csv_path: str,
    ontology_path: str,
    relation_names: Sequence[str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    relation_candidates = read_jsonl(pair_candidates_path)
    claims = read_csv_records(claims_csv_path)
    ontology = read_json(ontology_path)
    entity_type_by_id = {
        str(row.get("entity_id", "")).strip(): str(row.get("entity_type", "")).strip()
        for row in read_csv_records(entities_csv_path)
        if str(row.get("entity_id", "")).strip()
    }
    resolved_mentions: list[dict[str, Any]] = []
    for candidate in relation_candidates:
        for role in ("subject", "object"):
            entity_id = str(candidate.get(f"{role}_entity_id", "")).strip()
            if not entity_id:
                continue
            resolved_mentions.append(
                {
                    "entity_id": entity_id,
                    "linked_entity_type": candidate.get(f"{role}_entity_type") or entity_type_by_id.get(entity_id),
                }
            )
    configured_relation_names = relation_names
    if configured_relation_names is None and config_path:
        configured_relation_names = read_json(config_path).get("target_relations")
    resolved_relation_names = resolve_target_relations(
        ontology_path=Path(ontology_path),
        configured_target_relations=configured_relation_names,
        trigger_relation_names=SUPPORTED_RELATION_NAMES,
    )
    prepared_bundle = PreparedRelationBundle(
        sentences=[],
        resolved_mentions=resolved_mentions,
        relation_candidates=relation_candidates,
        summary={
            "candidate_count": len(relation_candidates),
            "candidate_source_path": str(pair_candidates_path),
        },
        ontology=ontology,
        claims=claims,
    )
    labeled_candidates, summary = weak_label_relation_candidates(
        prepared_bundle=prepared_bundle,
        relation_names=resolved_relation_names,
    )
    return {
        "relation_candidates": prepared_bundle.relation_candidates,
        "weak_labeled_candidates": labeled_candidates,
        "summary": summary,
        "ontology": prepared_bundle.ontology,
        "claims": prepared_bundle.claims,
    }


def weak_label_relations(
    *,
    pair_candidates_path: str | Path,
    entities_csv_path: str | Path,
    claims_csv_path: str | Path,
    ontology_path: str | Path,
    output_path: str | Path,
    review_output_path: str | Path | None = None,
    relation_names: Sequence[str] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = weak_label_relation_candidates_from_paths(
        pair_candidates_path=str(pair_candidates_path),
        entities_csv_path=str(entities_csv_path),
        claims_csv_path=str(claims_csv_path),
        ontology_path=str(ontology_path),
        relation_names=relation_names,
        config_path=str(config_path) if config_path else None,
    )
    resolved_output_path = Path(output_path)
    labeled_candidates = list(payload["weak_labeled_candidates"])
    write_jsonl(resolved_output_path, labeled_candidates)
    write_json(resolved_output_path.with_suffix(".summary.json"), payload["summary"])
    resolved_review_output_path = Path(review_output_path) if review_output_path else resolved_output_path.with_name("distant_label_review_queue.jsonl")
    review_records = _review_queue_records(labeled_candidates)
    write_jsonl(resolved_review_output_path, review_records)
    return {
        "weak_labeled_count": len(labeled_candidates),
        "output_path": resolved_output_path.as_posix(),
        "summary_path": resolved_output_path.with_suffix(".summary.json").as_posix(),
        "review_queue_count": len(review_records),
        "review_queue_path": resolved_review_output_path.as_posix(),
        "summary": payload["summary"],
    }
