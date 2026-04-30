from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from kg_core.io import read_csv_records, read_json, read_jsonl, write_json, write_jsonl
from relation_extraction.prepare import PreparedRelationBundle
from relation_extraction.rules import DEFAULT_RELATION_NAMES, build_relation_rules, build_sentence_trigger_map


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


def _label_one_candidate(
    *,
    candidate: dict[str, Any],
    trigger_map: dict[str, list[str]],
    claim_tuple_index: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    predicate = str(candidate.get("predicate", "")).strip().upper()
    subject_entity_id = str(candidate.get("subject_entity_id", "")).strip()
    object_entity_id = str(candidate.get("object_entity_id", "")).strip()
    trigger_hits = list(trigger_map.get(predicate, []))
    exact_claim_rows = claim_tuple_index.get((subject_entity_id, predicate, object_entity_id), [])
    exact_claim_match = bool(exact_claim_rows)
    uses_bridge = (
        candidate.get("subject_resolution") == "claim_guided_alias_bridge"
        or candidate.get("object_resolution") == "claim_guided_alias_bridge"
    )
    if exact_claim_match and trigger_hits:
        weak_label = "ds_strict"
        weak_label_reason = "exact_claim_with_trigger"
        supervision_tier = "bridge_supervised" if uses_bridge else "strict_text"
    elif exact_claim_match:
        weak_label = "ds_weak"
        weak_label_reason = "exact_claim_but_evidence_is_weaker"
        supervision_tier = "bridge_supervised" if uses_bridge else "weak_text"
    elif uses_bridge and predicate in set(candidate.get("bridge_predicates", [])) and trigger_hits:
        weak_label = "ds_weak"
        weak_label_reason = "bridge_predicate_supported_by_trigger"
        supervision_tier = "bridge_supervised"
    else:
        weak_label = "NA"
        weak_label_reason = "no_exact_claim_support"
        supervision_tier = "na"

    labeled_candidate = dict(candidate)
    labeled_candidate["trigger_hits"] = trigger_hits
    labeled_candidate["trigger_hit_count"] = len(trigger_hits)
    labeled_candidate["exact_claim_match"] = exact_claim_match
    labeled_candidate["matched_claim_ids"] = [row.get("claim_id") for row in exact_claim_rows]
    labeled_candidate["matched_claim_count"] = len(exact_claim_rows)
    labeled_candidate["weak_label"] = weak_label
    labeled_candidate["weak_label_reason"] = weak_label_reason
    labeled_candidate["weak_label_source"] = "distant_supervision"
    labeled_candidate["supervision_tier"] = supervision_tier
    labeled_candidate["uses_claim_guided_alias_bridge"] = uses_bridge
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
        name.strip().upper() for name in (relation_names or inferred_relation_names or DEFAULT_RELATION_NAMES) if name and name.strip()
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
            trigger_map=sentence_trigger_cache[sentence_id],
            claim_tuple_index=claim_tuple_index,
        )
        labeled_candidates.append(labeled_candidate)

        predicate = str(labeled_candidate.get("predicate", "")).strip().upper()
        weak_label = str(labeled_candidate.get("weak_label", "NA")).strip()
        predicate_counts[predicate] += 1
        weak_label_counts[weak_label] += 1
        weak_label_reason_counts[str(labeled_candidate.get("weak_label_reason", ""))] += 1
        supervision_tier_counts[str(labeled_candidate.get("supervision_tier", ""))] += 1
        pair_source_counts[str(labeled_candidate.get("pair_source", ""))] += 1
        if labeled_candidate.get("uses_claim_guided_alias_bridge"):
            bridge_usage_counts[predicate] += 1
        if labeled_candidate.get("exact_claim_match"):
            exact_claim_counts[predicate] += 1
        if labeled_candidate.get("trigger_hit_count", 0):
            trigger_positive_counts[predicate] += 1
        if weak_label == "ds_strict":
            predicate_strict_counts[predicate] += 1
        elif weak_label == "ds_weak":
            predicate_weak_counts[predicate] += 1
        else:
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
        "predicate_na_counts": _counter_to_sorted_dict(predicate_na_counts),
        "pair_source_counts": _counter_to_sorted_dict(pair_source_counts),
        "bridge_usage_counts": _counter_to_sorted_dict(bridge_usage_counts),
        "exact_claim_counts": _counter_to_sorted_dict(exact_claim_counts),
        "trigger_positive_counts": _counter_to_sorted_dict(trigger_positive_counts),
        "prepare_summary": prepared_bundle.summary,
    }
    return labeled_candidates, summary


def weak_label_relation_candidates_from_paths(
    *,
    pair_candidates_path: str,
    entities_csv_path: str,
    claims_csv_path: str,
    ontology_path: str,
    relation_names: Sequence[str] | None = None,
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
        relation_names=relation_names,
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
    relation_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = weak_label_relation_candidates_from_paths(
        pair_candidates_path=str(pair_candidates_path),
        entities_csv_path=str(entities_csv_path),
        claims_csv_path=str(claims_csv_path),
        ontology_path=str(ontology_path),
        relation_names=relation_names,
    )
    resolved_output_path = Path(output_path)
    labeled_candidates = list(payload["weak_labeled_candidates"])
    write_jsonl(resolved_output_path, labeled_candidates)
    write_json(resolved_output_path.with_suffix(".summary.json"), payload["summary"])
    return {
        "weak_labeled_count": len(labeled_candidates),
        "output_path": resolved_output_path.as_posix(),
        "summary_path": resolved_output_path.with_suffix(".summary.json").as_posix(),
        "summary": payload["summary"],
    }
