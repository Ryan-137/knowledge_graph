from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence

from pathlib import Path

from kg_core import ProjectPaths
from kg_core.entity_catalog import load_entity_catalog, normalize_alias_text
from kg_core.io import read_json, read_jsonl, write_json, write_jsonl
from kg_core.taxonomy import MENTION_TYPE_TO_ENTITY_TYPE, normalize_entity_type, normalize_mention_type
from relation_extraction.rules import (
    DEFAULT_RELATION_NAMES,
    build_relation_rules,
    build_sentence_trigger_map,
    infer_candidate_relations,
)


@dataclass(frozen=True, slots=True)
class PreparedRelationBundle:
    """关系准备阶段的统一返回结果。"""

    sentences: list[dict[str, Any]]
    resolved_mentions: list[dict[str, Any]]
    relation_candidates: list[dict[str, Any]]
    summary: dict[str, Any]
    ontology: dict[str, Any]
    claims: list[dict[str, Any]]


RESOLVED_LINK_DECISIONS = {"LINKED", "LINKED_BY_COREF"}
RESOLVED_MENTION_STATES = {"linked", "linked_by_coref"}


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _build_sentence_index(
    *,
    sentences: Sequence[dict[str, Any]],
    tokenized_sentences: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    sentence_index: dict[str, dict[str, Any]] = {}
    for record in sentences:
        sentence_id = str(record.get("sentence_id", "")).strip()
        if not sentence_id:
            continue
        sentence_index[sentence_id] = dict(record)
    for record in tokenized_sentences:
        sentence_id = str(record.get("sentence_id", "")).strip()
        if not sentence_id:
            continue
        merged = dict(sentence_index.get(sentence_id, {}))
        merged.update(record)
        sentence_index[sentence_id] = merged
    return sentence_index


def _extract_mention_text(
    *,
    sentence_text: str,
    token_spans: Sequence[Sequence[int]],
    token_start: int | None,
    token_end: int | None,
    char_start: int | None,
    char_end: int | None,
) -> tuple[str, int | None, int | None]:
    if (
        token_start is not None
        and token_end is not None
        and token_start >= 0
        and token_end > token_start
        and token_end <= len(token_spans)
    ):
        derived_char_start = int(token_spans[token_start][0])
        derived_char_end = int(token_spans[token_end - 1][1])
        return sentence_text[derived_char_start:derived_char_end], derived_char_start, derived_char_end
    if char_start is not None and char_end is not None and 0 <= char_start < char_end <= len(sentence_text):
        return sentence_text[char_start:char_end], char_start, char_end
    return "", char_start, char_end


def _build_entity_index_with_alias_surfaces(
    entity_catalog: Any,
    claims: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    entity_index = {entity_id: dict(row) for entity_id, row in entity_catalog.entities.items()}
    alias_surface_index: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_surface(entity_id: str, surface: str, surface_source: str) -> None:
        normalized_surface = normalize_alias_text(surface)
        if not entity_id or not normalized_surface:
            return
        alias_surface_index[entity_id].append(
            {
                "surface": surface,
                "normalized_surface": normalized_surface,
                "source": surface_source,
            }
        )

    for entity_id, entity_row in entity_index.items():
        for key in ("canonical_name", "label_en", "label_zh", "wikipedia_title_en"):
            add_surface(entity_id, str(entity_row.get(key, "")), key)

    for alias_row in entity_catalog.aliases:
        entity_id = str(alias_row.get("entity_id", "")).strip()
        add_surface(entity_id, str(alias_row.get("alias", "")), "alias")

    for claim in claims:
        entity_id = str(claim.get("object_id", "")).strip()
        add_surface(entity_id, str(claim.get("object_text", "")), "claim_object_text")
    return entity_index, alias_surface_index


def _repair_resolved_mentions(
    *,
    resolved_mentions: Sequence[dict[str, Any]],
    sentence_index: dict[str, dict[str, Any]],
    entity_index: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repaired_mentions: list[dict[str, Any]] = []
    repair_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    linked_entity_type_counts: Counter[str] = Counter()

    for raw_mention in resolved_mentions:
        mention = dict(raw_mention)
        sentence_id = str(mention.get("sentence_id", "")).strip()
        sentence_record = sentence_index.get(sentence_id)
        if sentence_record is None:
            raise ValueError(f"resolved_mentions 中存在未知 sentence_id: {sentence_id}")

        sentence_text = str(sentence_record.get("text", ""))
        token_spans = list(sentence_record.get("token_spans", []))
        tokens = list(sentence_record.get("tokens", []))
        mention["doc_id"] = str(mention.get("doc_id") or sentence_record.get("doc_id", "")).strip()
        if not mention.get("source_id"):
            mention["source_id"] = str(sentence_record.get("source_id", "")).strip()
            repair_counts["source_id_filled_from_sentence"] += 1
        mention["sentence_text"] = sentence_text
        mention["tokens"] = tokens
        mention["token_spans"] = token_spans
        mention["sentence_index_in_doc"] = sentence_record.get("sentence_index_in_doc")
        mention["normalized_time"] = list(sentence_record.get("normalized_time", []))
        mention["time_mentions"] = list(sentence_record.get("time_mentions", []))

        token_start = mention.get("token_start")
        token_end = mention.get("token_end")
        char_start = mention.get("char_start")
        char_end = mention.get("char_end")
        extracted_text, repaired_char_start, repaired_char_end = _extract_mention_text(
            sentence_text=sentence_text,
            token_spans=token_spans,
            token_start=token_start if isinstance(token_start, int) else None,
            token_end=token_end if isinstance(token_end, int) else None,
            char_start=char_start if isinstance(char_start, int) else None,
            char_end=char_end if isinstance(char_end, int) else None,
        )
        mention_text = str(mention.get("mention_text") or "").strip()
        if extracted_text:
            if mention_text != extracted_text:
                repair_counts["mention_text_repaired_from_span"] += 1
            mention_text = extracted_text
        mention["mention_text"] = mention_text
        mention["char_start"] = repaired_char_start
        mention["char_end"] = repaired_char_end

        mention_type = normalize_mention_type(mention.get("mention_type"))
        mention["mention_type"] = mention_type

        decision = str(mention.get("decision", "NIL")).strip().upper() or "NIL"
        mention["decision"] = decision
        decision_counts[decision] += 1
        entity_id = str(mention.get("entity_id") or "").strip()
        if decision == "LINKED" and not entity_id:
            candidate_list = list(mention.get("candidate_list") or [])
            if candidate_list:
                entity_id = str(candidate_list[0].get("entity_id") or "").strip()
                if entity_id:
                    mention["entity_id"] = entity_id
                    repair_counts["linked_entity_id_filled_from_top_candidate"] += 1

        if decision in RESOLVED_LINK_DECISIONS and entity_id:
            entity_row = entity_index.get(entity_id, {})
            linked_entity_type = normalize_entity_type(
                entity_row.get("entity_type")
                or mention.get("linked_entity_type")
                or mention.get("mention_type")
            )
            mention["canonical_name"] = str(
                mention.get("canonical_name")
                or entity_row.get("canonical_name")
                or mention_text
            ).strip()
            mention["linked_entity_type"] = linked_entity_type
            mention["mention_resolution"] = "linked_by_coref" if decision == "LINKED_BY_COREF" else "linked"
            linked_entity_type_counts[linked_entity_type] += 1
        else:
            mention["entity_id"] = entity_id or None
            mention["canonical_name"] = str(mention.get("canonical_name") or "").strip() or None
            mention["linked_entity_type"] = normalize_entity_type(
                mention.get("linked_entity_type") or MENTION_TYPE_TO_ENTITY_TYPE.get(mention_type)
            )
            mention["mention_resolution"] = "nil"

        mention["normalized_mention_text"] = normalize_alias_text(mention_text)
        if not mention.get("context_window"):
            mention["context_window"] = sentence_text
            repair_counts["context_window_filled_from_sentence"] += 1
        repaired_mentions.append(mention)

    summary = {
        "repair_counts": _counter_to_sorted_dict(repair_counts),
        "decision_counts": _counter_to_sorted_dict(decision_counts),
        "linked_entity_type_counts": _counter_to_sorted_dict(linked_entity_type_counts),
    }
    return repaired_mentions, summary


def _build_claim_indexes(
    *,
    claims: Sequence[dict[str, Any]],
    relation_names: Sequence[str],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    claim_tuple_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    claim_adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed_relations = {name.strip().upper() for name in relation_names}
    for claim in claims:
        predicate = str(claim.get("predicate", "")).strip().upper()
        if predicate not in allowed_relations:
            continue
        subject_id = str(claim.get("subject_id", "")).strip()
        object_id = str(claim.get("object_id", "")).strip()
        if not subject_id or not object_id:
            continue
        claim_tuple_index[(subject_id, predicate, object_id)].append(dict(claim))
        claim_adjacency[subject_id].append(dict(claim))
        claim_adjacency[object_id].append(dict(claim))
    return claim_tuple_index, claim_adjacency


def _surface_match_score(mention_surface: str, candidate_surface: str) -> float:
    if not mention_surface or not candidate_surface:
        return 0.0
    if mention_surface == candidate_surface:
        return 1.0
    mention_tokens = set(mention_surface.split())
    candidate_tokens = set(candidate_surface.split())
    if mention_tokens and mention_tokens.issubset(candidate_tokens):
        overlap_ratio = len(mention_tokens) / max(len(candidate_tokens), 1)
        return max(0.9, overlap_ratio)
    return SequenceMatcher(None, mention_surface, candidate_surface).ratio()


def _mention_token_distance(first_mention: dict[str, Any], second_mention: dict[str, Any]) -> int | None:
    first_start = first_mention.get("token_start")
    first_end = first_mention.get("token_end")
    second_start = second_mention.get("token_start")
    second_end = second_mention.get("token_end")
    if any(value is None for value in (first_start, first_end, second_start, second_end)):
        return None
    first_start_int, first_end_int = int(first_start), int(first_end)
    second_start_int, second_end_int = int(second_start), int(second_end)
    if first_end_int <= second_start_int:
        return second_start_int - first_end_int
    if second_end_int <= first_start_int:
        return first_start_int - second_end_int
    return 0


def _bridge_claim_ids(*mentions: dict[str, Any]) -> set[str]:
    claim_ids: set[str] = set()
    for mention in mentions:
        if mention.get("mention_resolution") != "claim_guided_alias_bridge":
            continue
        claim_id = str(mention.get("bridge_claim_id") or "").strip()
        if claim_id:
            claim_ids.add(claim_id)
    return claim_ids


def _build_claim_guided_alias_bridges(
    *,
    repaired_mentions: Sequence[dict[str, Any]],
    entity_index: dict[str, dict[str, Any]],
    alias_surface_index: dict[str, list[dict[str, Any]]],
    claim_adjacency: dict[str, list[dict[str, Any]]],
    relation_names: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped_mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in repaired_mentions:
        grouped_mentions[str(mention.get("sentence_id", ""))].append(mention)

    allowed_relations = {name.strip().upper() for name in relation_names}
    bridge_records: list[dict[str, Any]] = []
    bridge_counts_by_predicate: Counter[str] = Counter()
    bridge_counts_by_match_source: Counter[str] = Counter()
    seen_bridge_keys: set[tuple[str, str, str]] = set()

    for sentence_mentions in grouped_mentions.values():
        linked_mentions = [mention for mention in sentence_mentions if mention.get("mention_resolution") in RESOLVED_MENTION_STATES]
        nil_mentions = [mention for mention in sentence_mentions if mention.get("mention_resolution") == "nil"]
        if not linked_mentions or not nil_mentions:
            continue

        for nil_mention in nil_mentions:
            nil_surface = str(nil_mention.get("normalized_mention_text", "")).strip()
            if not nil_surface:
                continue
            nil_expected_type = normalize_entity_type(MENTION_TYPE_TO_ENTITY_TYPE.get(str(nil_mention.get("mention_type", ""))))
            best_bridge_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}

            for anchor_mention in linked_mentions:
                anchor_entity_id = str(anchor_mention.get("entity_id") or "").strip()
                if not anchor_entity_id:
                    continue
                for claim in claim_adjacency.get(anchor_entity_id, []):
                    predicate = str(claim.get("predicate", "")).strip().upper()
                    if predicate not in allowed_relations:
                        continue
                    subject_id = str(claim.get("subject_id", "")).strip()
                    object_id = str(claim.get("object_id", "")).strip()
                    if not subject_id or not object_id:
                        continue
                    if anchor_entity_id == subject_id:
                        bridge_entity_id = object_id
                        bridge_direction = "object"
                    elif anchor_entity_id == object_id:
                        bridge_entity_id = subject_id
                        bridge_direction = "subject"
                    else:
                        continue
                    bridge_entity_row = entity_index.get(bridge_entity_id, {})
                    bridge_entity_type = normalize_entity_type(bridge_entity_row.get("entity_type"))
                    if nil_expected_type and bridge_entity_type != nil_expected_type:
                        continue

                    best_surface: dict[str, Any] | None = None
                    for surface_info in alias_surface_index.get(bridge_entity_id, []):
                        score = _surface_match_score(
                            nil_surface,
                            str(surface_info.get("normalized_surface", "")),
                        )
                        if score < 0.88:
                            continue
                        candidate_surface = dict(surface_info)
                        candidate_surface["score"] = round(score, 6)
                        if best_surface is None or candidate_surface["score"] > best_surface["score"]:
                            best_surface = candidate_surface
                    if best_surface is None:
                        continue

                    bridge_key = (str(nil_mention.get("mention_id")), bridge_entity_id, predicate)
                    bridge_record = {
                        **nil_mention,
                        "entity_id": bridge_entity_id,
                        "canonical_name": str(bridge_entity_row.get("canonical_name") or nil_mention.get("mention_text") or "").strip(),
                        "linked_entity_type": bridge_entity_type,
                        "decision": "BRIDGED",
                        "mention_resolution": "claim_guided_alias_bridge",
                        "bridge_anchor_mention_id": anchor_mention.get("mention_id"),
                        "bridge_anchor_entity_id": anchor_entity_id,
                        "bridge_predicate": predicate,
                        "bridge_claim_id": str(claim.get("claim_id", "")).strip(),
                        "bridge_direction": bridge_direction,
                        "bridge_match_surface": best_surface.get("surface"),
                        "bridge_match_source": best_surface.get("source"),
                        "bridge_match_score": best_surface.get("score"),
                    }
                    current_best = best_bridge_by_key.get(bridge_key)
                    if current_best is None or float(bridge_record["bridge_match_score"]) > float(
                        current_best["bridge_match_score"]
                    ):
                        best_bridge_by_key[bridge_key] = bridge_record

            for bridge_key, bridge_record in best_bridge_by_key.items():
                if bridge_key in seen_bridge_keys:
                    continue
                seen_bridge_keys.add(bridge_key)
                bridge_records.append(bridge_record)
                bridge_counts_by_predicate[str(bridge_record.get("bridge_predicate", ""))] += 1
                bridge_counts_by_match_source[str(bridge_record.get("bridge_match_source", ""))] += 1

    summary = {
        "bridge_count": len(bridge_records),
        "bridge_counts_by_predicate": _counter_to_sorted_dict(bridge_counts_by_predicate),
        "bridge_counts_by_match_source": _counter_to_sorted_dict(bridge_counts_by_match_source),
    }
    return bridge_records, summary


def _build_relation_candidates(
    *,
    sentence_index: dict[str, dict[str, Any]],
    repaired_mentions: Sequence[dict[str, Any]],
    bridged_mentions: Sequence[dict[str, Any]],
    relation_rules: dict[str, Any],
    claim_tuple_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    max_token_distance: int = 24,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mentions_by_sentence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in repaired_mentions:
        if mention.get("mention_resolution") in RESOLVED_MENTION_STATES:
            mentions_by_sentence[str(mention.get("sentence_id", ""))].append(mention)
    for mention in bridged_mentions:
        mentions_by_sentence[str(mention.get("sentence_id", ""))].append(mention)

    candidate_counts_by_relation: Counter[str] = Counter()
    candidate_counts_by_source: Counter[str] = Counter()
    exact_claim_counts: Counter[str] = Counter()
    filtered_counts: Counter[str] = Counter()
    relation_candidates: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str, str, str, str, str]] = set()
    seen_token_distance_filter_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    candidate_index = 0

    for sentence_id, sentence_mentions in mentions_by_sentence.items():
        sentence_record = sentence_index[sentence_id]
        trigger_map = build_sentence_trigger_map(
            tokens=list(sentence_record.get("tokens", [])),
            relation_rules=relation_rules,
        )
        triggered_relation_names = set(trigger_map)
        for subject_mention in sentence_mentions:
            for object_mention in sentence_mentions:
                if subject_mention.get("mention_id") == object_mention.get("mention_id"):
                    continue
                subject_entity_id = str(subject_mention.get("entity_id") or "").strip()
                object_entity_id = str(object_mention.get("entity_id") or "").strip()
                if not subject_entity_id or not object_entity_id or subject_entity_id == object_entity_id:
                    continue
                subject_type = normalize_entity_type(subject_mention.get("linked_entity_type"))
                object_type = normalize_entity_type(object_mention.get("linked_entity_type"))
                type_compatible_relations = infer_candidate_relations(
                    subject_type=subject_type,
                    object_type=object_type,
                    relation_rules=relation_rules,
                )
                if not type_compatible_relations:
                    continue
                token_distance = _mention_token_distance(subject_mention, object_mention)
                if token_distance is not None and token_distance > max_token_distance:
                    filter_key = (
                        sentence_id,
                        "||".join(
                            sorted(
                                [
                                    str(subject_mention.get("mention_id")),
                                    str(object_mention.get("mention_id")),
                                ]
                            )
                        ),
                        tuple(type_compatible_relations),
                    )
                    if filter_key not in seen_token_distance_filter_keys:
                        filtered_counts["token_distance"] += 1
                        seen_token_distance_filter_keys.add(filter_key)
                    continue

                bridge_claim_ids = _bridge_claim_ids(subject_mention, object_mention)
                exact_claim_rows_by_relation: dict[str, list[dict[str, Any]]] = {}
                excluded_bridge_claim_ids: set[str] = set()
                for relation_name in type_compatible_relations:
                    exact_claim_key = (subject_entity_id, relation_name, object_entity_id)
                    raw_exact_claim_rows = claim_tuple_index.get(exact_claim_key, [])
                    filtered_exact_claim_rows = []
                    for row in raw_exact_claim_rows:
                        claim_id = str(row.get("claim_id") or "").strip()
                        if claim_id and claim_id in bridge_claim_ids:
                            excluded_bridge_claim_ids.add(claim_id)
                            continue
                        filtered_exact_claim_rows.append(row)
                    if filtered_exact_claim_rows:
                        exact_claim_rows_by_relation[relation_name] = filtered_exact_claim_rows

                # 关系候选必须来自句内触发词或非 bridge 来源的精确 claim，避免纯类型枚举污染候选空间。
                candidate_relations = sorted(
                    {
                        relation_name
                        for relation_name in type_compatible_relations
                        if relation_name in triggered_relation_names or relation_name in exact_claim_rows_by_relation
                    }
                )
                if not candidate_relations:
                    filtered_counts["no_trigger_or_non_bridge_claim"] += 1
                    continue
                for relation_name in candidate_relations:
                    candidate_key = (
                        sentence_id,
                        str(subject_mention.get("mention_id")),
                        str(object_mention.get("mention_id")),
                        subject_entity_id,
                        object_entity_id,
                        relation_name,
                    )
                    if candidate_key in seen_candidate_keys:
                        continue
                    seen_candidate_keys.add(candidate_key)
                    candidate_index += 1
                    exact_claim_rows = exact_claim_rows_by_relation.get(relation_name, [])
                    resolutions = {str(subject_mention.get("mention_resolution")), str(object_mention.get("mention_resolution"))}
                    if resolutions == {"linked"}:
                        pair_source = "linked_linked"
                    elif "claim_guided_alias_bridge" in resolutions:
                        pair_source = "bridge_augmented"
                    else:
                        pair_source = "coref_augmented"
                    candidate_record = {
                        "candidate_id": f"relcand_{candidate_index:06d}",
                        "sentence_id": sentence_id,
                        "doc_id": sentence_record.get("doc_id"),
                        "source_id": sentence_record.get("source_id"),
                        "sentence_index_in_doc": sentence_record.get("sentence_index_in_doc"),
                        "text": sentence_record.get("text"),
                        "tokens": list(sentence_record.get("tokens", [])),
                        "token_spans": list(sentence_record.get("token_spans", [])),
                        "predicate": relation_name,
                        "subject_mention_id": subject_mention.get("mention_id"),
                        "subject_text": subject_mention.get("mention_text"),
                        "subject_entity_id": subject_entity_id,
                        "subject_canonical_name": subject_mention.get("canonical_name"),
                        "subject_entity_type": subject_type,
                        "subject_token_start": subject_mention.get("token_start"),
                        "subject_token_end": subject_mention.get("token_end"),
                        "subject_token_span": [
                            subject_mention.get("token_start"),
                            subject_mention.get("token_end"),
                        ],
                        "subject_resolution": subject_mention.get("mention_resolution"),
                        "object_mention_id": object_mention.get("mention_id"),
                        "object_text": object_mention.get("mention_text"),
                        "object_entity_id": object_entity_id,
                        "object_canonical_name": object_mention.get("canonical_name"),
                        "object_entity_type": object_type,
                        "object_token_start": object_mention.get("token_start"),
                        "object_token_end": object_mention.get("token_end"),
                        "object_token_span": [
                            object_mention.get("token_start"),
                            object_mention.get("token_end"),
                        ],
                        "object_resolution": object_mention.get("mention_resolution"),
                        "pair_source": pair_source,
                        "token_distance": token_distance,
                        "exact_claim_match": bool(exact_claim_rows),
                        "matched_claim_ids": [row.get("claim_id") for row in exact_claim_rows],
                        "matched_claim_count": len(exact_claim_rows),
                        "bridge_source_claim_ids": sorted(bridge_claim_ids),
                        "excluded_bridge_claim_ids": sorted(excluded_bridge_claim_ids),
                        "bridge_predicates": sorted(
                            {
                                predicate
                                for predicate in (
                                    str(subject_mention.get("bridge_predicate") or "").strip().upper(),
                                    str(object_mention.get("bridge_predicate") or "").strip().upper(),
                                )
                                if predicate
                            }
                        ),
                        "bridge_details": {
                            "subject": {
                                "bridge_claim_id": subject_mention.get("bridge_claim_id"),
                                "bridge_anchor_entity_id": subject_mention.get("bridge_anchor_entity_id"),
                                "bridge_predicate": subject_mention.get("bridge_predicate"),
                                "bridge_match_surface": subject_mention.get("bridge_match_surface"),
                                "bridge_match_score": subject_mention.get("bridge_match_score"),
                            },
                            "object": {
                                "bridge_claim_id": object_mention.get("bridge_claim_id"),
                                "bridge_anchor_entity_id": object_mention.get("bridge_anchor_entity_id"),
                                "bridge_predicate": object_mention.get("bridge_predicate"),
                                "bridge_match_surface": object_mention.get("bridge_match_surface"),
                                "bridge_match_score": object_mention.get("bridge_match_score"),
                            },
                        },
                    }
                    relation_candidates.append(candidate_record)
                    candidate_counts_by_relation[relation_name] += 1
                    candidate_counts_by_source[pair_source] += 1
                    if exact_claim_rows:
                        exact_claim_counts[relation_name] += 1

    summary = {
        "candidate_count": len(relation_candidates),
        "candidate_counts_by_relation": _counter_to_sorted_dict(candidate_counts_by_relation),
        "candidate_counts_by_source": _counter_to_sorted_dict(candidate_counts_by_source),
        "exact_claim_counts_by_relation": _counter_to_sorted_dict(exact_claim_counts),
        "filtered_counts": _counter_to_sorted_dict(filtered_counts),
    }
    return relation_candidates, summary


def prepare_relation_candidates_from_paths(
    *,
    sentences_path: str,
    tokenized_sentences_path: str,
    resolved_mentions_path: str,
    entities_csv_path: str,
    aliases_csv_path: str,
    claims_csv_path: str,
    ontology_path: str,
    relation_names: Sequence[str] | None = None,
    max_token_distance: int = 24,
) -> PreparedRelationBundle:
    """读取全量上游资源，生成关系抽取的同句候选与 bridge 扩充结果。"""

    normalized_relation_names = tuple(
        name.strip().upper() for name in (relation_names or DEFAULT_RELATION_NAMES) if name and name.strip()
    )
    sentences = read_jsonl(sentences_path)
    tokenized_sentences = read_jsonl(tokenized_sentences_path)
    resolved_mentions = read_jsonl(resolved_mentions_path)
    ontology = read_json(ontology_path)
    entity_catalog = load_entity_catalog(
        entities_csv_path=entities_csv_path,
        aliases_csv_path=aliases_csv_path,
        claims_csv_path=claims_csv_path,
    )

    sentence_index = _build_sentence_index(sentences=sentences, tokenized_sentences=tokenized_sentences)
    entity_index, alias_surface_index = _build_entity_index_with_alias_surfaces(entity_catalog, entity_catalog.claims)
    repaired_mentions, repair_summary = _repair_resolved_mentions(
        resolved_mentions=resolved_mentions,
        sentence_index=sentence_index,
        entity_index=entity_index,
    )
    claim_tuple_index, claim_adjacency = _build_claim_indexes(
        claims=entity_catalog.claims,
        relation_names=normalized_relation_names,
    )
    relation_rules = build_relation_rules(
        ontology=ontology,
        claims=entity_catalog.claims,
        entity_index=entity_index,
        relation_names=normalized_relation_names,
    )
    bridged_mentions, bridge_summary = _build_claim_guided_alias_bridges(
        repaired_mentions=repaired_mentions,
        entity_index=entity_index,
        alias_surface_index=alias_surface_index,
        claim_adjacency=claim_adjacency,
        relation_names=normalized_relation_names,
    )
    relation_candidates, candidate_summary = _build_relation_candidates(
        sentence_index=sentence_index,
        repaired_mentions=repaired_mentions,
        bridged_mentions=bridged_mentions,
        relation_rules=relation_rules,
        claim_tuple_index=claim_tuple_index,
        max_token_distance=max_token_distance,
    )

    summary = {
        "sentence_count": len(sentence_index),
        "tokenized_sentence_count": len(tokenized_sentences),
        "resolved_mention_count": len(repaired_mentions),
        "bridge_mention_count": len(bridged_mentions),
        "relation_names": list(normalized_relation_names),
        "linked_mention_repair": repair_summary,
        "alias_bridge": bridge_summary,
        "relation_candidates": candidate_summary,
    }
    all_mentions = list(repaired_mentions) + list(bridged_mentions)
    prepared_sentences = [sentence_index[sentence_id] for sentence_id in sorted(sentence_index)]
    return PreparedRelationBundle(
        sentences=prepared_sentences,
        resolved_mentions=all_mentions,
        relation_candidates=relation_candidates,
        summary=summary,
        ontology=ontology,
        claims=list(entity_catalog.claims),
    )


def prepare_relation_pairs(
    *,
    resolved_mentions_path: Path,
    sentences_path: Path,
    ontology_path: Path,
    output_path: Path,
    max_token_distance: int = 24,
    include_nil: bool = False,
    tokenized_sentences_path: Path | None = None,
    entities_csv_path: Path | None = None,
    aliases_csv_path: Path | None = None,
    claims_csv_path: Path | None = None,
) -> dict[str, Any]:
    """CLI 入口：基于 linked mentions 生成关系候选并落盘。

    当前关系候选生成逻辑已经内置 span/句子级过滤；`max_token_distance`
    在候选生成阶段生效，`include_nil` 仍由上游 mention resolution 契约控制。
    """

    _ = include_nil
    project_paths = ProjectPaths.discover(start=output_path)
    resolved_tokenized_path = tokenized_sentences_path or (project_paths.mentions_dir / "tokenized_sentences.jsonl")
    resolved_entities_csv = entities_csv_path or (project_paths.structured_csv_dir / "entities.csv")
    resolved_aliases_csv = aliases_csv_path or (project_paths.structured_csv_dir / "aliases.csv")
    resolved_claims_csv = claims_csv_path or (project_paths.structured_csv_dir / "claims.csv")

    prepared_bundle = prepare_relation_candidates_from_paths(
        sentences_path=str(sentences_path),
        tokenized_sentences_path=str(resolved_tokenized_path),
        resolved_mentions_path=str(resolved_mentions_path),
        entities_csv_path=str(resolved_entities_csv),
        aliases_csv_path=str(resolved_aliases_csv),
        claims_csv_path=str(resolved_claims_csv),
        ontology_path=str(ontology_path),
        max_token_distance=max_token_distance,
    )
    write_jsonl(output_path, prepared_bundle.relation_candidates)
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, prepared_bundle.summary)
    return {
        "output": output_path.as_posix(),
        "summary_output": summary_path.as_posix(),
        "candidate_count": len(prepared_bundle.relation_candidates),
        "bridge_mention_count": prepared_bundle.summary.get("bridge_mention_count", 0),
        "summary": prepared_bundle.summary,
    }
