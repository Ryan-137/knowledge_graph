from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_json, read_jsonl, write_json
from kg_core.taxonomy import normalize_entity_type

from .pattern_rules import canonicalize_predicate, load_relation_patterns, match_pattern_signals
from .schema import Evidence, FactCandidate, FactSignal
from .writer import write_fact_jsonl

V1_RELATIONS = {
    "BORN_IN",
    "DIED_IN",
    "STUDIED_AT",
    "WORKED_AT",
    "WORKED_WITH",
    "AUTHORED",
    "PROPOSED",
    "DESIGNED",
    "AWARDED",
    "INFLUENCED",
    "LOCATED_IN",
}


def _relation_constraints(ontology: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    constraints: dict[str, dict[str, set[str]]] = {}
    for relation in ontology.get("relations", []):
        predicate = canonicalize_predicate(str(relation.get("name", "")))
        if predicate not in V1_RELATIONS:
            continue
        domain = relation.get("domain", "Entity")
        range_value = relation.get("range", "Entity")
        domain_values = domain if isinstance(domain, list) else [domain]
        range_values = range_value if isinstance(range_value, list) else [range_value]
        constraints[predicate] = {
            "domain": {normalize_entity_type(str(value)) for value in domain_values},
            "range": {normalize_entity_type(str(value)) for value in range_values},
        }
    return constraints


def _matches_type_constraint(candidate: dict[str, Any], constraints: dict[str, dict[str, set[str]]]) -> bool:
    predicate = canonicalize_predicate(str(candidate.get("predicate", "")))
    constraint = constraints.get(predicate)
    if constraint is None:
        return False
    subject_type = normalize_entity_type(candidate.get("subject_entity_type"))
    object_type = normalize_entity_type(candidate.get("object_entity_type"))
    return subject_type in constraint["domain"] and object_type in constraint["range"]


def _span_from_candidate(candidate: dict[str, Any], prefix: str) -> list[int | None]:
    span = list(candidate.get(f"{prefix}_token_span") or [])
    if len(span) == 2:
        return [span[0], span[1]]
    return [candidate.get(f"{prefix}_token_start"), candidate.get(f"{prefix}_token_end")]


def _char_span_from_tokens(token_spans: list[Any], token_span: list[int | None]) -> list[int | None]:
    if len(token_span) != 2 or token_span[0] is None or token_span[1] is None:
        return []
    start, end = int(token_span[0]), int(token_span[1])
    if start < 0 or end <= start or end > len(token_spans):
        return []
    return [int(token_spans[start][0]), int(token_spans[end - 1][1])]


def _sentence_time_qualifiers(sentence: dict[str, Any]) -> dict[str, Any]:
    normalized_time = list(sentence.get("normalized_time") or [])
    time_mentions = list(sentence.get("time_mentions") or [])
    qualifiers: dict[str, Any] = {}
    if normalized_time:
        qualifiers["sentence_times"] = normalized_time
    if time_mentions:
        qualifiers["time_mentions"] = time_mentions
    return qualifiers


def _type_from_constraint(
    explicit_type: Any,
    predicate: str,
    constraints: dict[str, dict[str, set[str]]],
    side: str,
) -> str:
    """extracted_claims 当前不直接输出实体类型时，用本体约束补齐事实候选契约。"""

    if str(explicit_type or "").strip():
        return normalize_entity_type(explicit_type)
    constraint_values = sorted(constraints.get(predicate, {}).get(side, set()))
    return constraint_values[0] if len(constraint_values) == 1 else "CONCEPT"


def _span_from_extracted_claim(claim: dict[str, Any], key: str) -> list[int | None]:
    span = list(claim.get(key) or [])
    if len(span) == 2:
        return [span[0], span[1]]
    return []


def _primary_supporting_evidence(claim: dict[str, Any]) -> dict[str, Any]:
    supporting_evidence = list(claim.get("supporting_evidence") or [])
    return dict(supporting_evidence[0]) if supporting_evidence else {}


def _require_extracted_claim_field(claim: dict[str, Any], field_name: str, row_number: int) -> str:
    value = str(claim.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"extracted_claims 第 {row_number} 行缺少必填字段：{field_name}")
    return value


def generate_fact_candidates_from_extracted_claims(
    extracted_claims: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    ontology: dict[str, Any],
    *,
    relation_patterns: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """将 relation_extraction/predict.py 的 extracted_claims 契约转换为事实候选契约。"""

    constraints = _relation_constraints(ontology)
    sentence_by_id = {str(sentence.get("sentence_id")): sentence for sentence in sentences}
    generated: list[dict[str, Any]] = []
    filtered_relation_out_of_scope = 0
    pattern_hit_count = 0

    for running_index, claim in enumerate(extracted_claims, start=1):
        predicate = canonicalize_predicate(_require_extracted_claim_field(claim, "predicate", running_index))
        if predicate not in V1_RELATIONS:
            filtered_relation_out_of_scope += 1
            continue
        subject_id = _require_extracted_claim_field(claim, "subject_id", running_index)
        object_id = _require_extracted_claim_field(claim, "object_id", running_index)
        evidence_sentence_id = _require_extracted_claim_field(claim, "evidence_sentence_id", running_index)
        primary_evidence = _primary_supporting_evidence(claim)
        sentence = sentence_by_id.get(evidence_sentence_id, {})
        subject_token_span = _span_from_extracted_claim(claim, "subject_span")
        object_token_span = _span_from_extracted_claim(claim, "object_span")
        evidence_text = str(claim.get("evidence_text") or primary_evidence.get("text") or sentence.get("text") or "")
        evidence = Evidence(
            doc_id=str(claim.get("doc_id") or primary_evidence.get("doc_id") or sentence.get("doc_id") or ""),
            sentence_id=evidence_sentence_id,
            source_id=str(claim.get("source_id") or primary_evidence.get("source_id") or sentence.get("source_id") or ""),
            text=evidence_text,
            subject_mention_id=str(claim.get("subject_mention_id") or primary_evidence.get("subject_mention_id") or ""),
            object_mention_id=str(claim.get("object_mention_id") or primary_evidence.get("object_mention_id") or ""),
            subject_text=str(claim.get("subject_text") or primary_evidence.get("subject_text") or ""),
            object_text=str(claim.get("object_text") or primary_evidence.get("object_text") or ""),
            subject_token_span=subject_token_span,
            object_token_span=object_token_span,
        )
        candidate_for_patterns = {
            "predicate": predicate,
            "text": evidence_text,
            "subject_token_span": subject_token_span,
            "object_token_span": object_token_span,
        }
        pattern_result = match_pattern_signals(candidate_for_patterns, relation_patterns=relation_patterns)
        probability = float(claim.get("probability", claim.get("confidence", 0.0)) or 0.0)
        signals = [
            FactSignal(
                "relation_model",
                probability,
                "PREDICTED",
                {
                    "threshold_used": claim.get("threshold_used"),
                    "claim_candidate_id": claim.get("claim_candidate_id"),
                    "bag_id": claim.get("bag_id"),
                    "evidence_candidate_id": claim.get("evidence_candidate_id"),
                    "supporting_candidate_ids": list(claim.get("supporting_candidate_ids") or []),
                    "supporting_sentence_ids": list(claim.get("supporting_sentence_ids") or []),
                    "pair_source": claim.get("pair_source", ""),
                    "is_from_bridge": bool(claim.get("is_from_bridge", False)),
                    "matched_claim_ids": list(claim.get("matched_claim_ids") or []),
                    "extractor": claim.get("extractor", ""),
                },
            )
        ]
        if pattern_result["matched"]:
            pattern_hit_count += 1
            signals.append(
                FactSignal(
                    "pattern_match",
                    float(pattern_result["score"]),
                    "MATCHED",
                    {
                        "hits": pattern_result["hits"],
                        "token_distance": pattern_result["token_distance"],
                    },
                )
            )

        fact_candidate = FactCandidate(
            fact_candidate_id=f"factcand_{running_index:06d}",
            source_candidate_id=str(claim.get("claim_candidate_id") or claim.get("evidence_candidate_id") or f"relclaim_{running_index:06d}"),
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            subject_type=_type_from_constraint(claim.get("subject_type"), predicate, constraints, "domain"),
            object_type=_type_from_constraint(claim.get("object_type"), predicate, constraints, "range"),
            subject_text=evidence.subject_text,
            object_text=evidence.object_text,
            qualifiers=_sentence_time_qualifiers(sentence),
            evidence=evidence,
            extractor="relation_model_extracted_claims",
            signals=signals,
            confidence=probability,
        )
        generated.append(fact_candidate.to_dict())

    summary = {
        "input_extracted_claim_count": len(extracted_claims),
        "fact_candidate_count": len(generated),
        "pattern_hit_count": pattern_hit_count,
        "filtered_relation_out_of_scope_count": filtered_relation_out_of_scope,
        "relation_counts": _count_by_key(generated, "predicate"),
    }
    return generated, summary


def _link_quality_signal(candidate: dict[str, Any]) -> FactSignal:
    resolutions = {
        str(candidate.get("subject_resolution") or ""),
        str(candidate.get("object_resolution") or ""),
    }
    if resolutions <= {"linked", "linked_by_coref"}:
        return FactSignal("link_quality", 0.0, "HIGH", {"resolutions": sorted(resolutions)})
    if "claim_guided_alias_bridge" in resolutions:
        return FactSignal("link_quality", 0.0, "BRIDGE_SUPPORTED", {"resolutions": sorted(resolutions)})
    return FactSignal("link_quality", -0.15, "LOW", {"resolutions": sorted(resolutions)})


def generate_fact_candidates(
    pair_candidates: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    ontology: dict[str, Any],
    *,
    relation_patterns: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    constraints = _relation_constraints(ontology)
    sentence_by_id = {str(sentence.get("sentence_id")): sentence for sentence in sentences}
    generated: list[dict[str, Any]] = []
    filtered_type_mismatch = 0
    pattern_hit_count = 0

    for running_index, candidate in enumerate(pair_candidates, start=1):
        predicate = canonicalize_predicate(str(candidate.get("predicate", "")))
        if predicate not in V1_RELATIONS:
            continue
        normalized_candidate = dict(candidate)
        normalized_candidate["predicate"] = predicate
        if not _matches_type_constraint(normalized_candidate, constraints):
            filtered_type_mismatch += 1
            continue

        sentence = sentence_by_id.get(str(candidate.get("sentence_id")), {})
        token_spans = list(candidate.get("token_spans") or sentence.get("token_spans") or [])
        subject_token_span = _span_from_candidate(candidate, "subject")
        object_token_span = _span_from_candidate(candidate, "object")
        evidence = Evidence(
            doc_id=str(candidate.get("doc_id") or sentence.get("doc_id") or ""),
            sentence_id=str(candidate.get("sentence_id") or ""),
            source_id=str(candidate.get("source_id") or sentence.get("source_id") or ""),
            text=str(candidate.get("text") or sentence.get("text") or ""),
            subject_mention_id=str(candidate.get("subject_mention_id") or ""),
            object_mention_id=str(candidate.get("object_mention_id") or ""),
            subject_text=str(candidate.get("subject_text") or ""),
            object_text=str(candidate.get("object_text") or ""),
            subject_token_span=subject_token_span,
            object_token_span=object_token_span,
            subject_char_span=_char_span_from_tokens(token_spans, subject_token_span),
            object_char_span=_char_span_from_tokens(token_spans, object_token_span),
        )
        pattern_result = match_pattern_signals(
            normalized_candidate,
            relation_patterns=relation_patterns,
        )
        signals = [_link_quality_signal(normalized_candidate)]
        if pattern_result["matched"]:
            pattern_hit_count += 1
            signals.append(
                FactSignal(
                    "pattern_match",
                    float(pattern_result["score"]),
                    "MATCHED",
                    {
                        "hits": pattern_result["hits"],
                        "token_distance": pattern_result["token_distance"],
                    },
                )
            )

        fact_candidate = FactCandidate(
            fact_candidate_id=f"factcand_{running_index:06d}",
            source_candidate_id=str(candidate.get("candidate_id") or f"relcand_{running_index:06d}"),
            subject_id=str(candidate.get("subject_entity_id") or ""),
            predicate=predicate,
            object_id=str(candidate.get("object_entity_id") or ""),
            subject_type=normalize_entity_type(candidate.get("subject_entity_type")),
            object_type=normalize_entity_type(candidate.get("object_entity_type")),
            subject_text=str(candidate.get("subject_text") or ""),
            object_text=str(candidate.get("object_text") or ""),
            qualifiers=_sentence_time_qualifiers(sentence),
            evidence=evidence,
            extractor="fact_candidate_generator",
            signals=signals,
        )
        generated.append(fact_candidate.to_dict())

    summary = {
        "input_pair_candidate_count": len(pair_candidates),
        "fact_candidate_count": len(generated),
        "pattern_hit_count": pattern_hit_count,
        "filtered_type_mismatch_count": filtered_type_mismatch,
        "relation_counts": _count_by_key(generated, "predicate"),
    }
    return generated, summary


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def generate_fact_candidates_from_paths(
    *,
    pair_candidates_path: str | Path,
    sentences_path: str | Path,
    ontology_path: str | Path,
    relation_patterns_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    candidates, summary = generate_fact_candidates(
        read_jsonl(pair_candidates_path),
        read_jsonl(sentences_path),
        read_json(ontology_path),
        relation_patterns=load_relation_patterns(relation_patterns_path),
    )
    write_fact_jsonl(output_path, candidates)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {
        "output_path": Path(output_path).as_posix(),
        "summary_path": Path(output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }


def generate_fact_candidates_from_extracted_claims_paths(
    *,
    extracted_claims_path: str | Path,
    sentences_path: str | Path,
    ontology_path: str | Path,
    relation_patterns_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    candidates, summary = generate_fact_candidates_from_extracted_claims(
        read_jsonl(extracted_claims_path),
        read_jsonl(sentences_path),
        read_json(ontology_path),
        relation_patterns=load_relation_patterns(relation_patterns_path),
    )
    write_fact_jsonl(output_path, candidates)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {
        "output_path": Path(output_path).as_posix(),
        "summary_path": Path(output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }
