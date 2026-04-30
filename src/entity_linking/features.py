from __future__ import annotations

from typing import Any

from kg_core.taxonomy import normalize_entity_type, normalize_mention_type

from .models import CandidateScore, EntityProfile
from .normalization import normalize_text, tokenize_for_similarity


def context_similarity(context_tokens: set[str], entity: EntityProfile) -> float:
    entity_text = " ".join([entity.canonical_name, entity.description, *entity.aliases])
    entity_tokens = tokenize_for_similarity(entity_text)
    if not context_tokens or not entity_tokens:
        return 0.0
    overlap = len(context_tokens & entity_tokens)
    return min(1.0, overlap / max(3, min(len(context_tokens), len(entity_tokens))))


def document_topic_score(topic_tokens: set[str], entity: EntityProfile) -> float:
    if not topic_tokens:
        return 0.0
    entity_tokens = tokenize_for_similarity(" ".join([entity.canonical_name, entity.description, *entity.aliases]))
    if not entity_tokens:
        return 0.0
    overlap = len(topic_tokens & entity_tokens)
    return min(1.0, overlap / max(2, min(len(topic_tokens), len(entity_tokens))))


def type_consistency(mention_type: str, entity_type: str) -> float:
    expected_type = normalize_mention_type(mention_type)
    entity_norm = normalize_entity_type(entity_type)
    if expected_type == entity_norm:
        return 1.0
    if expected_type == "CONCEPT" and entity_norm in {"MACHINE", "WORK", "AWARD"}:
        return 0.45
    if expected_type == "ORGANIZATION" and entity_norm == "PLACE":
        return 0.3
    return 0.0


def source_prior(entity: EntityProfile) -> float:
    confidence = max(0.0, min(1.0, entity.confidence))
    alias_bonus = min(0.1, len(entity.aliases) / 120)
    return max(0.1, min(1.0, confidence + alias_bonus))


def mention_confidence(value: float | None) -> float:
    if value is None:
        return 0.5
    return max(0.0, min(1.0, float(value)))


def alias_match_score(candidate: CandidateScore, mention_text: str) -> float:
    if "exact_alias" in candidate.candidate_sources:
        return 1.0
    if "normalized_alias" in candidate.candidate_sources:
        return 0.96
    if "abbreviation_alias" in candidate.candidate_sources:
        return 0.95
    if "short_name_alias" in candidate.candidate_sources or "place_variant_alias" in candidate.candidate_sources:
        return 0.9
    if "surname_alias" in candidate.candidate_sources:
        return 0.88
    if "tfidf_recall" in candidate.candidate_sources:
        return min(0.85, max(candidate.features.get("tfidf_recall_score", 0.0), 0.35))
    mention_norm = normalize_text(mention_text)
    alias_norms = {normalize_text(alias) for alias in candidate.matched_aliases}
    if mention_norm and mention_norm in alias_norms:
        return 0.96
    return 0.3


def abbreviation_match_score(candidate: CandidateScore, mention_text: str) -> float:
    mention = mention_text.strip()
    if mention.isupper() and len(mention) <= 10 and "abbreviation_alias" in candidate.candidate_sources:
        return 1.0
    return 0.0


def name_structure_score(candidate: CandidateScore, mention_text: str, mention_type: str) -> float:
    mention_norm = normalize_mention_type(mention_type)
    mention_tokens = mention_text.split()
    if mention_norm == "PERSON" and len(mention_tokens) == 1 and "surname_alias" in candidate.candidate_sources:
        return 0.95
    if mention_norm == "ORGANIZATION" and "short_name_alias" in candidate.candidate_sources:
        return 0.92
    if mention_norm == "PLACE" and "place_variant_alias" in candidate.candidate_sources:
        return 0.9
    return 0.0


def compute_linear_score(features: dict[str, float], weights: dict[str, float]) -> float:
    score = sum(features.get(name, 0.0) * weight for name, weight in weights.items())
    return max(0.0, min(1.0, score))


def decision_reason_from_scores(
    *,
    top_candidate: CandidateScore | None,
    second_score: float | None,
    decision: str,
    document_support_score: float,
) -> str:
    if decision == "SKIPPED_PRONOUN":
        return "SKIPPED_PRONOUN"
    if decision == "SKIPPED_GENERIC":
        return "SKIPPED_GENERIC"
    if decision == "SKIPPED_LOW_INFORMATION":
        return "SKIPPED_LOW_INFORMATION"
    if top_candidate is None:
        return "NO_CANDIDATE"
    if decision == "NIL":
        if top_candidate.features.get("type_consistency_score", 0.0) == 0.0:
            return "TYPE_CONFLICT"
        if second_score is not None and top_candidate.link_confidence - second_score < 0.05:
            return "AMBIGUOUS"
        if not top_candidate.candidate_sources:
            return "NO_CANDIDATE"
        return "LOW_EVIDENCE"
    if decision == "REVIEW":
        if top_candidate.features.get("type_consistency_score", 0.0) == 0.0:
            return "TYPE_CONFLICT"
        if second_score is not None and top_candidate.link_confidence - second_score < 0.05:
            return "AMBIGUOUS"
        return "LOW_EVIDENCE"
    if document_support_score > 0.05:
        return "DOCUMENT_SUPPORT_MATCH"
    if "exact_alias" in top_candidate.candidate_sources and top_candidate.link_confidence >= 0.9:
        return "HIGH_CONFIDENCE_EXACT_ALIAS"
    return "ALIAS_AND_CONTEXT_MATCH"


def safe_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
