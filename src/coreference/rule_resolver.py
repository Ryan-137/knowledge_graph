from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_core.entity_catalog import normalize_alias_text
from kg_core.mention_filters import is_generic_mention, is_pronoun_mention

from .anchor_builder import CoreferenceAnchor


PRONOUN_TYPE_RULES = {
    "he": {"PERSON"},
    "him": {"PERSON"},
    "his": {"PERSON"},
    "she": {"PERSON"},
    "her": {"PERSON"},
    "hers": {"PERSON"},
    "it": {"ORGANIZATION", "PLACE", "WORK", "CONCEPT", "MACHINE", "AWARD", "EVENT"},
    "its": {"ORGANIZATION", "PLACE", "WORK", "CONCEPT", "MACHINE", "AWARD", "EVENT"},
    "they": {"PERSON", "ORGANIZATION"},
    "them": {"PERSON", "ORGANIZATION"},
    "their": {"PERSON", "ORGANIZATION"},
    "theirs": {"PERSON", "ORGANIZATION"},
}

GENERIC_TYPE_RULES = {
    "the mathematician": {"PERSON"},
    "the scientist": {"PERSON"},
    "the school": {"ORGANIZATION"},
    "the laboratory": {"ORGANIZATION"},
    "the university": {"ORGANIZATION"},
    "the machine": {"MACHINE"},
    "the model": {"MACHINE", "CONCEPT", "WORK"},
    "this model": {"MACHINE", "CONCEPT", "WORK"},
    "that model": {"MACHINE", "CONCEPT", "WORK"},
    "the paper": {"WORK"},
    "this paper": {"WORK"},
    "the theory": {"CONCEPT", "WORK"},
    "this theory": {"CONCEPT", "WORK"},
    "the concept": {"CONCEPT"},
    "the method": {"CONCEPT", "WORK"},
    "the work": {"WORK"},
    "the result": {"CONCEPT", "WORK"},
    "the idea": {"CONCEPT", "WORK"},
    "the problem": {"CONCEPT", "WORK"},
}


@dataclass(frozen=True)
class CoreferenceResolution:
    anchor: CoreferenceAnchor | None
    reason: str


def _mention_position(mention: dict[str, Any], sentence_index_by_id: dict[str, int]) -> tuple[int, int]:
    sentence_id = str(mention.get("sentence_id") or "").strip()
    return int(sentence_index_by_id.get(sentence_id, 0)), int(mention.get("token_start") or 0)


def _allowed_types(normalized_text: str) -> set[str]:
    if normalized_text in PRONOUN_TYPE_RULES:
        return PRONOUN_TYPE_RULES[normalized_text]
    return GENERIC_TYPE_RULES.get(normalized_text, set())


def _has_generic_cue(normalized_text: str, anchor: CoreferenceAnchor) -> bool:
    head = normalized_text.split()[-1]
    if head in {"model", "paper", "theory", "concept", "method", "work", "result", "idea", "problem"}:
        return True
    return head in anchor.cues or anchor.linked_entity_type in _allowed_types(normalized_text)


def resolve_by_rules(
    mention: dict[str, Any],
    anchors: list[CoreferenceAnchor],
    sentence_index_by_id: dict[str, int],
    *,
    max_sentence_distance: int = 3,
) -> CoreferenceResolution:
    normalized_text = normalize_alias_text(str(mention.get("mention_text") or mention.get("text") or ""))
    if not is_pronoun_mention(normalized_text) and not is_generic_mention(normalized_text):
        return CoreferenceResolution(anchor=None, reason="NOT_COREFERENCE_TARGET")

    mention_sentence_index, mention_token_start = _mention_position(mention, sentence_index_by_id)
    allowed_types = _allowed_types(normalized_text)
    candidates: list[tuple[tuple[int, int], CoreferenceAnchor]] = []

    for anchor in anchors:
        sentence_distance = mention_sentence_index - anchor.sentence_index_in_doc
        if sentence_distance < 0 or sentence_distance > max_sentence_distance:
            continue
        if sentence_distance == 0 and anchor.token_start >= mention_token_start:
            continue
        if allowed_types and anchor.linked_entity_type not in allowed_types:
            continue
        if normalized_text in GENERIC_TYPE_RULES and not _has_generic_cue(normalized_text, anchor):
            continue
        token_rank = mention_token_start - anchor.token_start if sentence_distance == 0 else -anchor.token_start
        candidates.append(((sentence_distance, token_rank), anchor))

    if not candidates:
        return CoreferenceResolution(anchor=None, reason="NO_COMPATIBLE_ANTECEDENT")

    candidates.sort(key=lambda item: (item[0][0], item[0][1]))
    best_distance, best_anchor = candidates[0]
    if len(candidates) > 1 and best_distance[0] > 0 and candidates[1][0][0] == best_distance[0]:
        return CoreferenceResolution(anchor=None, reason="AMBIGUOUS_ANTECEDENT")
    if len(candidates) > 1 and candidates[1][0] == best_distance:
        return CoreferenceResolution(anchor=None, reason="AMBIGUOUS_ANTECEDENT")
    return CoreferenceResolution(anchor=best_anchor, reason="RULE_NEAREST_ANTECEDENT")
