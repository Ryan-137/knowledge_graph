from __future__ import annotations

import hashlib
import re
from typing import Any

from kg_core.taxonomy import normalize_entity_type

from .schema import EventCandidate, EventEvidence, EventRole


_TRIGGER_TEMPLATES = sorted(
    [
        {
            "event_type": "CollaborationEvent",
            "trigger": "worked with",
            "pattern": r"\bworked\s+with\b",
            "left_role": "person_a",
            "right_role": "person_b",
            "passive": False,
            "trigger_pattern_id": "collaboration_worked_with",
            "template_confidence": 0.86,
        },
        {
            "event_type": "CollaborationEvent",
            "trigger": "collaborated with",
            "pattern": r"\bcollaborated\s+with\b",
            "left_role": "person_a",
            "right_role": "person_b",
            "passive": False,
            "trigger_pattern_id": "collaboration_collaborated_with",
            "template_confidence": 0.86,
        },
        {
            "event_type": "InfluenceEvent",
            "trigger": "was influenced by",
            "pattern": r"\bwas\s+influenced\s+by\b",
            "left_role": "target_person",
            "right_role": "source_person",
            "passive": True,
            "trigger_pattern_id": "influence_passive_was_influenced_by",
            "template_confidence": 0.88,
        },
        {
            "event_type": "InfluenceEvent",
            "trigger": "influenced",
            "pattern": r"\binfluenced\b",
            "left_role": "source_person",
            "right_role": "target_person",
            "passive": False,
            "trigger_pattern_id": "influence_active_influenced",
            "template_confidence": 0.8,
        },
    ],
    key=lambda item: len(str(item["trigger"])),
    reverse=True,
)


def extract_event_candidates(sentence: dict[str, Any], *, entity_mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """基于少量高精度模板抽取第一版文本事件。"""

    text = str(sentence.get("text") or "")
    person_mentions = [_normalize_mention(mention, sentence) for mention in entity_mentions if _is_person(mention)]
    generated: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []
    for template in _TRIGGER_TEMPLATES:
        event_type = str(template["event_type"])
        trigger = str(template["trigger"])
        pattern = str(template["pattern"])
        left_role = str(template["left_role"])
        right_role = str(template["right_role"])
        passive = bool(template["passive"])
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if _span_overlaps(match.start(), match.end(), occupied_spans):
                continue
            if trigger == "influenced" and _is_passive_influence(text, match.start()):
                continue
            left, right = _nearest_argument_pair(person_mentions, match.start(), match.end())
            if left is None or right is None:
                continue
            review_reasons = _trigger_review_reasons(text, trigger, match.end(), right)
            template_confidence = float(template["template_confidence"])
            confidence = 0.49 if review_reasons else template_confidence
            roles = [
                _role_from_mention(left_role, left),
                _role_from_mention(right_role, right),
            ]
            if passive:
                roles = [
                    _role_from_mention(right_role, right),
                    _role_from_mention(left_role, left),
                ]
            event_id = _event_id(sentence, event_type, trigger, roles)
            event = EventCandidate(
                event_candidate_id=event_id,
                event_type=event_type,
                trigger=trigger,
                roles=roles,
                evidence=EventEvidence(
                    doc_id=str(sentence.get("doc_id") or ""),
                    sentence_id=str(sentence.get("sentence_id") or ""),
                    source_id=str(sentence.get("source_id") or ""),
                    text=text,
                ),
                trigger_token_span=_trigger_token_span(sentence, match.start(), match.end()),
                confidence=confidence,
                extractor="event_pattern_rules",
                signals=[
                    {
                        "name": "event_trigger_match",
                        "score": 0.2,
                        "label": "MATCHED",
                        "details": {
                            "trigger": trigger,
                            "trigger_pattern_id": template["trigger_pattern_id"],
                            "template_confidence": template_confidence,
                        },
                    }
                ],
            ).to_dict()
            event["trigger_pattern_id"] = template["trigger_pattern_id"]
            event["template_confidence"] = template_confidence
            if review_reasons:
                event["review_reasons"] = {"trigger_review_reasons": review_reasons}
            generated.append(event)
            occupied_spans.append((match.start(), match.end()))
    return generated


def _normalize_mention(mention: dict[str, Any], sentence: dict[str, Any]) -> dict[str, Any]:
    sentence_text = str(sentence.get("text") or "")
    span = list(mention.get("token_span") or mention.get("entity_token_span") or [])
    if len(span) != 2:
        span = [mention.get("token_start"), mention.get("token_end")]
    mention_text = str(mention.get("text") or mention.get("mention_text") or mention.get("surface") or "")
    char_span = list(mention.get("char_span") or [])
    if len(char_span) != 2:
        char_span = _char_span_from_token_span(sentence, span)
    if len(char_span) != 2 and mention_text:
        start = sentence_text.casefold().find(mention_text.casefold())
        char_span = [start, start + len(mention_text)] if start >= 0 else []
    return {
        "mention_id": str(mention.get("mention_id") or mention.get("id") or ""),
        "entity_id": str(mention.get("entity_id") or mention.get("resolved_entity_id") or ""),
        "entity_type": normalize_entity_type(mention.get("entity_type") or mention.get("resolved_entity_type")),
        "text": mention_text,
        "token_span": span if len(span) == 2 and all(value is not None for value in span) else [],
        "char_span": char_span,
        "confidence": mention.get("confidence"),
        "source": str(mention.get("source") or mention.get("resolution") or ""),
    }


def _span_overlaps(start: int, end: int, occupied_spans: list[tuple[int, int]]) -> bool:
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans)


def _is_passive_influence(text: str, trigger_start: int) -> bool:
    return re.search(r"\bwas\s+$", text[:trigger_start], flags=re.IGNORECASE) is not None


def _trigger_review_reasons(text: str, trigger: str, trigger_end: int, right_mention: dict[str, Any]) -> list[str]:
    """主动 influence 的右侧 PERSON 落入介词短语时，只保留 review 证据。"""

    if trigger != "influenced":
        return []
    right_start = _mention_text_start(right_mention)
    if right_start < trigger_end:
        return []
    between = text[trigger_end:right_start]
    if re.search(r"\b(by|of|from)\b\s*$", between, flags=re.IGNORECASE):
        return ["active_influence_right_argument_in_prepositional_phrase"]
    return []


def _char_span_from_token_span(sentence: dict[str, Any], token_span: list[Any]) -> list[int]:
    token_spans = list(sentence.get("token_spans") or [])
    if len(token_span) != 2 or not token_spans:
        return []
    start_token = int(token_span[0])
    end_token = int(token_span[1])
    if start_token < 0 or end_token <= start_token or end_token > len(token_spans):
        return []
    start_span = token_spans[start_token]
    end_span = token_spans[end_token - 1]
    if len(start_span) != 2 or len(end_span) != 2:
        return []
    return [int(start_span[0]), int(end_span[1])]


def _is_person(mention: dict[str, Any]) -> bool:
    return normalize_entity_type(mention.get("entity_type") or mention.get("resolved_entity_type")) == "PERSON"


def _nearest_argument_pair(mentions: list[dict[str, Any]], trigger_start: int, trigger_end: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    left_candidates = [mention for mention in mentions if _mention_text_end(mention) <= trigger_start]
    right_candidates = [mention for mention in mentions if _mention_text_start(mention) >= trigger_end]
    left = max(left_candidates, key=_mention_text_end) if left_candidates else None
    right = min(right_candidates, key=_mention_text_start) if right_candidates else None
    return left, right


def _mention_text_start(mention: dict[str, Any]) -> int:
    return int(mention.get("_char_start", -1)) if "_char_start" in mention else _find_text_bound(mention, start=True)


def _mention_text_end(mention: dict[str, Any]) -> int:
    return int(mention.get("_char_end", -1)) if "_char_end" in mention else _find_text_bound(mention, start=False)


def _find_text_bound(mention: dict[str, Any], *, start: bool) -> int:
    char_span = list(mention.get("char_span") or [])
    if len(char_span) == 2:
        return int(char_span[0 if start else 1])
    token_span = list(mention.get("token_span") or [])
    if len(token_span) == 2:
        return int(token_span[0 if start else 1]) * 100
    return -1 if start else -1


def _role_from_mention(role: str, mention: dict[str, Any]) -> EventRole:
    return EventRole(
        role=role,
        entity_id=str(mention.get("entity_id") or ""),
        entity_type=normalize_entity_type(mention.get("entity_type")),
        text=str(mention.get("text") or ""),
        mention_id=str(mention.get("mention_id") or ""),
        token_span=list(mention.get("token_span") or []),
        char_span=list(mention.get("char_span") or []),
        confidence=mention.get("confidence"),
        source=str(mention.get("source") or ""),
    )


def _event_id(sentence: dict[str, Any], event_type: str, trigger: str, roles: list[EventRole]) -> str:
    payload = "|".join(
        [
            str(sentence.get("sentence_id") or ""),
            event_type,
            trigger,
            *[f"{role.role}:{role.entity_id}" for role in roles],
        ]
    )
    return "eventcand_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _trigger_token_span(sentence: dict[str, Any], start_char: int, end_char: int) -> list[int]:
    token_spans = list(sentence.get("token_spans") or [])
    covered = [
        index
        for index, span in enumerate(token_spans)
        if len(span) == 2 and int(span[0]) < end_char and int(span[1]) > start_char
    ]
    return [covered[0], covered[-1] + 1] if covered else []
