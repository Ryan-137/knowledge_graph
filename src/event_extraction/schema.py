from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in ("", None, [], {})}


@dataclass(frozen=True)
class EventEvidence:
    """文本事件候选的原文证据。"""

    doc_id: str
    sentence_id: str
    source_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sentence_id": self.sentence_id,
            "source_id": self.source_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class EventRole:
    """事件论元角色，保留实体、mention 和 span 信息。"""

    role: str
    entity_id: str
    entity_type: str
    text: str
    mention_id: str = ""
    token_span: list[int | None] = field(default_factory=list)
    char_span: list[int | None] = field(default_factory=list)
    confidence: float | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _compact(
            {
                "role": self.role,
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "text": self.text,
                "mention_id": self.mention_id,
                "token_span": self.token_span,
                "char_span": self.char_span,
                "confidence": None if self.confidence is None else round(float(self.confidence), 6),
                "source": self.source,
            }
        )


EventArgument = EventRole


@dataclass(frozen=True)
class EventCandidate:
    """文本事件候选的稳定契约。"""

    event_candidate_id: str
    event_type: str
    trigger: str
    roles: list[EventRole]
    evidence: EventEvidence
    confidence: float
    extractor: str
    time_text: str = ""
    start_time_norm: str = ""
    end_time_norm: str = ""
    location_id: str = ""
    location_text: str = ""
    trigger_token_span: list[int | None] = field(default_factory=list)
    source_relation_candidate_ids: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    status: str = "CANDIDATE"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_candidate_id": self.event_candidate_id,
            "event_id": self.event_candidate_id,
            "event_type": self.event_type,
            "trigger": self.trigger,
            "roles": [role.to_dict() for role in self.roles],
            "time_text": self.time_text,
            "start_time_norm": self.start_time_norm,
            "end_time_norm": self.end_time_norm,
            "location_id": self.location_id,
            "location_text": self.location_text,
            "trigger_token_span": self.trigger_token_span,
            "source_relation_candidate_ids": self.source_relation_candidate_ids,
            "evidence": self.evidence.to_dict(),
            "confidence": round(float(self.confidence), 6),
            "extractor": self.extractor,
            "signals": self.signals,
            "status": self.status,
        }
        for optional_key in ("trigger_token_span", "source_relation_candidate_ids", "signals"):
            if payload[optional_key] in ([], {}):
                payload.pop(optional_key)
        return payload


@dataclass(frozen=True)
class VerifiedEvent:
    """校验后的事件，保持与候选事件同形，区别在 status。"""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
