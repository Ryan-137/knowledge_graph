from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """事实候选的原文证据，保留句子、来源和实体 span。"""

    doc_id: str
    sentence_id: str
    source_id: str
    text: str
    subject_mention_id: str
    object_mention_id: str
    subject_text: str
    object_text: str
    subject_token_span: list[int | None]
    object_token_span: list[int | None]
    subject_char_span: list[int | None] = field(default_factory=list)
    object_char_span: list[int | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "sentence_id": self.sentence_id,
            "source_id": self.source_id,
            "text": self.text,
            "subject_mention_id": self.subject_mention_id,
            "object_mention_id": self.object_mention_id,
            "subject_text": self.subject_text,
            "object_text": self.object_text,
            "subject_token_span": self.subject_token_span,
            "object_token_span": self.object_token_span,
            "subject_char_span": self.subject_char_span,
            "object_char_span": self.object_char_span,
        }


@dataclass(frozen=True)
class FactSignal:
    """单个弱监督信号，分数可正可负，方便解释最终置信度。"""

    name: str
    score: float
    label: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(float(self.score), 6),
            "label": self.label,
            "details": self.details,
        }


@dataclass(frozen=True)
class FactCandidate:
    """schema-first 的事实候选。"""

    fact_candidate_id: str
    source_candidate_id: str
    subject_id: str
    predicate: str
    object_id: str
    subject_type: str
    object_type: str
    subject_text: str
    object_text: str
    qualifiers: dict[str, Any]
    evidence: Evidence
    extractor: str
    signals: list[FactSignal] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "CANDIDATE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_candidate_id": self.fact_candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "subject_type": self.subject_type,
            "object_type": self.object_type,
            "subject_text": self.subject_text,
            "object_text": self.object_text,
            "qualifiers": self.qualifiers,
            "evidence": self.evidence.to_dict(),
            "confidence": round(float(self.confidence), 6),
            "extractor": self.extractor,
            "signals": [signal.to_dict() for signal in self.signals],
            "status": self.status,
        }


@dataclass(frozen=True)
class VerifiedFact:
    """聚合后的事实记录，后续图数据库/RDF 导出只消费这个结构。"""

    fact_id: str
    subject_id: str
    predicate: str
    object_id: str
    qualifiers: dict[str, Any]
    evidence: list[dict[str, Any]]
    confidence: float
    extractor: str
    signals: dict[str, Any]
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "qualifiers": self.qualifiers,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 6),
            "extractor": self.extractor,
            "signals": self.signals,
            "status": self.status,
        }


def signal_from_dict(payload: dict[str, Any]) -> FactSignal:
    return FactSignal(
        name=str(payload.get("name", "")),
        score=float(payload.get("score", 0.0) or 0.0),
        label=str(payload.get("label", "")),
        details=dict(payload.get("details") or {}),
    )
