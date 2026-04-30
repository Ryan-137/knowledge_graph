from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 非结构化原始输入数据
@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    tier: int
    authority_level: str
    source_type: str
    original_url: str
    raw_path: str
    organization: str
    verification_status: str
    notes: str = ""

#清洗后的document
@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    source_id: str
    title: str
    tier: int
    language: str
    clean_text: str
    authority_level: str = ""
    source_type: str = ""
    original_url: str = ""
    raw_path: str = ""
    organization: str = ""
    verification_status: str = ""
    char_count: int = 0
    processed_at: str = ""
    notes: str = ""

# 句子里的time
@dataclass(frozen=True)
class TimeMentionRecord:
    text: str
    normalized: str
    type: str
    offset_start: int
    offset_end: int

#清洗后的sentence
@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    doc_id: str
    source_id: str
    sentence_index_in_doc: int
    text: str
    offset_start: int
    offset_end: int
    normalized_time: list[str] = field(default_factory=list)
    time_mentions: list[TimeMentionRecord] = field(default_factory=list)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class TokenizedSentence:
    sentence_id: str
    doc_id: str
    source_id: str
    sentence_index_in_doc: int
    text: str
    tokens: list[str]
    token_spans: list[list[int]]
    normalized_time: list[str]
    time_mentions: list[dict[str, Any]]


# mention结果 （新旧兼容，最好修改）
@dataclass(frozen=True)
class MentionRecord:
    mention_id: str
    sentence_id: str
    doc_id: str
    source_id: str
    text: str
    normalized_text: str
    mention_type: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    extractor: str
    confidence: float | None
    recall_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "mention_type": self.mention_type,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "recall_source": self.recall_source,
        }


@dataclass(frozen=True)
class LinkedMentionRecord:
    """实体链接后的 mention 统一结构，供关系抽取直接消费。"""

    mention_id: str
    sentence_id: str
    doc_id: str
    source_id: str
    text: str
    normalized_text: str
    mention_type: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    extractor: str
    mention_confidence: float | None
    recall_source: str
    decision: str
    link_status: str = ""
    entity_id: str = ""
    canonical_name: str = ""
    entity_type: str = ""
    linked_entity_type: str = ""
    local_score: float | None = None
    document_support_score: float | None = None
    link_confidence: float | None = None
    final_score: float | None = None
    score_margin: float | None = None
    nil_reason: str = ""
    decision_reason: str = ""
    resolution_stage: str = ""
    context_window: str = ""
    candidate_list: list[dict[str, Any]] = field(default_factory=list)
    top_candidates: list[dict[str, Any]] = field(default_factory=list)
    matched_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id,
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "mention_type": self.mention_type,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "extractor": self.extractor,
            "mention_confidence": self.mention_confidence,
            "recall_source": self.recall_source,
            "decision": self.decision,
            "link_status": self.link_status,
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "linked_entity_type": self.linked_entity_type,
            "local_score": self.local_score,
            "document_support_score": self.document_support_score,
            "link_confidence": self.link_confidence,
            "final_score": self.final_score,
            "score_margin": self.score_margin,
            "nil_reason": self.nil_reason,
            "decision_reason": self.decision_reason,
            "resolution_stage": self.resolution_stage,
            "context_window": self.context_window,
            "candidate_list": self.candidate_list,
            "top_candidates": self.top_candidates,
            "matched_features": self.matched_features,
        }


@dataclass(frozen=True)
class RelationPairRecord:
    """关系抽取 prepare 阶段生成的实体对候选。"""

    pair_id: str
    sentence_id: str
    doc_id: str
    source_id: str
    sentence_text: str
    subject_mention_id: str
    subject_entity_id: str
    subject_text: str
    subject_entity_type: str
    object_mention_id: str
    object_entity_id: str
    object_text: str
    object_entity_type: str
    token_distance: int | None = None
    candidate_predicates: list[str] = field(default_factory=list)
    normalized_time: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "sentence_text": self.sentence_text,
            "subject_mention_id": self.subject_mention_id,
            "subject_entity_id": self.subject_entity_id,
            "subject_text": self.subject_text,
            "subject_entity_type": self.subject_entity_type,
            "object_mention_id": self.object_mention_id,
            "object_entity_id": self.object_entity_id,
            "object_text": self.object_text,
            "object_entity_type": self.object_entity_type,
            "token_distance": self.token_distance,
            "candidate_predicates": self.candidate_predicates,
            "normalized_time": self.normalized_time,
        }


@dataclass(frozen=True)
class DistantLabeledRelationRecord:
    """远程监督生成的关系训练样本。"""

    pair_id: str
    sentence_id: str
    doc_id: str
    source_id: str
    sentence_text: str
    subject_mention_id: str
    subject_entity_id: str
    subject_text: str
    subject_entity_type: str
    object_mention_id: str
    object_entity_id: str
    object_text: str
    object_entity_type: str
    relation_label: str
    is_positive: bool
    label_source: str
    evidence_claim_id: str = ""
    label_confidence: float | None = None
    normalized_time: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "sentence_text": self.sentence_text,
            "subject_mention_id": self.subject_mention_id,
            "subject_entity_id": self.subject_entity_id,
            "subject_text": self.subject_text,
            "subject_entity_type": self.subject_entity_type,
            "object_mention_id": self.object_mention_id,
            "object_entity_id": self.object_entity_id,
            "object_text": self.object_text,
            "object_entity_type": self.object_entity_type,
            "relation_label": self.relation_label,
            "is_positive": self.is_positive,
            "label_source": self.label_source,
            "evidence_claim_id": self.evidence_claim_id,
            "label_confidence": self.label_confidence,
            "normalized_time": self.normalized_time,
        }


@dataclass(frozen=True)
class ExtractedClaimRecord:
    """关系预测阶段输出的文本事实。"""

    claim_id: str
    subject_id: str
    predicate: str
    object_id: str
    evidence_sentence_id: str
    doc_id: str
    source_id: str
    extractor: str
    confidence: float | None
    subject_mention_id: str = ""
    object_mention_id: str = ""
    subject_text: str = ""
    object_text: str = ""
    evidence_text: str = ""
    qualifiers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "evidence_sentence_id": self.evidence_sentence_id,
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "extractor": self.extractor,
            "confidence": self.confidence,
            "subject_mention_id": self.subject_mention_id,
            "object_mention_id": self.object_mention_id,
            "subject_text": self.subject_text,
            "object_text": self.object_text,
            "evidence_text": self.evidence_text,
            "qualifiers": self.qualifiers,
        }


def normalize_mention_text(text: str) -> str:
    """统一 mention 规范文本，供 linking、去重和评测复用。"""

    return " ".join(text.casefold().split())


## 好像主要是结构化库在调用下面三个
@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    description: str = ""
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AliasRecord:
    alias: str
    entity_id: str
    entity_type: str
    source: str = ""


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    subject_id: str
    predicate: str
    object_id: str
    extra: dict[str, Any] = field(default_factory=dict)
