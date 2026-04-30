from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityProfile:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    description: str
    source_name: str | None
    confidence: float


@dataclass
class CandidateScore:
    entity: EntityProfile
    matched_aliases: list[str]
    candidate_sources: set[str]
    alias_types: set[str] = field(default_factory=set)
    features: dict[str, float] = field(default_factory=dict)
    local_score: float = 0.0
    document_support_score: float = 0.0
    link_confidence: float = 0.0
    final_score: float = 0.0

    def to_output(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity.entity_id,
            "canonical_name": self.entity.canonical_name,
            "entity_type": self.entity.entity_type,
            "score": round(self.link_confidence or self.final_score, 6),
            "local_score": round(self.local_score, 6),
            "document_support_score": round(self.document_support_score, 6),
            "matched_aliases": self.matched_aliases[:5],
            "candidate_sources": sorted(self.candidate_sources),
            "alias_types": sorted(self.alias_types),
            "matched_features": {
                name: round(value, 6) for name, value in sorted(self.features.items())
            },
        }


@dataclass
class MentionDraft:
    mention: dict[str, Any]
    context_window: str
    doc_title: str
    candidates: list[CandidateScore]
    skip_reason: str | None = None
