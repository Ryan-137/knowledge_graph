from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LinkingConfig:
    """实体链接第一版配置：可解释线性打分，不依赖监督模型。"""

    top_k: int = 5
    link_threshold: float = 0.85
    multi_token_link_threshold: float = 0.82
    single_token_link_threshold: float = 0.88
    review_threshold: float = 0.65
    ambiguity_margin_threshold: float = 0.05
    fuzzy_threshold: float = 0.78
    tfidf_recall_limit: int = 8
    anchor_threshold: float = 0.88
    enable_document_disambiguation: bool = True
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "alias_match_score": 0.30,
            "context_similarity_score": 0.20,
            "type_consistency_score": 0.20,
            "document_topic_score": 0.10,
            "mention_confidence_score": 0.10,
            "source_prior_score": 0.10,
            "abbreviation_match_score": 0.00,
            "name_structure_score": 0.00,
        }
    )
