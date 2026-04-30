from __future__ import annotations

from collections import Counter, defaultdict

from .config import LinkingConfig
from .models import MentionDraft
from .normalization import normalize_text, tokenize_for_similarity


class DocumentDisambiguator:
    """一期只做轻量文档支持，不引入复杂 collective EL。"""

    def __init__(self, claim_adjacency: dict[str, set[str]], config: LinkingConfig) -> None:
        self.claim_adjacency = claim_adjacency
        self.config = config

    def apply(self, drafts: list[MentionDraft]) -> None:
        if not self.config.enable_document_disambiguation:
            return

        doc_to_anchor_counts: dict[str, Counter[str]] = defaultdict(Counter)
        doc_to_exact_surface_entity: dict[str, dict[str, str]] = defaultdict(dict)
        for draft in drafts:
            if draft.skip_reason or not draft.candidates:
                continue
            top_candidate = draft.candidates[0]
            if top_candidate.local_score >= self.config.anchor_threshold:
                doc_to_anchor_counts[str(draft.mention.get("doc_id", ""))][top_candidate.entity.entity_id] += 1
            if "exact_alias" in top_candidate.candidate_sources and top_candidate.local_score >= self.config.review_threshold:
                doc_to_exact_surface_entity[str(draft.mention.get("doc_id", ""))][
                    normalize_text(str(draft.mention.get("text") or ""))
                ] = top_candidate.entity.entity_id

        for draft in drafts:
            if draft.skip_reason or not draft.candidates:
                continue
            doc_id = str(draft.mention.get("doc_id", ""))
            mention_surface = normalize_text(str(draft.mention.get("text") or ""))
            title_tokens = tokenize_for_similarity(draft.doc_title)
            anchors = doc_to_anchor_counts.get(doc_id, Counter())
            exact_surface_entity = doc_to_exact_surface_entity.get(doc_id, {})
            for candidate in draft.candidates:
                support = 0.0
                if exact_surface_entity.get(mention_surface) == candidate.entity.entity_id:
                    support += 0.05
                if anchors.get(candidate.entity.entity_id, 0) > 0:
                    support += min(0.06, anchors[candidate.entity.entity_id] * 0.02)
                if title_tokens:
                    candidate_tokens = tokenize_for_similarity(
                        " ".join([candidate.entity.canonical_name, candidate.entity.description, *candidate.entity.aliases])
                    )
                    if candidate_tokens and (candidate_tokens & title_tokens):
                        support += 0.03
                candidate.document_support_score = round(min(0.15, support), 6)
                candidate.link_confidence = round(min(1.0, candidate.local_score + candidate.document_support_score), 6)
                candidate.final_score = candidate.link_confidence
            draft.candidates.sort(key=lambda item: item.link_confidence, reverse=True)
