from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kg_core.entity_catalog import EntityCatalog, load_entity_catalog, normalize_alias_text, normalize_exact_alias_text
from kg_core.io import read_json, read_jsonl, write_json, write_jsonl
from kg_core.mention_filters import classify_low_information_mention
from kg_core.schemas import MentionRecord
from kg_core.taxonomy import normalize_entity_type, normalize_mention_type

from .candidate_generation import build_entity_profiles
from .config import LinkingConfig
from .disambiguation import DocumentDisambiguator
from .features import (
    abbreviation_match_score,
    alias_match_score,
    compute_linear_score,
    context_similarity,
    decision_reason_from_scores,
    document_topic_score,
    mention_confidence,
    name_structure_score,
    source_prior,
    type_consistency,
)
from .models import CandidateScore, EntityProfile, MentionDraft
from .normalization import normalize_text, tokenize_for_similarity
from .reporting import build_linking_report
from .retrieval import TfidfEntityRetriever, build_entity_retrieval_texts


class EntityLinker:
    def __init__(
        self,
        entity_catalog: EntityCatalog,
        config: LinkingConfig | None = None,
    ) -> None:
        self.entity_catalog = entity_catalog
        self.entity_profiles = build_entity_profiles(entity_catalog)
        self.config = config or LinkingConfig()
        self.disambiguator = DocumentDisambiguator(entity_catalog.claims_adjacency, self.config)
        self.retriever = TfidfEntityRetriever(
            build_entity_retrieval_texts(self.entity_profiles, entity_catalog.retrieval_text_by_entity)
        )

    def link_mentions(
        self,
        mentions: list[dict[str, Any]],
        sentences: list[dict[str, Any]],
        documents: list[dict[str, Any]] | None = None,
        ontology: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        _ = ontology
        sentence_by_id = {str(item["sentence_id"]): item for item in sentences}
        doc_title_by_id = {
            str(item.get("doc_id", "")): str(item.get("title", "")).strip()
            for item in (documents or [])
            if item.get("doc_id")
        }
        drafts = [
            self._build_local_draft(self._coerce_mention_record(mention), sentence_by_id, doc_title_by_id)
            for mention in mentions
        ]
        self.disambiguator.apply(drafts)
        return [self._decide_link(draft) for draft in drafts]

    def _coerce_mention_record(self, mention: dict[str, Any]) -> dict[str, Any]:
        normalized_mention = dict(mention)
        if "mention_type" not in normalized_mention and "mention_type_hint" in normalized_mention:
            normalized_mention["mention_type"] = normalized_mention.get("mention_type_hint")
        normalized_mention["mention_type"] = normalize_mention_type(normalized_mention.get("mention_type"))
        normalized_mention.setdefault("source_id", "")
        normalized_mention["normalized_text"] = normalize_alias_text(normalized_mention.get("text", ""))
        return MentionRecord(**{
            "mention_id": normalized_mention["mention_id"],
            "sentence_id": normalized_mention["sentence_id"],
            "doc_id": normalized_mention["doc_id"],
            "source_id": normalized_mention.get("source_id", ""),
            "text": normalized_mention["text"],
            "normalized_text": normalized_mention["normalized_text"],
            "mention_type": normalized_mention["mention_type"],
            "char_start": normalized_mention["char_start"],
            "char_end": normalized_mention["char_end"],
            "token_start": normalized_mention["token_start"],
            "token_end": normalized_mention["token_end"],
            "extractor": normalized_mention["extractor"],
            "confidence": normalized_mention.get("confidence"),
            "recall_source": normalized_mention.get("recall_source", ""),
        }).to_dict()

    def _classify_skip_reason(self, mention: dict[str, Any]) -> str | None:
        return classify_low_information_mention(str(mention.get("text") or "").strip())

    def _build_local_draft(
        self,
        mention: dict[str, Any],
        sentence_by_id: dict[str, dict[str, Any]],
        doc_title_by_id: dict[str, str],
    ) -> MentionDraft:
        sentence = sentence_by_id.get(str(mention.get("sentence_id", "")), {})
        context_window = str(sentence.get("text") or "")
        doc_id = str(mention.get("doc_id", ""))
        doc_title = doc_title_by_id.get(doc_id, "")
        skip_reason = self._classify_skip_reason(mention)
        return MentionDraft(
            mention=mention,
            context_window=context_window,
            doc_title=doc_title,
            skip_reason=skip_reason,
            candidates=[] if skip_reason else self._generate_candidates(mention, context_window, doc_title),
        )

    def _add_candidate(
        self,
        candidate_map: dict[str, CandidateScore],
        entity_id: str,
        matched_alias: str,
        source: str,
        alias_type: str,
    ) -> None:
        entity = self.entity_profiles.get(entity_id)
        if entity is None:
            return
        candidate = candidate_map.get(entity_id)
        if candidate is None:
            candidate = CandidateScore(entity=entity, matched_aliases=[], candidate_sources=set())
            candidate_map[entity_id] = candidate
        if matched_alias and matched_alias not in candidate.matched_aliases:
            candidate.matched_aliases.append(matched_alias)
        candidate.candidate_sources.add(source)
        if alias_type:
            candidate.alias_types.add(alias_type)

    def _generate_candidates(
        self,
        mention: dict[str, Any],
        context_window: str,
        doc_title: str,
    ) -> list[CandidateScore]:
        mention_text = str(mention.get("text") or "").strip()
        mention_type = normalize_mention_type(mention.get("mention_type"))
        exact_key = normalize_exact_alias_text(mention_text)
        normalized_key = normalize_alias_text(mention_text)
        candidate_map: dict[str, CandidateScore] = {}

        def add_rows(rows: list[dict[str, Any]], source_name: str) -> None:
            for row in rows:
                self._add_candidate(
                    candidate_map,
                    entity_id=str(row.get("entity_id") or "").strip(),
                    matched_alias=str(row.get("alias") or "").strip(),
                    source=source_name,
                    alias_type=str(row.get("alias_type") or ""),
                )

        add_rows(self.entity_catalog.exact_alias_index.get(exact_key, []), "exact_alias")
        add_rows(self.entity_catalog.normalized_alias_index.get(normalized_key, []), "normalized_alias")
        if mention_type == "PERSON":
            add_rows(self.entity_catalog.person_surname_index.get(normalized_key, []), "surname_alias")
        if mention_type == "ORGANIZATION":
            add_rows(self.entity_catalog.organization_abbreviation_index.get(normalized_key, []), "abbreviation_alias")
            add_rows(self.entity_catalog.full_short_name_index.get(normalized_key, []), "short_name_alias")
        if mention_type == "PLACE":
            add_rows(self.entity_catalog.place_variant_index.get(normalized_key, []), "place_variant_alias")

        retrieval_query = " ".join(part for part in [mention_text, context_window, doc_title] if part).strip()
        for hit in self.retriever.search(retrieval_query, self.config.tfidf_recall_limit):
            candidate = candidate_map.get(hit.entity_id)
            if candidate is None:
                self._add_candidate(
                    candidate_map,
                    entity_id=hit.entity_id,
                    matched_alias=self.entity_profiles[hit.entity_id].canonical_name,
                    source="tfidf_recall",
                    alias_type="retrieval",
                )
                candidate = candidate_map[hit.entity_id]
            candidate.features["tfidf_recall_score"] = max(candidate.features.get("tfidf_recall_score", 0.0), hit.score)

        context_tokens = tokenize_for_similarity(context_window)
        topic_tokens = tokenize_for_similarity(" ".join(part for part in [doc_title, context_window] if part))
        for candidate in candidate_map.values():
            candidate.features = self._score_candidate(
                mention=mention,
                mention_type=mention_type,
                context_tokens=context_tokens,
                topic_tokens=topic_tokens,
                candidate=candidate,
            )
            candidate.local_score = compute_linear_score(candidate.features, self.config.weights)
            candidate.link_confidence = candidate.local_score
            candidate.final_score = candidate.local_score

        return sorted(candidate_map.values(), key=lambda item: item.local_score, reverse=True)[: self.config.top_k]

    def _score_candidate(
        self,
        mention: dict[str, Any],
        mention_type: str,
        context_tokens: set[str],
        topic_tokens: set[str],
        candidate: CandidateScore,
    ) -> dict[str, float]:
        mention_text = str(mention.get("text") or "")
        return {
            "alias_match_score": alias_match_score(candidate, mention_text),
            "type_consistency_score": type_consistency(mention_type, candidate.entity.entity_type),
            "context_similarity_score": context_similarity(context_tokens, candidate.entity),
            "document_topic_score": document_topic_score(topic_tokens, candidate.entity),
            "mention_confidence_score": mention_confidence(mention.get("confidence")),
            "source_prior_score": source_prior(candidate.entity),
            "abbreviation_match_score": abbreviation_match_score(candidate, mention_text),
            "name_structure_score": name_structure_score(candidate, mention_text, mention_type),
            "tfidf_recall_score": candidate.features.get("tfidf_recall_score", 0.0),
        }

    def _determine_link_threshold(self, draft: MentionDraft, top_candidate: CandidateScore) -> float:
        token_count = int(draft.mention.get("token_end", 0)) - int(draft.mention.get("token_start", 0))
        if token_count <= 1:
            second_score = draft.candidates[1].link_confidence if len(draft.candidates) > 1 else 0.0
            margin = top_candidate.link_confidence - second_score
            direct_alias_match = bool(top_candidate.candidate_sources & {"exact_alias", "normalized_alias", "surname_alias"})
            type_consistent = top_candidate.features.get("type_consistency_score", 0.0) >= 1.0
            # 单 token 姓氏/简称不能一概放宽，只有直连别名且候选优势明显时才降低阈值。
            if direct_alias_match and type_consistent and margin >= 0.2:
                return min(self.config.single_token_link_threshold, 0.8)
            return self.config.single_token_link_threshold
        if "exact_alias" in top_candidate.candidate_sources or "normalized_alias" in top_candidate.candidate_sources:
            return self.config.multi_token_link_threshold
        return self.config.link_threshold

    def _decide_link(self, draft: MentionDraft) -> dict[str, Any]:
        mention = draft.mention
        candidates = draft.candidates
        base_output = {
            "mention_id": mention.get("mention_id"),
            "sentence_id": mention.get("sentence_id"),
            "doc_id": mention.get("doc_id"),
            "source_id": mention.get("source_id"),
            "mention_text": mention.get("text"),
            "normalized_mention_text": mention.get("normalized_text"),
            "mention_type": mention.get("mention_type"),
            "token_start": mention.get("token_start"),
            "token_end": mention.get("token_end"),
            "char_start": mention.get("char_start"),
            "char_end": mention.get("char_end"),
            "extractor": mention.get("extractor"),
            "mention_confidence": mention.get("confidence"),
            "recall_source": mention.get("recall_source"),
            "context_window": draft.context_window,
            "doc_title": draft.doc_title,
            "candidate_list": [candidate.to_output() for candidate in candidates],
            "top_candidates": [candidate.to_output() for candidate in candidates[: self.config.top_k]],
        }
        if draft.skip_reason is not None:
            return {
                **base_output,
                "decision": draft.skip_reason,
                "link_status": draft.skip_reason,
                "entity_id": None,
                "canonical_name": None,
                "linked_entity_type": None,
                "local_score": 0.0,
                "document_support_score": 0.0,
                "link_confidence": 0.0,
                "final_score": 0.0,
                "score_margin": None,
                "nil_reason": draft.skip_reason,
                "decision_reason": draft.skip_reason,
                "resolution_stage": "SKIPPED",
            }

        if not candidates:
            return {
                **base_output,
                "decision": "NIL",
                "link_status": "NIL",
                "entity_id": None,
                "canonical_name": None,
                "linked_entity_type": None,
                "local_score": 0.0,
                "document_support_score": 0.0,
                "link_confidence": 0.0,
                "final_score": 0.0,
                "score_margin": None,
                "nil_reason": "NO_CANDIDATE",
                "decision_reason": "NO_CANDIDATE",
                "resolution_stage": "LOCAL_ONLY",
            }

        top_candidate = candidates[0]
        second_score = candidates[1].link_confidence if len(candidates) > 1 else None
        margin = top_candidate.link_confidence - second_score if second_score is not None else None
        link_threshold = self._determine_link_threshold(draft, top_candidate)
        decision = "NIL"
        if top_candidate.link_confidence >= link_threshold:
            decision = "LINKED"
        elif top_candidate.link_confidence >= self.config.review_threshold:
            decision = "REVIEW"
        if margin is not None and margin < self.config.ambiguity_margin_threshold and top_candidate.link_confidence >= self.config.review_threshold:
            decision = "REVIEW"
        if top_candidate.features.get("type_consistency_score", 0.0) == 0.0 and decision == "LINKED":
            decision = "REVIEW"
        decision_reason = decision_reason_from_scores(
            top_candidate=top_candidate,
            second_score=second_score,
            decision=decision,
            document_support_score=top_candidate.document_support_score,
        )
        resolution_stage = "LOCAL_WITH_DOCUMENT_SUPPORT" if top_candidate.document_support_score > 0 else "LOCAL_ONLY"
        return {
            **base_output,
            "decision": decision,
            "link_status": decision,
            "entity_id": top_candidate.entity.entity_id if decision == "LINKED" else None,
            "canonical_name": top_candidate.entity.canonical_name if decision == "LINKED" else None,
            "entity_type": top_candidate.entity.entity_type,
            "linked_entity_type": normalize_entity_type(top_candidate.entity.entity_type) if decision == "LINKED" else None,
            "local_score": round(top_candidate.local_score, 6),
            "document_support_score": round(top_candidate.document_support_score, 6),
            "link_confidence": round(top_candidate.link_confidence, 6),
            "final_score": round(top_candidate.link_confidence, 6),
            "score_margin": round(margin, 6) if margin is not None else None,
            "nil_reason": decision_reason if decision != "LINKED" else None,
            "decision_reason": decision_reason,
            "resolution_stage": resolution_stage,
        }


def _build_sidecar_paths(output_path: Path, report_path: Path | None) -> dict[str, Path]:
    base_dir = report_path.parent if report_path is not None else output_path.parent
    return {
        "candidates": base_dir / "linking_candidates.jsonl",
        "review": base_dir / "linking_review.jsonl",
        "nil": base_dir / "nil_mentions.jsonl",
    }


def _split_linking_outputs(linked_mentions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        {
            "mention_id": item.get("mention_id"),
            "doc_id": item.get("doc_id"),
            "sentence_id": item.get("sentence_id"),
            "mention_text": item.get("mention_text"),
            "mention_type": item.get("mention_type"),
            "candidate_list": item.get("candidate_list", []),
            "top_candidates": item.get("top_candidates", []),
            "decision": item.get("decision"),
            "decision_reason": item.get("decision_reason"),
        }
        for item in linked_mentions
    ]
    review = [item for item in linked_mentions if item.get("decision") == "REVIEW"]
    nil_mentions = [item for item in linked_mentions if item.get("decision") == "NIL"]
    return candidates, review, nil_mentions


def link_mentions_from_paths(
    mentions_path: str | Path,
    sentences_path: str | Path,
    documents_path: str | Path | None,
    entities_csv_path: str | Path,
    aliases_csv_path: str | Path,
    claims_csv_path: str | Path,
    ontology_path: str | Path | None,
    output_path: str | Path,
    report_path: str | Path | None = None,
    config: LinkingConfig | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    mentions = read_jsonl(mentions_path)
    if limit is not None:
        mentions = mentions[:limit]
    sentences = read_jsonl(sentences_path)
    documents = read_jsonl(documents_path) if documents_path and Path(documents_path).exists() else []
    ontology = read_json(ontology_path) if ontology_path else None
    catalog = load_entity_catalog(entities_csv_path, aliases_csv_path, claims_csv_path)
    linker = EntityLinker(entity_catalog=catalog, config=config)
    linked_mentions = linker.link_mentions(mentions=mentions, sentences=sentences, documents=documents, ontology=ontology)
    output_path = Path(output_path)
    write_jsonl(output_path, linked_mentions)
    resolved_report_path = Path(report_path) if report_path is not None else None
    candidates, review, nil_mentions = _split_linking_outputs(linked_mentions)
    sidecar_paths = _build_sidecar_paths(output_path, resolved_report_path)
    write_jsonl(sidecar_paths["candidates"], candidates)
    write_jsonl(sidecar_paths["review"], review)
    write_jsonl(sidecar_paths["nil"], nil_mentions)
    if resolved_report_path is not None:
        write_json(resolved_report_path, build_linking_report(linked_mentions))
    return linked_mentions
