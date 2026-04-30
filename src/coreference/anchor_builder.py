from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_core.entity_catalog import normalize_alias_text
from kg_core.taxonomy import normalize_entity_type


@dataclass(frozen=True)
class CoreferenceAnchor:
    mention_id: str
    doc_id: str
    sentence_id: str
    sentence_index_in_doc: int
    token_start: int
    entity_id: str
    canonical_name: str
    linked_entity_type: str
    cues: set[str]


def _token_cues(text: str) -> set[str]:
    return {
        token
        for token in normalize_alias_text(text).split()
        if len(token) > 2 and token not in {"the", "this", "that", "and", "for", "with"}
    }


def build_anchor(record: dict[str, Any], sentence_index_by_id: dict[str, int]) -> CoreferenceAnchor | None:
    decision = str(record.get("decision") or "").upper()
    entity_id = str(record.get("entity_id") or "").strip()
    if decision != "LINKED" or not entity_id:
        return None

    cues: set[str] = set()
    for value in (
        record.get("mention_text"),
        record.get("canonical_name"),
        record.get("normalized_mention_text"),
        record.get("context_window"),
    ):
        cues.update(_token_cues(str(value or "")))
    for candidate in record.get("top_candidates") or []:
        if str(candidate.get("entity_id") or "").strip() != entity_id:
            continue
        for alias in candidate.get("matched_aliases") or []:
            cues.update(_token_cues(str(alias)))

    sentence_id = str(record.get("sentence_id") or "").strip()
    return CoreferenceAnchor(
        mention_id=str(record.get("mention_id") or "").strip(),
        doc_id=str(record.get("doc_id") or "").strip(),
        sentence_id=sentence_id,
        sentence_index_in_doc=int(sentence_index_by_id.get(sentence_id, 0)),
        token_start=int(record.get("token_start") or 0),
        entity_id=entity_id,
        canonical_name=str(record.get("canonical_name") or "").strip(),
        linked_entity_type=normalize_entity_type(record.get("linked_entity_type") or record.get("entity_type")),
        cues=cues,
    )
