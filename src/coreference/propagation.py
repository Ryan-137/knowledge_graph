from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json, write_jsonl
from kg_core.mention_filters import is_generic_mention, is_pronoun_mention

from .anchor_builder import CoreferenceAnchor, build_anchor
from .evaluation import build_coreference_report
from .rule_resolver import resolve_by_rules


def _sentence_index(sentences: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(record.get("sentence_id") or "").strip(): int(record.get("sentence_index_in_doc") or 0)
        for record in sentences
        if record.get("sentence_id")
    }


def _sort_key(record: dict[str, Any], sentence_index_by_id: dict[str, int]) -> tuple[str, int, int, str]:
    sentence_id = str(record.get("sentence_id") or "").strip()
    return (
        str(record.get("doc_id") or "").strip(),
        int(sentence_index_by_id.get(sentence_id, 0)),
        int(record.get("token_start") or 0),
        str(record.get("mention_id") or ""),
    )


def _base_record(record: dict[str, Any]) -> dict[str, Any]:
    decision = str(record.get("decision") or "").upper()
    resolved = dict(record)
    resolved["original_decision"] = decision
    resolved["original_nil_reason"] = record.get("nil_reason")
    return resolved


def _is_coreference_target(record: dict[str, Any]) -> bool:
    decision = str(record.get("decision") or "").upper()
    mention_text = str(record.get("mention_text") or record.get("text") or "")
    return decision in {"SKIPPED_PRONOUN", "SKIPPED_GENERIC"} and (
        is_pronoun_mention(mention_text) or is_generic_mention(mention_text)
    )


def _linked_by_coref_record(
    record: dict[str, Any],
    anchor: CoreferenceAnchor,
    sentence_index_by_id: dict[str, int],
    reason: str,
) -> dict[str, Any]:
    resolved = _base_record(record)
    sentence_id = str(record.get("sentence_id") or "").strip()
    distance = int(sentence_index_by_id.get(sentence_id, 0)) - anchor.sentence_index_in_doc
    resolved.update(
        {
            "decision": "LINKED_BY_COREF",
            "link_status": "LINKED_BY_COREF",
            "entity_id": anchor.entity_id,
            "canonical_name": anchor.canonical_name,
            "linked_entity_type": anchor.linked_entity_type,
            "nil_reason": None,
            "decision_reason": reason,
            "resolution_stage": "COREFERENCE",
            "mention_resolution": "linked_by_coref",
            "coref_source": "RULE",
            "coref_reason": reason,
            "antecedent_mention_id": anchor.mention_id,
            "antecedent_entity_id": anchor.entity_id,
            "antecedent_canonical_name": anchor.canonical_name,
            "antecedent_sentence_id": anchor.sentence_id,
            "antecedent_distance_sentences": distance,
        }
    )
    return resolved


def _coref_unresolved_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
    resolved = _base_record(record)
    resolved.update(
        {
            "decision": "COREF_UNRESOLVED",
            "link_status": "COREF_UNRESOLVED",
            "entity_id": None,
            "canonical_name": None,
            "linked_entity_type": None,
            "nil_reason": "COREF_UNRESOLVED",
            "decision_reason": reason,
            "resolution_stage": "COREFERENCE",
            "mention_resolution": "coref_unresolved",
            "coref_source": "RULE",
            "coref_reason": reason,
        }
    )
    return resolved


def _preserve_record(record: dict[str, Any]) -> dict[str, Any]:
    resolved = _base_record(record)
    decision = str(resolved.get("decision") or "").upper()
    if decision == "LINKED" and resolved.get("entity_id"):
        resolved["mention_resolution"] = "linked"
    elif not resolved.get("mention_resolution"):
        resolved["mention_resolution"] = "nil"
    return resolved


def resolve_coreferences(
    linked_mentions: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    *,
    max_sentence_distance: int = 3,
) -> list[dict[str, Any]]:
    sentence_index_by_id = _sentence_index(sentences)
    anchors_by_doc: dict[str, list[CoreferenceAnchor]] = {}
    resolved_records: list[dict[str, Any]] = []

    for record in sorted(linked_mentions, key=lambda item: _sort_key(item, sentence_index_by_id)):
        doc_id = str(record.get("doc_id") or "").strip()
        doc_anchors = anchors_by_doc.setdefault(doc_id, [])
        if _is_coreference_target(record):
            resolution = resolve_by_rules(
                record,
                doc_anchors,
                sentence_index_by_id,
                max_sentence_distance=max_sentence_distance,
            )
            if resolution.anchor is None:
                resolved_records.append(_coref_unresolved_record(record, resolution.reason))
            else:
                resolved_records.append(_linked_by_coref_record(record, resolution.anchor, sentence_index_by_id, resolution.reason))
            continue

        resolved_record = _preserve_record(record)
        resolved_records.append(resolved_record)
        anchor = build_anchor(resolved_record, sentence_index_by_id)
        if anchor is not None:
            doc_anchors.append(anchor)

    resolved_records.sort(key=lambda item: _sort_key(item, sentence_index_by_id))
    return resolved_records


def resolve_coreferences_from_paths(
    *,
    linked_mentions_path: str | Path,
    sentences_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    unresolved_output_path: str | Path | None = None,
    max_sentence_distance: int = 3,
) -> list[dict[str, Any]]:
    linked_mentions = read_jsonl(linked_mentions_path)
    sentences = read_jsonl(sentences_path)
    resolved_mentions = resolve_coreferences(
        linked_mentions,
        sentences,
        max_sentence_distance=max_sentence_distance,
    )
    write_jsonl(output_path, resolved_mentions)
    if report_path is not None:
        write_json(report_path, build_coreference_report(resolved_mentions))
    if unresolved_output_path is not None:
        write_jsonl(
            unresolved_output_path,
            [record for record in resolved_mentions if str(record.get("decision") or "") == "COREF_UNRESOLVED"],
        )
    return resolved_mentions
