from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json, write_jsonl

from .argument_builder import mentions_by_sentence, relation_candidate_ids_by_sentence
from .trigger_detector import extract_event_candidates


def generate_text_event_candidates(
    sentences: list[dict[str, Any]],
    resolved_mentions: list[dict[str, Any]],
    relation_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    mention_index = mentions_by_sentence(resolved_mentions)
    relation_index = relation_candidate_ids_by_sentence(relation_candidates)
    relations_by_id = _relations_by_id(relation_candidates)
    events: list[dict[str, Any]] = []
    arguments: list[dict[str, Any]] = []
    for sentence in sentences:
        sentence_id = str(sentence.get("sentence_id") or "")
        for event in extract_event_candidates(sentence, entity_mentions=mention_index.get(sentence_id, [])):
            event_entity_ids = _event_role_entity_ids(event)
            event["source_relation_candidate_ids"] = [
                candidate_id
                for candidate_id in relation_index.get(sentence_id, [])
                if _has_two_entity_overlap(event_entity_ids, _relation_entity_ids(relations_by_id.get(candidate_id, {})))
            ]
            sentence_times = list(sentence.get("normalized_time") or sentence.get("sentence_times") or [])
            time_mentions = list(sentence.get("time_mentions") or [])
            if sentence_times:
                event["sentence_times"] = sentence_times
            if time_mentions:
                event["time_mentions"] = time_mentions
            events.append(event)
    arguments = build_event_arguments(events)
    summary = {
        "sentence_count": len(sentences),
        "event_candidate_count": len(events),
        "event_argument_count": len(arguments),
        "event_type_counts": _count_by_key(events, "event_type"),
    }
    return events, arguments, summary


def build_event_arguments(events: list[dict[str, Any]], *, argument_source: str = "event_candidate") -> list[dict[str, Any]]:
    arguments: list[dict[str, Any]] = []
    for event in events:
        for role in event.get("roles", []):
            arguments.append(
                {
                    "event_candidate_id": event["event_candidate_id"],
                    "event_status": str(event.get("status") or "CANDIDATE"),
                    "argument_source": argument_source,
                    **dict(role),
                }
            )
    return arguments


def generate_text_event_candidates_from_paths(
    *,
    sentences_path: str | Path,
    resolved_mentions_path: str | Path,
    relation_candidates_path: str | Path,
    event_candidates_output_path: str | Path,
    event_arguments_output_path: str | Path,
) -> dict[str, Any]:
    events, arguments, summary = generate_text_event_candidates(
        read_jsonl(sentences_path),
        read_jsonl(resolved_mentions_path),
        read_jsonl(relation_candidates_path),
    )
    write_jsonl(event_candidates_output_path, events)
    write_jsonl(event_arguments_output_path, arguments)
    write_json(Path(event_candidates_output_path).with_suffix(".summary.json"), summary)
    return {
        "event_candidates_output_path": Path(event_candidates_output_path).as_posix(),
        "event_arguments_output_path": Path(event_arguments_output_path).as_posix(),
        "summary": summary,
    }


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _relations_by_id(relation_candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for candidate in relation_candidates:
        candidate_id = str(candidate.get("candidate_id") or candidate.get("claim_candidate_id") or "")
        if candidate_id:
            indexed[candidate_id] = candidate
    return indexed


def _event_role_entity_ids(event: dict[str, Any]) -> set[str]:
    return {
        entity_id
        for role in event.get("roles", [])
        if (entity_id := str(role.get("entity_id") or "").strip())
    }


def _relation_entity_ids(candidate: dict[str, Any]) -> set[str]:
    keys = ("subject_entity_id", "object_entity_id", "subject_id", "object_id")
    return {entity_id for key in keys if (entity_id := str(candidate.get(key) or "").strip())}


def _has_two_entity_overlap(event_entity_ids: set[str], relation_entity_ids: set[str]) -> bool:
    return len(event_entity_ids & relation_entity_ids) >= 2
