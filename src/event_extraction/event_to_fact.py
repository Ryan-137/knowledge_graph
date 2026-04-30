from __future__ import annotations

from pathlib import Path
from typing import Any

from fact_extraction.schema import Evidence, FactCandidate, FactSignal
from fact_extraction.writer import write_fact_jsonl
from kg_core.io import read_json, read_jsonl, write_json

from .ontology_adapter import load_event_type_specs


EVENT_FACT_PROJECTIONS: dict[str, tuple[str, str, str, bool]] = {
    "BirthEvent": ("BORN_IN", "person", "birth_place", False),
    "DeathEvent": ("DIED_IN", "person", "death_place", False),
    "EducationEvent": ("STUDIED_AT", "student", "institution", False),
    "EmploymentEvent": ("WORKED_AT", "employee", "employer", False),
    "PublicationEvent": ("AUTHORED", "author", "work", False),
    "AuthorshipEvent": ("AUTHORED", "author", "work", False),
    "ProposalEvent": ("PROPOSED", "proposer", "concept", False),
    "DesignEvent": ("DESIGNED", "designer", "machine", False),
    "HonorEvent": ("AWARDED", "recipient", "award", False),
    "AwardEvent": ("AWARDED", "recipient", "award", False),
    "CollaborationEvent": ("WORKED_WITH", "person_a", "person_b", True),
    "InfluenceEvent": ("INFLUENCED", "source_person", "target_person", False),
}


def event_candidates_to_fact_candidates(
    verified_events: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = load_event_type_specs(ontology)
    generated: list[dict[str, Any]] = []
    skipped = 0
    for index, event in enumerate(verified_events, start=1):
        if str(event.get("status") or "") != "VERIFIED":
            skipped += 1
            continue
        spec = specs.get(str(event.get("event_type") or ""))
        projection = EVENT_FACT_PROJECTIONS.get(str(event.get("event_type") or ""))
        if spec is None or projection is None:
            skipped += 1
            continue
        role_map = {str(role.get("role") or ""): dict(role) for role in event.get("roles", [])}
        predicate, subject_role, object_role, symmetric = projection
        subject = role_map.get(subject_role)
        obj = role_map.get(object_role)
        if subject is None or obj is None:
            skipped += 1
            continue
        if not str(subject.get("entity_id") or "").strip() or not str(obj.get("entity_id") or "").strip():
            skipped += 1
            continue
        if symmetric:
            subject, obj = sorted([subject, obj], key=lambda role: (str(role.get("entity_id") or ""), str(role.get("role") or "")))
        generated.append(_build_fact_candidate(index, event, predicate, subject, obj))
    summary = {
        "input_event_count": len(verified_events),
        "fact_candidate_count": len(generated),
        "skipped_event_count": skipped,
        "relation_counts": _count_by_key(generated, "predicate"),
    }
    return generated, summary


def event_candidates_to_fact_candidates_from_paths(
    *,
    verified_events_path: str | Path,
    ontology_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    candidates, summary = event_candidates_to_fact_candidates(read_jsonl(verified_events_path), read_json(ontology_path))
    write_fact_jsonl(output_path, candidates)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {"output_path": Path(output_path).as_posix(), "summary_path": Path(output_path).with_suffix(".summary.json").as_posix(), "summary": summary}


def _build_fact_candidate(index: int, event: dict[str, Any], predicate: str, subject: dict[str, Any], obj: dict[str, Any]) -> dict[str, Any]:
    evidence_payload = dict(event.get("evidence") or {})
    evidence = Evidence(
        doc_id=str(evidence_payload.get("doc_id") or ""),
        sentence_id=str(evidence_payload.get("sentence_id") or ""),
        source_id=str(evidence_payload.get("source_id") or ""),
        text=str(evidence_payload.get("text") or ""),
        subject_mention_id=str(subject.get("mention_id") or ""),
        object_mention_id=str(obj.get("mention_id") or ""),
        subject_text=str(subject.get("text") or ""),
        object_text=str(obj.get("text") or ""),
        subject_token_span=list(subject.get("token_span") or []),
        object_token_span=list(obj.get("token_span") or []),
        subject_char_span=list(subject.get("char_span") or []),
        object_char_span=list(obj.get("char_span") or []),
    )
    qualifiers = _event_qualifiers(event)
    event_time = _event_time_details(event)
    signals = [
        FactSignal(
            "event_projection",
            0.0,
            "DERIVED",
            {
                "event_type": event.get("event_type"),
                "event_candidate_id": event.get("event_candidate_id") or event.get("event_id"),
                "source_relation_candidate_ids": list(event.get("source_relation_candidate_ids") or []),
                "event_time": event_time,
            },
        ),
        FactSignal("event_verified", 0.45, "SUPPORTED", {"event_type": event.get("event_type"), "event_time": event_time}),
        FactSignal("event_role_complete", 0.2, "COMPLETE", {"roles": [role.get("role") for role in event.get("roles", [])]}),
        FactSignal(
            "event_trigger_match",
            0.2,
            "MATCHED",
            {
                "trigger": event.get("trigger"),
                "trigger_pattern_id": event.get("trigger_pattern_id"),
                "template_confidence": event.get("template_confidence"),
            },
        ),
    ]
    return FactCandidate(
        fact_candidate_id=f"factcand_event_{index:06d}",
        source_candidate_id=str(event.get("event_candidate_id") or event.get("event_id") or f"eventcand_{index:06d}"),
        subject_id=str(subject.get("entity_id") or ""),
        predicate=predicate,
        object_id=str(obj.get("entity_id") or ""),
        subject_type=str(subject.get("entity_type") or ""),
        object_type=str(obj.get("entity_type") or ""),
        subject_text=str(subject.get("text") or ""),
        object_text=str(obj.get("text") or ""),
        qualifiers=qualifiers,
        evidence=evidence,
        extractor="event_projection",
        signals=signals,
        confidence=0.85,
    ).to_dict()


def _event_qualifiers(event: dict[str, Any]) -> dict[str, Any]:
    qualifiers: dict[str, Any] = {}
    sentence_times = list(event.get("sentence_times") or [])
    time_mentions = list(event.get("time_mentions") or [])
    if sentence_times:
        qualifiers["sentence_times"] = sentence_times
    if time_mentions:
        qualifiers["time_mentions"] = time_mentions
    return qualifiers


def _event_time_details(event: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "time_text": event.get("time_text"),
            "start_time": event.get("start_time_norm"),
            "end_time": event.get("end_time_norm"),
        }.items()
        if value
    }


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
