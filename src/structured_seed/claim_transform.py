from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

from kg_core.event_mapping import EventRelationSpec, resolve_event_relation_spec

from .entity_transform import extract_qid, get_binding_value, simplify_binding_row


def build_claim_record(row: dict[str, Any], predicate: str, confidence: float) -> dict[str, Any]:
    """把 Wikidata statement 查询结果转换为 claims 表记录。"""
    subject_id = extract_qid(get_binding_value(row, "subject"))
    object_id = extract_qid(get_binding_value(row, "object"))
    statement_id = get_binding_value(row, "statement") or f"{subject_id}-{predicate}-{object_id}"
    qualifiers = {
        "start_time": get_binding_value(row, "startTime"),
        "end_time": get_binding_value(row, "endTime"),
        "point_in_time": get_binding_value(row, "pointInTime"),
        "publication_date": get_binding_value(row, "publicationDate"),
    }
    qualifiers = {key: value for key, value in qualifiers.items() if value}
    claim_id = build_claim_id(subject_id, predicate, object_id, statement_id)
    return {
        "claim_id": claim_id,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "object_text": get_binding_value(row, "objectLabel"),
        "statement_id": statement_id,
        "qualifiers_json": qualifiers,
        "source_name": "wikidata_sparql",
        "source_record_id": statement_id,
        "retrieved_at": utc_now_text(),
        "confidence": confidence,
        "raw_payload_json": simplify_binding_row(row),
    }


def build_event_candidate_from_claim(
    claim_row: sqlite3.Row,
    *,
    entity_index: dict[str, sqlite3.Row],
) -> dict[str, Any] | None:
    """从 claims 派生事件候选，供后续事件抽取/融合使用。"""
    predicate = claim_row["predicate"]
    subject_id = claim_row["subject_id"]
    object_id = claim_row["object_id"]
    subject_type = _entity_type(entity_index, subject_id)
    object_type = _entity_type(entity_index, object_id)
    event_spec = resolve_event_relation_spec(predicate, subject_type=subject_type, object_type=object_type)
    if event_spec is None:
        return None
    qualifiers = json.loads(claim_row["qualifiers_json"])
    start_time_raw = (
        qualifiers.get("start_time")
        or qualifiers.get("publication_date")
        or qualifiers.get("point_in_time")
        or _subject_lifecycle_time(entity_index, subject_id, event_spec)
    )
    end_time_raw = qualifiers.get("end_time")
    time_text = " / ".join(value for value in [start_time_raw, end_time_raw] if value) or None
    location_id = object_id if event_spec.object_role in {"birth_place", "death_place"} else None
    event_candidate_id = build_claim_id(
        subject_id,
        event_spec.event_type,
        object_id,
        claim_row["statement_id"],
    )
    roles = build_event_roles(event_spec, subject_id, object_id)
    return {
        "event_id": event_candidate_id,
        "event_candidate_id": event_candidate_id,
        "event_type": event_spec.event_type,
        "subject_id": subject_id,
        "object_id": object_id,
        "start_time_raw": start_time_raw,
        "end_time_raw": end_time_raw,
        "start_time_norm": normalize_wikidata_time(start_time_raw),
        "end_time_norm": normalize_wikidata_time(end_time_raw),
        "location_id": location_id,
        "time_text": time_text,
        "source_name": claim_row["source_name"],
        "statement_id": claim_row["statement_id"],
        "predicate": event_spec.predicate,
        "roles_json": roles,
        "confidence": claim_row["confidence"],
        "raw_payload_json": {
            "claim_id": claim_row["claim_id"],
            "qualifiers": qualifiers,
            "source_predicate": predicate,
        },
    }


def build_event_roles(event_spec: EventRelationSpec, subject_id: str, object_id: str) -> list[dict[str, str]]:
    return [
        {"role": event_spec.subject_role, "entity_id": subject_id, "entity_type": event_spec.subject_type},
        {"role": event_spec.object_role, "entity_id": object_id, "entity_type": event_spec.object_type},
    ]


def _entity_type(entity_index: dict[str, sqlite3.Row], entity_id: str | None) -> str | None:
    if not entity_id or entity_id not in entity_index:
        return None
    return entity_index[entity_id]["entity_type"]


def _entity_raw_payload(entity_index: dict[str, sqlite3.Row], entity_id: str | None) -> dict[str, Any]:
    if not entity_id or entity_id not in entity_index:
        return {}
    return json.loads(entity_index[entity_id]["raw_payload_json"])


def _subject_lifecycle_time(
    entity_index: dict[str, sqlite3.Row],
    subject_id: str | None,
    event_spec: EventRelationSpec,
) -> str | None:
    raw_payload = _entity_raw_payload(entity_index, subject_id)
    if event_spec.event_type == "BirthEvent":
        return raw_payload.get("birth_date_raw")
    if event_spec.event_type == "DeathEvent":
        return raw_payload.get("death_date_raw")
    return None


def normalize_wikidata_time(time_text: str | None) -> str | None:
    if not time_text:
        return None
    match = re.match(r"^[+-]?(\d{4})-(\d{2})-(\d{2})T", time_text)
    if match:
        year, month, day = match.groups()
        if month == "00" or day == "00":
            return year
        return f"{year}-{month}-{day}"
    match = re.match(r"^[+-]?(\d{4})", time_text)
    if match:
        return match.group(1)
    return None


def build_claim_id(subject_id: str | None, predicate: str, object_id: str | None, statement_id: str) -> str:
    payload = f"{subject_id}|{predicate}|{object_id}|{statement_id}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_job_key(job_name: str, params: dict[str, Any], limit: int | None, offset: int | None) -> str:
    payload = json.dumps({"job_name": job_name, "params": params, "limit": limit, "offset": offset}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
