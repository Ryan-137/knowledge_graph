from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

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


def build_event_candidate_from_claim(claim_row: sqlite3.Row) -> dict[str, Any] | None:
    """从 claims 派生事件候选，供后续事件抽取/融合使用。"""
    event_type_map = {
        "BORN_IN": "BirthEvent",
        "DIED_IN": "DeathEvent",
        "STUDIED_AT": "EducationEvent",
        "WORKED_AT": "EmploymentEvent",
        "AUTHORED": "PublicationEvent",
        "PROPOSED": "ProposalEvent",
        "DESIGNED": "DesignEvent",
        "AWARDED": "HonorEvent",
    }
    predicate = claim_row["predicate"]
    if predicate not in event_type_map:
        return None
    qualifiers = json.loads(claim_row["qualifiers_json"])
    start_time_raw = qualifiers.get("start_time") or qualifiers.get("publication_date") or qualifiers.get("point_in_time")
    end_time_raw = qualifiers.get("end_time")
    time_text = " / ".join(value for value in [start_time_raw, end_time_raw] if value) or None
    location_id = claim_row["object_id"] if predicate in {"BORN_IN", "DIED_IN"} else None
    event_candidate_id = build_claim_id(
        claim_row["subject_id"],
        event_type_map[predicate],
        claim_row["object_id"],
        claim_row["statement_id"],
    )
    return {
        "event_candidate_id": event_candidate_id,
        "event_type": event_type_map[predicate],
        "subject_id": claim_row["subject_id"],
        "object_id": claim_row["object_id"],
        "start_time_raw": start_time_raw,
        "end_time_raw": end_time_raw,
        "start_time_norm": normalize_wikidata_time(start_time_raw),
        "end_time_norm": normalize_wikidata_time(end_time_raw),
        "location_id": location_id,
        "time_text": time_text,
        "source_name": claim_row["source_name"],
        "statement_id": claim_row["statement_id"],
        "predicate": predicate,
        "raw_payload_json": {
            "claim_id": claim_row["claim_id"],
            "qualifiers": qualifiers,
        },
    }


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
