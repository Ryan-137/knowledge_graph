from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_json, read_jsonl, write_json, write_jsonl
from kg_core.taxonomy import normalize_entity_type

from .ontology_adapter import load_event_type_specs


def verify_text_events(
    event_candidates: list[dict[str, Any]],
    ontology: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = load_event_type_specs(ontology)
    verified: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    missing_entity_role_counts: dict[str, int] = {}
    missing_entity_mentions: list[dict[str, Any]] = []
    review_event_count_by_reason: dict[str, int] = {}
    for candidate in event_candidates:
        record = dict(candidate)
        spec = specs.get(str(record.get("event_type") or ""))
        role_map = {str(role.get("role") or ""): role for role in record.get("roles", [])}
        missing_roles = sorted((spec.required_roles if spec else set()) - set(role_map))
        empty_entity_roles = _empty_entity_roles(role_map)
        type_errors = _type_errors(role_map, spec)
        incoming_review_reasons = _compact_review_reasons(record.get("review_reasons"))
        if missing_roles or empty_entity_roles or type_errors or spec is None or incoming_review_reasons:
            record["status"] = "REVIEW"
            record["confidence"] = min(float(record.get("confidence", 0.0) or 0.0), 0.49)
            review_reasons = {
                **incoming_review_reasons,
                "missing_roles": missing_roles,
                "empty_entity_roles": empty_entity_roles,
                "type_errors": type_errors,
                "unknown_event_type": spec is None,
            }
            record["review_reasons"] = _compact_review_reasons(review_reasons)
            record["review_info"] = _review_info(record, role_map)
            _add_review_reason_counts(review_event_count_by_reason, record["review_reasons"])
            for role_name in empty_entity_roles:
                missing_entity_role_counts[role_name] = missing_entity_role_counts.get(role_name, 0) + 1
                missing_entity_mentions.append(_missing_entity_mention(record, role_map.get(role_name, {}), role_name))
        else:
            record["status"] = "VERIFIED"
            record["confidence"] = max(float(record.get("confidence", 0.0) or 0.0), 0.85)
            signals = list(record.get("signals") or [])
            signals.append({"name": "event_verified", "score": 0.45, "label": "SUPPORTED", "details": {}})
            signals.append({"name": "event_role_complete", "score": 0.2, "label": "COMPLETE", "details": {"roles": sorted(role_map)}})
            record["signals"] = signals
        status = str(record.get("status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        verified.append(record)
    return verified, {
        "input_event_count": len(event_candidates),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_entity_role_counts": dict(sorted(missing_entity_role_counts.items())),
        "missing_entity_mentions": missing_entity_mentions,
        "review_event_count_by_reason": dict(sorted(review_event_count_by_reason.items())),
    }


def verify_text_events_from_paths(
    *,
    event_candidates_path: str | Path,
    ontology_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    verified, summary = verify_text_events(read_jsonl(event_candidates_path), read_json(ontology_path))
    write_jsonl(output_path, verified)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {"output_path": Path(output_path).as_posix(), "summary": summary}


def _type_errors(role_map: dict[str, dict[str, Any]], spec: Any) -> list[dict[str, str]]:
    if spec is None:
        return []
    errors: list[dict[str, str]] = []
    for role_name, role in role_map.items():
        expected = spec.role_type(role_name)
        actual = normalize_entity_type(role.get("entity_type"))
        if expected and actual != expected:
            errors.append({"role": role_name, "expected": expected, "actual": actual})
    return errors


def _empty_entity_roles(role_map: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(role_name for role_name, role in role_map.items() if not str(role.get("entity_id") or "").strip())


def _compact_review_reasons(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, False, [], {}, ""):
            continue
        compact[str(key)] = value
    return compact


def _add_review_reason_counts(counts: dict[str, int], reasons: dict[str, Any]) -> None:
    for key, value in reasons.items():
        if isinstance(value, list):
            if not value:
                continue
            for item in value:
                reason_key = f"{key}:{item}" if isinstance(item, str) else key
                counts[reason_key] = counts.get(reason_key, 0) + 1
        else:
            counts[key] = counts.get(key, 0) + 1


def _review_info(record: dict[str, Any], role_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = dict(record.get("evidence") or {})
    return {
        "event_candidate_id": str(record.get("event_candidate_id") or ""),
        "event_type": str(record.get("event_type") or ""),
        "trigger": str(record.get("trigger") or ""),
        "doc_id": str(evidence.get("doc_id") or ""),
        "sentence_id": str(evidence.get("sentence_id") or ""),
        "source_id": str(evidence.get("source_id") or ""),
        "sentence_text": str(evidence.get("text") or ""),
        "roles": [
            {
                "role": role_name,
                "text": str(role.get("text") or ""),
                "mention_id": str(role.get("mention_id") or ""),
                "entity_id": str(role.get("entity_id") or ""),
            }
            for role_name, role in sorted(role_map.items())
        ],
    }


def _missing_entity_mention(record: dict[str, Any], role: dict[str, Any], role_name: str) -> dict[str, Any]:
    evidence = dict(record.get("evidence") or {})
    return {
        "event_candidate_id": str(record.get("event_candidate_id") or ""),
        "event_type": str(record.get("event_type") or ""),
        "role": role_name,
        "mention_text": str(role.get("text") or ""),
        "mention_id": str(role.get("mention_id") or ""),
        "doc_id": str(evidence.get("doc_id") or ""),
        "sentence_id": str(evidence.get("sentence_id") or ""),
        "source_id": str(evidence.get("source_id") or ""),
        "trigger": str(record.get("trigger") or ""),
    }
