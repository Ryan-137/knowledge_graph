from __future__ import annotations

from typing import Any

from event_extraction.event_to_fact import event_candidates_to_fact_candidates


_EVENT_MAPPING_ONTOLOGY = {
    "event_types": [
        {
            "name": "CollaborationEvent",
            "participants": [
                {"role": "person_a", "class": "Person"},
                {"role": "person_b", "class": "Person"},
            ],
            "derived_relation": "WORKED_WITH",
        },
        {
            "name": "InfluenceEvent",
            "participants": [
                {"role": "source_person", "class": "Person"},
                {"role": "target_person", "class": "Person"},
            ],
            "derived_relation": "INFLUENCED",
        },
    ],
    "constraints": [],
}


def event_to_fact(event: dict[str, Any]) -> dict[str, Any]:
    """事实层旧入口统一复用事件层投影，避免两套 role map 漂移。"""

    normalized_event = dict(event)
    normalized_event["status"] = str(normalized_event.get("status") or "VERIFIED")
    candidates, _summary = event_candidates_to_fact_candidates([normalized_event], _EVENT_MAPPING_ONTOLOGY)
    if not candidates:
        raise ValueError(f"event cannot be projected to fact: {event.get('event_type')}")
    return candidates[0]


__all__ = ["event_to_fact"]
