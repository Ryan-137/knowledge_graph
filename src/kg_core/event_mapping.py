from __future__ import annotations

from dataclasses import dataclass

from kg_core.taxonomy import normalize_entity_type


@dataclass(frozen=True, slots=True)
class EventRelationSpec:
    """关系事实事件化时使用的权威映射。"""

    predicate: str
    event_type: str
    subject_role: str
    object_role: str
    subject_type: str
    object_type: str


EVENT_RELATION_SPECS: dict[str, EventRelationSpec] = {
    "BORN_IN": EventRelationSpec("BORN_IN", "BirthEvent", "person", "birth_place", "PERSON", "PLACE"),
    "DIED_IN": EventRelationSpec("DIED_IN", "DeathEvent", "person", "death_place", "PERSON", "PLACE"),
    "STUDIED_AT": EventRelationSpec("STUDIED_AT", "EducationEvent", "student", "institution", "PERSON", "ORGANIZATION"),
    "WORKED_AT": EventRelationSpec("WORKED_AT", "EmploymentEvent", "employee", "employer", "PERSON", "ORGANIZATION"),
    "WORKED_WITH": EventRelationSpec("WORKED_WITH", "CollaborationEvent", "person_a", "person_b", "PERSON", "PERSON"),
    "AUTHORED": EventRelationSpec("AUTHORED", "PublicationEvent", "author", "work", "PERSON", "WORK"),
    "PROPOSED": EventRelationSpec("PROPOSED", "ProposalEvent", "proposer", "concept", "PERSON", "CONCEPT"),
    "DESIGNED": EventRelationSpec("DESIGNED", "DesignEvent", "designer", "machine", "PERSON", "MACHINE"),
    "AWARDED": EventRelationSpec("AWARDED", "HonorEvent", "recipient", "award", "PERSON", "AWARD"),
    "INFLUENCED": EventRelationSpec("INFLUENCED", "InfluenceEvent", "source_person", "target_person", "PERSON", "PERSON"),
}


def resolve_event_relation_spec(
    predicate: str,
    *,
    subject_type: str | None = None,
    object_type: str | None = None,
) -> EventRelationSpec | None:
    """根据本体语义解析关系对应的事件类型和角色。"""

    normalized_predicate = str(predicate or "").strip().upper()
    normalized_subject_type = normalize_entity_type(subject_type)
    normalized_object_type = normalize_entity_type(object_type)
    if normalized_predicate == "PROPOSED" and normalized_object_type == "MACHINE":
        normalized_predicate = "DESIGNED"
    spec = EVENT_RELATION_SPECS.get(normalized_predicate)
    if spec is None:
        return None
    if normalized_subject_type != spec.subject_type or normalized_object_type != spec.object_type:
        return None
    return spec


def event_type_for_predicate(predicate: str) -> str | None:
    spec = EVENT_RELATION_SPECS.get(str(predicate or "").strip().upper())
    return spec.event_type if spec else None
