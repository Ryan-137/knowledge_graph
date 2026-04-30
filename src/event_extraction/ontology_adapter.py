from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_core.taxonomy import normalize_entity_type


@dataclass(frozen=True)
class EventRoleSpec:
    role: str
    entity_type: str


@dataclass(frozen=True)
class EventTypeSpec:
    event_type: str
    roles: list[EventRoleSpec]
    required_roles: set[str]
    derived_relation: str

    def role_type(self, role_name: str) -> str:
        for role in self.roles:
            if role.role == role_name:
                return role.entity_type
        return ""


def load_event_type_specs(ontology: dict[str, Any]) -> dict[str, EventTypeSpec]:
    required_roles = {
        str(item.get("event_type")): {str(role) for role in item.get("roles", [])}
        for item in ontology.get("constraints", [])
        if item.get("type") == "event_role_required"
    }
    specs: dict[str, EventTypeSpec] = {}
    for event_type in ontology.get("event_types", []):
        name = str(event_type.get("name") or "")
        roles = [
            EventRoleSpec(
                role=str(role.get("role") or ""),
                entity_type=normalize_entity_type(role.get("class")),
            )
            for role in event_type.get("participants", [])
        ]
        specs[name] = EventTypeSpec(
            event_type=name,
            roles=roles,
            required_roles=required_roles.get(name, {role.role for role in roles}),
            derived_relation=str(event_type.get("derived_relation") or ""),
        )
    return specs


def event_type_by_relation(ontology: dict[str, Any]) -> dict[str, str]:
    return {
        str(relation.get("name") or ""): str(relation.get("event_type") or "")
        for relation in ontology.get("relations", [])
        if relation.get("event_type")
    }
