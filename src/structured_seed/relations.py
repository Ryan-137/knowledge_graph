from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .queries import (
    build_authored_query,
    build_awarded_query,
    build_born_in_query,
    build_designed_query,
    build_died_in_query,
    build_located_in_query,
    build_proposed_query,
    build_studied_at_query,
    build_worked_at_query,
)


@dataclass(slots=True)
class RelationTemplate:
    """单关系、单模板的抓取定义。"""

    name: str
    subject_types: tuple[str, ...]
    event_type: str | None
    build_query: Callable[[list[str], int, int], str]


def build_relation_templates() -> list[RelationTemplate]:
    """定义所有允许的结构化抓取关系模板。"""
    return [
        RelationTemplate(
            name="BORN_IN",
            subject_types=("Person",),
            event_type="BirthEvent",
            build_query=build_born_in_query,
        ),
        RelationTemplate(
            name="DIED_IN",
            subject_types=("Person",),
            event_type="DeathEvent",
            build_query=build_died_in_query,
        ),
        RelationTemplate(
            name="STUDIED_AT",
            subject_types=("Person",),
            event_type="EducationEvent",
            build_query=build_studied_at_query,
        ),
        RelationTemplate(
            name="WORKED_AT",
            subject_types=("Person",),
            event_type="EmploymentEvent",
            build_query=build_worked_at_query,
        ),
        RelationTemplate(
            name="AUTHORED",
            subject_types=("Person",),
            event_type="PublicationEvent",
            build_query=build_authored_query,
        ),
        RelationTemplate(
            name="PROPOSED",
            subject_types=("Person",),
            event_type="ProposalEvent",
            build_query=build_proposed_query,
        ),
        RelationTemplate(
            name="DESIGNED",
            subject_types=("Person",),
            event_type="DesignEvent",
            build_query=build_designed_query,
        ),
        RelationTemplate(
            name="AWARDED",
            subject_types=("Person",),
            event_type="HonorEvent",
            build_query=build_awarded_query,
        ),
        RelationTemplate(
            name="LOCATED_IN",
            subject_types=("Organization",),
            event_type=None,
            build_query=build_located_in_query,
        ),
    ]
