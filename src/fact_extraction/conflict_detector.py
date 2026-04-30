from __future__ import annotations

from collections import defaultdict
from typing import Any


FUNCTIONAL_PREDICATES = {"BORN_IN", "DIED_IN"}


def detect_fact_conflicts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """检测 V1 中最明确的一主体多值冲突，冲突只记录不删除。"""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        predicate = str(fact.get("predicate") or "")
        if predicate in FUNCTIONAL_PREDICATES:
            grouped[(str(fact.get("subject_id") or ""), predicate)].append(fact)

    conflicts: list[dict[str, Any]] = []
    for (subject_id, predicate), rows in sorted(grouped.items()):
        object_ids = sorted({str(row.get("object_id") or "") for row in rows if row.get("object_id")})
        if len(object_ids) <= 1:
            continue
        conflicts.append(
            {
                "conflict_id": f"conflict_{len(conflicts) + 1:06d}",
                "subject_id": subject_id,
                "predicate": predicate,
                "object_ids": object_ids,
                "fact_ids": [row.get("fact_id") for row in rows],
                "reason": "functional_predicate_multi_object",
            }
        )
    return conflicts
