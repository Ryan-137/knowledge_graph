from __future__ import annotations

from collections import Counter
from typing import Any


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_coreference_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = Counter(str(record.get("decision") or "UNKNOWN") for record in records)
    original_decision_counts = Counter(str(record.get("original_decision") or record.get("decision") or "UNKNOWN") for record in records)
    reason_counts = Counter(
        str(record.get("coref_reason") or record.get("decision_reason") or "UNKNOWN")
        for record in records
        if str(record.get("resolution_stage") or "") == "COREFERENCE"
    )
    antecedent_type_counts = Counter(
        str(record.get("linked_entity_type") or "UNKNOWN")
        for record in records
        if str(record.get("decision") or "") == "LINKED_BY_COREF"
    )
    distance_counts = Counter(
        str(record.get("antecedent_distance_sentences"))
        for record in records
        if str(record.get("decision") or "") == "LINKED_BY_COREF"
    )
    return {
        "mention_count": len(records),
        "decision_counts": _sorted_counter(decision_counts),
        "original_decision_counts": _sorted_counter(original_decision_counts),
        "linked_by_coref_count": decision_counts["LINKED_BY_COREF"],
        "coref_unresolved_count": decision_counts["COREF_UNRESOLVED"],
        "coref_reason_counts": _sorted_counter(reason_counts),
        "antecedent_type_counts": _sorted_counter(antecedent_type_counts),
        "antecedent_distance_sentence_counts": _sorted_counter(distance_counts),
    }
