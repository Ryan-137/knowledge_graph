from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json

from .conflict_detector import detect_fact_conflicts
from .schema import VerifiedFact
from .writer import write_fact_jsonl


def _qualifier_signature(qualifiers: dict[str, Any]) -> str:
    return json.dumps(qualifiers or {}, ensure_ascii=False, sort_keys=True)


def _fact_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("subject_id") or ""),
        str(record.get("predicate") or ""),
        str(record.get("object_id") or ""),
        _qualifier_signature(dict(record.get("qualifiers") or {})),
    )


def _score_record(record: dict[str, Any]) -> float:
    return sum(float(signal.get("score", 0.0) or 0.0) for signal in record.get("signals", []))


def _aggregate_signal_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    signal_counts: dict[str, int] = {}
    signal_score_sums: dict[str, float] = {}
    for record in records:
        for signal in record.get("signals", []):
            name = str(signal.get("name") or "")
            signal_counts[name] = signal_counts.get(name, 0) + 1
            signal_score_sums[name] = signal_score_sums.get(name, 0.0) + float(signal.get("score", 0.0) or 0.0)
    return {
        "signal_counts": dict(sorted(signal_counts.items())),
        "signal_score_sums": {key: round(value, 6) for key, value in sorted(signal_score_sums.items())},
        "source_candidate_ids": [record.get("source_candidate_id") for record in records],
    }


def aggregate_fact_candidates(
    verified_candidates: list[dict[str, Any]],
    *,
    final_threshold: float = 0.75,
    review_threshold: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    usable = [
        record
        for record in verified_candidates
        if str(record.get("status") or "") != "REJECTED" and _score_record(record) >= review_threshold
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in usable:
        grouped[_fact_key(record)].append(record)

    verified_facts: list[dict[str, Any]] = []
    final_facts: list[dict[str, Any]] = []
    for running_index, (_, rows) in enumerate(sorted(grouped.items()), start=1):
        best_score = max(_score_record(row) for row in rows)
        evidence_bonus = min(0.10, max(0, len(rows) - 1) * 0.05)
        confidence = min(1.0, best_score + evidence_bonus)
        status = "FINAL" if confidence >= final_threshold else "REVIEW"
        prototype = rows[0]
        fact = VerifiedFact(
            fact_id=f"fact_{running_index:06d}",
            subject_id=str(prototype.get("subject_id") or ""),
            predicate=str(prototype.get("predicate") or ""),
            object_id=str(prototype.get("object_id") or ""),
            qualifiers=dict(prototype.get("qualifiers") or {}),
            evidence=[dict(row.get("evidence") or {}) for row in rows],
            confidence=confidence,
            extractor="pattern+ds+llm_verify",
            signals={
                **_aggregate_signal_summary(rows),
                "evidence_count": len(rows),
                "evidence_bonus": round(evidence_bonus, 6),
            },
            status=status,
        ).to_dict()
        verified_facts.append(fact)
        if status == "FINAL":
            final_facts.append(fact)

    conflicts = detect_fact_conflicts(final_facts)
    summary = {
        "input_candidate_count": len(verified_candidates),
        "usable_candidate_count": len(usable),
        "verified_fact_count": len(verified_facts),
        "final_fact_count": len(final_facts),
        "conflict_count": len(conflicts),
        "relation_counts": _count_by_key(final_facts, "predicate"),
    }
    return verified_facts, final_facts, conflicts, summary


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def aggregate_fact_candidates_from_paths(
    *,
    verified_candidates_path: str | Path,
    verified_facts_output_path: str | Path,
    final_facts_output_path: str | Path,
    conflicts_output_path: str | Path,
) -> dict[str, Any]:
    verified_facts, final_facts, conflicts, summary = aggregate_fact_candidates(read_jsonl(verified_candidates_path))
    if Path(verified_facts_output_path).resolve() != Path(verified_candidates_path).resolve():
        write_fact_jsonl(verified_facts_output_path, verified_facts)
    write_fact_jsonl(final_facts_output_path, final_facts)
    write_fact_jsonl(conflicts_output_path, conflicts)
    write_json(Path(final_facts_output_path).with_suffix(".summary.json"), summary)
    return {
        "verified_facts_output_path": Path(verified_facts_output_path).as_posix(),
        "final_facts_output_path": Path(final_facts_output_path).as_posix(),
        "conflicts_output_path": Path(conflicts_output_path).as_posix(),
        "summary_path": Path(final_facts_output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }
