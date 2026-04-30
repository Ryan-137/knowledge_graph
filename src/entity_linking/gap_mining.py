from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json


def _priority_from_counts(*, no_candidate_count: int, nil_count: int, review_count: int, total_count: int) -> str:
    if no_candidate_count >= 3 or total_count >= 8:
        return "P0"
    if nil_count >= 3 or review_count >= 4 or total_count >= 4:
        return "P1"
    return "P2"


def mine_linking_gaps(linked_mentions: list[dict[str, Any]], *, top_n: int = 50) -> list[dict[str, Any]]:
    """从 linking 结果中挖掘高频断链点，生成补库优先级清单。"""

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in linked_mentions:
        decision = str(record.get("decision") or "").upper()
        decision_reason = str(record.get("decision_reason") or "")
        if decision not in {"NIL", "REVIEW"} and decision_reason != "NO_CANDIDATE":
            continue
        mention_text = str(record.get("mention_text") or "").strip()
        mention_type = str(record.get("mention_type") or "").strip()
        if not mention_text:
            continue
        key = (mention_text, mention_type)
        bucket = grouped.setdefault(
            key,
            {
                "mention_text": mention_text,
                "mention_type": mention_type,
                "count": 0,
                "doc_ids": set(),
                "sample_contexts": [],
                "decision_counts": Counter(),
                "decision_reason_counts": Counter(),
            },
        )
        bucket["count"] += 1
        bucket["decision_counts"][decision] += 1
        if decision_reason:
            bucket["decision_reason_counts"][decision_reason] += 1
        doc_id = str(record.get("doc_id") or "").strip()
        if doc_id:
            bucket["doc_ids"].add(doc_id)
        context_window = str(record.get("context_window") or "").strip()
        if context_window and context_window not in bucket["sample_contexts"] and len(bucket["sample_contexts"]) < 3:
            bucket["sample_contexts"].append(context_window)

    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            -int(item["decision_reason_counts"].get("NO_CANDIDATE", 0)),
            -int(item["count"]),
            -len(item["doc_ids"]),
            item["mention_text"],
        ),
    )
    results: list[dict[str, Any]] = []
    for item in ranked[:top_n]:
        no_candidate_count = int(item["decision_reason_counts"].get("NO_CANDIDATE", 0))
        nil_count = int(item["decision_counts"].get("NIL", 0))
        review_count = int(item["decision_counts"].get("REVIEW", 0))
        results.append(
            {
                "mention_text": item["mention_text"],
                "mention_type": item["mention_type"],
                "count": item["count"],
                "doc_count": len(item["doc_ids"]),
                "sample_contexts": list(item["sample_contexts"]),
                "suggested_priority": _priority_from_counts(
                    no_candidate_count=no_candidate_count,
                    nil_count=nil_count,
                    review_count=review_count,
                    total_count=int(item["count"]),
                ),
                "decision_counts": dict(sorted(item["decision_counts"].items(), key=lambda pair: (-pair[1], pair[0]))),
                "decision_reason_counts": dict(
                    sorted(item["decision_reason_counts"].items(), key=lambda pair: (-pair[1], pair[0]))
                ),
            }
        )
    return results


def mine_linking_gaps_from_path(
    *,
    linked_mentions_path: str | Path,
    output_path: str | Path | None = None,
    top_n: int = 50,
) -> list[dict[str, Any]]:
    linked_mentions = read_jsonl(linked_mentions_path)
    results = mine_linking_gaps(linked_mentions, top_n=top_n)
    if output_path is not None:
        write_json(output_path, results)
    return results
