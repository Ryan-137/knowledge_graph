from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _mention_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("mention_text") or "").strip(),
        str(record.get("mention_type") or "").strip(),
    )


def _collect_top_mentions(
    records: Iterable[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        mention_text, mention_type = _mention_key(record)
        if not mention_text:
            continue
        bucket = grouped.setdefault(
            (mention_text, mention_type),
            {
                "mention_text": mention_text,
                "mention_type": mention_type,
                "count": 0,
                "doc_ids": set(),
                "sample_contexts": [],
                "decision_reasons": Counter(),
            },
        )
        bucket["count"] += 1
        doc_id = str(record.get("doc_id") or "").strip()
        if doc_id:
            bucket["doc_ids"].add(doc_id)
        context_window = str(record.get("context_window") or "").strip()
        if context_window and context_window not in bucket["sample_contexts"] and len(bucket["sample_contexts"]) < 3:
            bucket["sample_contexts"].append(context_window)
        decision_reason = str(record.get("decision_reason") or "").strip()
        if decision_reason:
            bucket["decision_reasons"][decision_reason] += 1
    ranked = sorted(
        grouped.values(),
        key=lambda item: (-int(item["count"]), -len(item["doc_ids"]), item["mention_text"], item["mention_type"]),
    )
    results: list[dict[str, Any]] = []
    for item in ranked[:limit]:
        results.append(
            {
                "mention_text": item["mention_text"],
                "mention_type": item["mention_type"],
                "count": item["count"],
                "doc_count": len(item["doc_ids"]),
                "sample_contexts": list(item["sample_contexts"]),
                "decision_reasons": _sorted_counter(item["decision_reasons"]),
            }
        )
    return results


def build_linking_report(
    linked_mentions: list[dict[str, Any]],
    *,
    evaluation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 linking 阶段统计报告。

    这里默认不依赖人工 gold，因此评测指标字段允许为空。
    后续若 evaluate 阶段产出指标，可通过 evaluation_summary 注入同结构结果。
    """

    decision_counts = Counter(str(item.get("decision") or "UNKNOWN") for item in linked_mentions)
    link_status_counts = Counter(str(item.get("link_status") or item.get("decision") or "UNKNOWN") for item in linked_mentions)
    nil_reason_counts = Counter(
        str(item.get("nil_reason") or "UNKNOWN")
        for item in linked_mentions
        if str(item.get("decision") or "").upper() == "NIL"
    )
    decision_reason_counts = Counter(
        str(item.get("decision_reason") or "UNKNOWN") for item in linked_mentions
    )
    type_counts = Counter(str(item.get("mention_type") or "UNKNOWN") for item in linked_mentions)
    candidate_counts = [len(item.get("candidate_list") or []) for item in linked_mentions]
    linked_records = [item for item in linked_mentions if str(item.get("decision") or "").upper() == "LINKED"]
    review_records = [item for item in linked_mentions if str(item.get("decision") or "").upper() == "REVIEW"]
    nil_records = [item for item in linked_mentions if str(item.get("decision") or "").upper() == "NIL"]
    no_candidate_records = [item for item in nil_records if str(item.get("decision_reason") or "") == "NO_CANDIDATE"]
    skipped_records = [
        item
        for item in linked_mentions
        if str(item.get("decision") or "").upper().startswith("SKIPPED_")
    ]
    short_records = [
        item
        for item in linked_mentions
        if int(item.get("token_end") or 0) - int(item.get("token_start") or 0) <= 1
    ]
    candidate_source_counts = Counter[str]()
    for item in linked_mentions:
        for candidate in item.get("top_candidates") or []:
            for source in candidate.get("candidate_sources") or []:
                candidate_source_counts[str(source)] += 1

    metrics = {
        "candidate_recall_at_1": None,
        "candidate_recall_at_5": None,
        "linking_accuracy": None,
        "nil_precision": None,
        "short_mention_accuracy": None,
    }
    if evaluation_summary:
        metrics.update(
            {
                "candidate_recall_at_1": evaluation_summary.get("candidate_recall_at_1"),
                "candidate_recall_at_5": evaluation_summary.get("candidate_recall_at_5"),
                "linking_accuracy": evaluation_summary.get("linking_accuracy"),
                "nil_precision": evaluation_summary.get("nil_precision"),
                "short_mention_accuracy": evaluation_summary.get("short_mention_accuracy"),
            }
        )

    return {
        "mention_count": len(linked_mentions),
        "decision_counts": _sorted_counter(decision_counts),
        "link_status_counts": _sorted_counter(link_status_counts),
        "nil_reason_counts": _sorted_counter(nil_reason_counts),
        "decision_reason_counts": _sorted_counter(decision_reason_counts),
        "mention_type_counts": _sorted_counter(type_counts),
        "linked_count": len(linked_records),
        "review_count": len(review_records),
        "nil_count": len(nil_records),
        "skipped_count": len(skipped_records),
        "short_mention_count": len(short_records),
        "avg_candidate_count": (
            round(sum(candidate_counts) / len(candidate_counts), 4)
            if candidate_counts
            else 0.0
        ),
        "candidate_source_counts": _sorted_counter(candidate_source_counts),
        "candidate_recall_at_1": metrics["candidate_recall_at_1"],
        "candidate_recall_at_5": metrics["candidate_recall_at_5"],
        "linking_accuracy": metrics["linking_accuracy"],
        "nil_precision": metrics["nil_precision"],
        "short_mention_accuracy": metrics["short_mention_accuracy"],
        "top_no_candidate_mentions": _collect_top_mentions(no_candidate_records),
        "top_nil_mentions": _collect_top_mentions(nil_records),
        "top_review_mentions": _collect_top_mentions(review_records),
    }
