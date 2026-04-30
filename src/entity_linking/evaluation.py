from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json


def _normalize_gold_entity_id(record: dict[str, Any]) -> str | None:
    raw_value = record.get("gold_entity_id", record.get("entity_id"))
    value = str(raw_value or "").strip()
    if not value or value.upper() == "NIL":
        return None
    return value


def _prediction_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(record.get("mention_id") or "").strip(),
        str(record.get("doc_id") or "").strip(),
        str(record.get("sentence_id") or "").strip(),
        int(record.get("token_start") or 0),
        int(record.get("token_end") or 0),
    )


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def evaluate_linking_predictions(
    predictions: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """基于小规模 dev gold 评测 linking 效果。

    评测标签规则：
    - gold_entity_id 为空或为 NIL 视为金标未链接
    - prediction 只有 decision=LINKED 且 entity_id 非空时视为成功链接
    - REVIEW / NIL / SKIPPED_* 一律按未链接处理
    """

    prediction_index: dict[tuple[str, str, str, int, int], dict[str, Any]] = {
        _prediction_key(record): record for record in predictions
    }
    fallback_prediction_index: dict[str, dict[str, Any]] = {
        str(record.get("mention_id") or "").strip(): record
        for record in predictions
        if str(record.get("mention_id") or "").strip()
    }

    total_count = 0
    gold_linked_count = 0
    matched_top1_count = 0
    matched_top5_count = 0
    exact_accuracy_count = 0
    predicted_nil_count = 0
    predicted_nil_correct_count = 0
    short_mention_total = 0
    short_mention_correct = 0
    missing_prediction_count = 0
    decision_counts = Counter[str]()
    error_examples: list[dict[str, Any]] = []

    for gold in gold_records:
        mention_id = str(gold.get("mention_id") or "").strip()
        prediction = prediction_index.get(_prediction_key(gold))
        if prediction is None and mention_id:
            prediction = fallback_prediction_index.get(mention_id)
        if prediction is None:
            missing_prediction_count += 1
            continue

        total_count += 1
        predicted_decision = str(prediction.get("decision") or "UNKNOWN").upper()
        decision_counts[predicted_decision] += 1
        gold_entity_id = _normalize_gold_entity_id(gold)
        predicted_entity_id = (
            str(prediction.get("entity_id") or "").strip()
            if predicted_decision == "LINKED"
            else ""
        )
        predicted_label = predicted_entity_id or None

        if gold_entity_id is not None:
            gold_linked_count += 1
            top_candidates = prediction.get("top_candidates") or prediction.get("candidate_list") or []
            top1_entity_id = str(top_candidates[0].get("entity_id") or "").strip() if top_candidates else ""
            if top1_entity_id == gold_entity_id:
                matched_top1_count += 1
            if any(str(candidate.get("entity_id") or "").strip() == gold_entity_id for candidate in top_candidates[:5]):
                matched_top5_count += 1

        if predicted_label == gold_entity_id:
            exact_accuracy_count += 1
        elif len(error_examples) < 20:
            error_examples.append(
                {
                    "mention_id": mention_id,
                    "mention_text": gold.get("mention_text") or prediction.get("mention_text"),
                    "mention_type": gold.get("mention_type") or prediction.get("mention_type"),
                    "gold_entity_id": gold_entity_id,
                    "predicted_entity_id": predicted_label,
                    "predicted_decision": predicted_decision,
                    "top_candidates": (prediction.get("top_candidates") or [])[:3],
                }
            )

        token_count = int(gold.get("token_end") or prediction.get("token_end") or 0) - int(
            gold.get("token_start") or prediction.get("token_start") or 0
        )
        if token_count <= 1:
            short_mention_total += 1
            if predicted_label == gold_entity_id:
                short_mention_correct += 1

        if predicted_label is None:
            predicted_nil_count += 1
            if gold_entity_id is None:
                predicted_nil_correct_count += 1

    return {
        "gold_count": len(gold_records),
        "matched_prediction_count": total_count,
        "missing_prediction_count": missing_prediction_count,
        "gold_linked_count": gold_linked_count,
        "decision_counts": _sorted_counter(decision_counts),
        "candidate_recall_at_1": round(matched_top1_count / gold_linked_count, 6) if gold_linked_count else None,
        "candidate_recall_at_5": round(matched_top5_count / gold_linked_count, 6) if gold_linked_count else None,
        "linking_accuracy": round(exact_accuracy_count / total_count, 6) if total_count else None,
        "nil_precision": round(predicted_nil_correct_count / predicted_nil_count, 6) if predicted_nil_count else None,
        "short_mention_accuracy": round(short_mention_correct / short_mention_total, 6) if short_mention_total else None,
        "error_examples": error_examples,
    }


def evaluate_linking_from_paths(
    *,
    predictions_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    gold_records = read_jsonl(gold_path)
    summary = evaluate_linking_predictions(predictions, gold_records)
    if output_path is not None:
        write_json(output_path, summary)
    return summary
