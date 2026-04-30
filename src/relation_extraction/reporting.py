from __future__ import annotations

from collections import Counter
from typing import Any


def build_training_report(
    *,
    config_payload: dict[str, Any],
    dataset_report: dict[str, Any],
    class_weights: dict[str, float],
    embedding_report: dict[str, Any],
    history: list[dict[str, Any]],
    best_epoch: int,
    best_dev_metrics: dict[str, Any],
    checkpoint_path: str,
) -> dict[str, Any]:
    return {
        "config": config_payload,
        "dataset": dataset_report,
        "class_weights": class_weights,
        "embedding": embedding_report,
        "history": history,
        "best_epoch": best_epoch,
        "best_dev_metrics": best_dev_metrics,
        "checkpoint_path": checkpoint_path,
    }


def build_prediction_report(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    relation_counter: Counter[str] = Counter()
    support_sentence_counter = 0
    for record in prediction_records:
        for item in record.get("predicted_relations", []):
            relation_counter[str(item["predicate"])] += 1
            support_sentence_counter += len(item.get("supporting_sentence_ids", []))
    predicted_relation_count = sum(relation_counter.values())
    return {
        "bag_count": len(prediction_records),
        "predicted_relation_count": predicted_relation_count,
        "relation_counts": dict(sorted(relation_counter.items())),
        "avg_supporting_sentence_count": round(
            support_sentence_counter / max(predicted_relation_count, 1),
            6,
        ),
    }


def build_evaluation_report(
    *,
    checkpoint_path: str,
    split_name: str,
    prediction_report: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_path": checkpoint_path,
        "split": split_name,
        "prediction_summary": prediction_report,
        "metrics": metrics,
    }
