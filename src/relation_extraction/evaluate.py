from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import NA_RELATION_LABEL, load_relation_extraction_config
from .dataset import RelationBag, build_relation_bags, infer_target_relations, read_json, read_jsonl, write_json
from .predict import predict_relations
from .reporting import build_evaluation_report, build_prediction_report


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _build_metric_triplet(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def build_prediction_label_sets(
    prediction_records: list[dict[str, Any]],
    *,
    include_na_label: bool = False,
) -> dict[str, set[str]]:
    label_sets: dict[str, set[str]] = {}
    for record in prediction_records:
        bag_id = str(record["bag_id"])
        predicted_relations = record.get("predicted_relations", [])
        label_set = {
            str(item["predicate"])
            for item in predicted_relations
            if include_na_label or str(item["predicate"]) != NA_RELATION_LABEL
        }
        if include_na_label and not label_set:
            label_set = {NA_RELATION_LABEL}
        label_sets[bag_id] = label_set
    return label_sets


def build_gold_label_sets(
    gold_bags: list[RelationBag],
    *,
    include_na_label: bool = False,
) -> dict[str, set[str]]:
    gold_label_sets: dict[str, set[str]] = {}
    for bag in gold_bags:
        labels = set(bag.label_names)
        if not include_na_label:
            labels.discard(NA_RELATION_LABEL)
        gold_label_sets[bag.bag_id] = labels
    return gold_label_sets


def evaluate_prediction_records(
    prediction_records: list[dict[str, Any]],
    gold_bags: list[RelationBag],
    *,
    labels: list[str],
) -> dict[str, Any]:
    evaluated_labels = [label for label in labels if label != NA_RELATION_LABEL]
    predicted_label_sets = build_prediction_label_sets(prediction_records)
    gold_label_sets = build_gold_label_sets(gold_bags)

    all_bag_ids = sorted(set(predicted_label_sets) | set(gold_label_sets))
    per_relation_metrics: dict[str, dict[str, float | int]] = {}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for relation_name in evaluated_labels:
        tp = 0
        fp = 0
        fn = 0
        for bag_id in all_bag_ids:
            predicted_positive = relation_name in predicted_label_sets.get(bag_id, set())
            gold_positive = relation_name in gold_label_sets.get(bag_id, set())
            if predicted_positive and gold_positive:
                tp += 1
            elif predicted_positive and not gold_positive:
                fp += 1
            elif (not predicted_positive) and gold_positive:
                fn += 1
        per_relation_metrics[relation_name] = _build_metric_triplet(tp, fp, fn)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_metrics = _build_metric_triplet(micro_tp, micro_fp, micro_fn)
    if per_relation_metrics:
        macro_precision = sum(float(item["precision"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
        macro_recall = sum(float(item["recall"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
        macro_f1 = sum(float(item["f1"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return {
        "bag_count": len(all_bag_ids),
        "micro": {
            "precision": micro_metrics["precision"],
            "recall": micro_metrics["recall"],
            "f1": micro_metrics["f1"],
            "tp": micro_metrics["tp"],
            "fp": micro_metrics["fp"],
            "fn": micro_metrics["fn"],
        },
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
        "per_relation": per_relation_metrics,
    }


def evaluate_prediction_records_against_gold_records(
    prediction_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    *,
    labels: list[str],
) -> dict[str, Any]:
    evaluated_labels = [label for label in labels if label != NA_RELATION_LABEL]
    predicted_label_sets = build_prediction_label_sets(prediction_records)
    predicted_pairs = {
        (bag_id, relation_name)
        for bag_id, relation_names in predicted_label_sets.items()
        for relation_name in relation_names
    }
    gold_positive_pairs: set[tuple[str, str]] = set()
    gold_negative_pairs: set[tuple[str, str]] = set()
    gold_unknown_pairs: set[tuple[str, str]] = set()
    gold_unknown_record_count = 0
    for record in gold_records:
        relation_name = str(record.get("predicate", "")).strip().upper()
        if relation_name not in evaluated_labels:
            continue
        bag_id = _gold_record_bag_id(record)
        if not bag_id:
            continue
        gold_label = str(record.get("gold_label", "")).strip().lower()
        pair_key = (bag_id, relation_name)
        if gold_label == "positive":
            gold_positive_pairs.add(pair_key)
        elif gold_label == "negative":
            gold_negative_pairs.add(pair_key)
        elif gold_label == "unknown":
            gold_unknown_pairs.add(pair_key)
            gold_unknown_record_count += 1
    covered_gold_pairs = gold_positive_pairs | gold_negative_pairs
    unlabeled_predicted_pairs = predicted_pairs - covered_gold_pairs - gold_unknown_pairs
    per_relation_counts: dict[str, dict[str, int]] = {
        relation_name: {"tp": 0, "fp": 0, "fn": 0}
        for relation_name in evaluated_labels
    }
    for bag_id, relation_name in gold_positive_pairs:
        if (bag_id, relation_name) in predicted_pairs:
            per_relation_counts[relation_name]["tp"] += 1
        else:
            per_relation_counts[relation_name]["fn"] += 1
    for bag_id, relation_name in gold_negative_pairs:
        if (bag_id, relation_name) in predicted_pairs:
            per_relation_counts[relation_name]["fp"] += 1

    per_relation_metrics = {
        relation_name: _build_metric_triplet(counts["tp"], counts["fp"], counts["fn"])
        for relation_name, counts in per_relation_counts.items()
    }
    micro_tp = sum(counts["tp"] for counts in per_relation_counts.values())
    micro_fp = sum(counts["fp"] for counts in per_relation_counts.values())
    micro_fn = sum(counts["fn"] for counts in per_relation_counts.values())
    micro_metrics = _build_metric_triplet(micro_tp, micro_fp, micro_fn)
    if per_relation_metrics:
        macro_precision = sum(float(item["precision"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
        macro_recall = sum(float(item["recall"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
        macro_f1 = sum(float(item["f1"]) for item in per_relation_metrics.values()) / len(per_relation_metrics)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
    return {
        "evaluation_scope": "closed_gold",
        "bag_count": len({bag_id for bag_id, _ in covered_gold_pairs}),
        "gold_record_count": len(gold_records),
        "gold_positive_pair_count": len(gold_positive_pairs),
        "gold_negative_pair_count": len(gold_negative_pairs),
        "gold_unknown_record_count": gold_unknown_record_count,
        "unlabeled_predicted_pair_count": len(unlabeled_predicted_pairs),
        "micro": {
            "precision": micro_metrics["precision"],
            "closed_gold_precision": micro_metrics["precision"],
            "recall": micro_metrics["recall"],
            "f1": micro_metrics["f1"],
            "tp": micro_metrics["tp"],
            "fp": micro_metrics["fp"],
            "fn": micro_metrics["fn"],
        },
        "macro": {
            "precision": round(macro_precision, 6),
            "closed_gold_precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
        "per_relation": per_relation_metrics,
    }


def _gold_record_bag_id(record: dict[str, Any]) -> str:
    bag_id = str(record.get("bag_id", "")).strip()
    if bag_id:
        return bag_id
    doc_id = str(record.get("doc_id", "")).strip()
    subject_id = str(record.get("subject_entity_id", "")).strip()
    object_id = str(record.get("object_entity_id", "")).strip()
    if not doc_id or not subject_id or not object_id:
        return ""
    return f"{doc_id}__{subject_id}__{object_id}"


def evaluate_relation_extractor(
    checkpoint_path: Path,
    *,
    config: Any,
    split_name: str = "test",
    output_path: Path | None = None,
    gold_path: Path | None = None,
    allow_distant_gold: bool = False,
) -> dict[str, Any]:
    gold_records: list[dict[str, Any]] = []
    if gold_path is not None and gold_path.exists():
        gold_records = read_jsonl(gold_path)
    if gold_path is not None and not gold_records and not allow_distant_gold:
        raise ValueError(
            "relation_gold.jsonl 为空或不存在，不能执行真实 gold evaluation。"
            "请补充人工 gold，或显式传入 allow_distant_gold=True 回退到 distant label。"
        )
    target_relations = infer_target_relations(config)
    gold_bags: list[RelationBag] = []
    if not gold_records:
        gold_bags, target_relations = build_relation_bags(config, include_gold_labels=True)
    if split_name != "all":
        prediction_records = predict_relations(
            checkpoint_path,
            config=config,
            split_name=split_name,
        )
        prediction_bag_ids = {str(record["bag_id"]) for record in prediction_records}
        gold_bags = [bag for bag in gold_bags if bag.bag_id in prediction_bag_ids]
        if gold_records:
            gold_records = [
                record
                for record in gold_records
                if _gold_record_bag_id(record) in prediction_bag_ids
            ]
    else:
        prediction_records = predict_relations(checkpoint_path, config=config)
    if gold_records:
        metrics = evaluate_prediction_records_against_gold_records(
            prediction_records,
            gold_records,
            labels=[NA_RELATION_LABEL] + sorted(target_relations),
        )
        evaluation_source = "manual_gold"
    else:
        metrics = evaluate_prediction_records(
            prediction_records,
            gold_bags,
            labels=[NA_RELATION_LABEL] + sorted(target_relations),
        )
        evaluation_source = "distant_label"
    evaluation_report = build_evaluation_report(
        checkpoint_path=checkpoint_path.as_posix(),
        split_name=split_name,
        prediction_report=build_prediction_report(prediction_records),
        metrics=metrics,
    )
    evaluation_report["evaluation_source"] = evaluation_source
    evaluation_report["gold_path"] = gold_path.as_posix() if gold_path else None
    if output_path is not None:
        write_json(output_path, evaluation_report)
    return evaluation_report


def evaluate_relation_predictions(
    *,
    model_dir: Path,
    config_path: Path,
    output_path: Path,
    sentences_path: Path,
    tokenized_sentences_path: Path,
    resolved_mentions_path: Path,
    entities_csv_path: Path,
    aliases_csv_path: Path,
    claims_csv_path: Path,
    ontology_path: Path,
    pair_candidates_path: Path,
    distant_labeled_path: Path,
    target_relations: list[str] | None = None,
    split_name: str = "test",
    gold_path: Path | None = None,
    allow_distant_gold: bool = False,
) -> dict[str, Any]:
    config = load_relation_extraction_config(
        config_path=config_path,
        output_dir=model_dir,
        sentences_path=sentences_path,
        tokenized_sentences_path=tokenized_sentences_path,
        resolved_mentions_path=resolved_mentions_path,
        entities_path=entities_csv_path,
        aliases_path=aliases_csv_path,
        claims_path=claims_csv_path,
        ontology_path=ontology_path,
        pair_candidates_path=pair_candidates_path,
        distant_labeled_path=distant_labeled_path,
        target_relations=target_relations,
    )
    checkpoint_path = model_dir / "relation_model.pt"
    return evaluate_relation_extractor(
        checkpoint_path,
        config=config,
        split_name=split_name,
        output_path=output_path,
        gold_path=gold_path,
        allow_distant_gold=allow_distant_gold,
    )
