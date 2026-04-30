from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import read_jsonl, write_json
from .decode import labels_to_spans, legalize_bio_labels
from .dictionary import MaxForwardDictionaryMatcher
from .features import FeatureConfig, build_sentence_features, ensure_feature_dependencies
from kg_core.taxonomy import ENTITY_TYPES

CONFIDENCE_BUCKET_RULES = (
    ("high", 0.85, 1.0),
    ("medium", 0.65, 0.85),
    ("low", 0.0, 0.65),
)

TARGETED_SLICE_SPECS = {
    "concept_core_terms": {
        "phrases": [
            "Turing machine",
            "universal Turing machine",
            "halting problem",
            "imitation game",
        ],
        "focus_types": ["CONCEPT"],
    },
    "machine_core_terms": {
        "phrases": [
            "Bombe",
            "ACE",
            "Pilot ACE",
            "Ferranti Mark 1",
            "Ferranti Mark I",
            "Enigma cipher machine",
            "Lorenz cipher machine",
        ],
        "focus_types": ["MACHINE"],
    },
    "work_core_terms": {
        "phrases": [
            "Computing Machinery and Intelligence",
            "Systems of Logic Based on Ordinals",
        ],
        "focus_types": ["WORK"],
    },
    "award_core_terms": {
        "phrases": [
            "Turing Award",
            "OBE",
            "Officer of the Most Excellent Order of the British Empire",
        ],
        "focus_types": ["AWARD"],
    },
}

_LAST_FEATURE_BATCH_CONTEXT: dict[str, Any] | None = None
_LAST_PREDICTION_CONFIDENCE_CONTEXT: dict[str, Any] | None = None
_SKLEARN_CRF_PREDICT_PATCHED = False


def require_sklearn_crfsuite() -> Any:
    try:
        import sklearn_crfsuite
    except ImportError as exc:  # pragma: no cover - 依赖缺失时需要清晰报错
        raise RuntimeError(
            "缺少 sklearn-crfsuite。请先在 knowgraph 环境中安装 sklearn-crfsuite。"
        ) from exc
    _patch_crf_predict_for_confidence_capture(sklearn_crfsuite)
    return sklearn_crfsuite


@dataclass(frozen=True)
class CrfTrainingConfig:
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 200
    algorithm: str = "lbfgs"
    all_possible_transitions: bool = True


def _remember_feature_batch(records: list[dict[str, Any]], feature_batch: list[list[dict[str, Any]]]) -> None:
    global _LAST_FEATURE_BATCH_CONTEXT

    _LAST_FEATURE_BATCH_CONTEXT = {
        "feature_batch_id": id(feature_batch),
        "sentence_ids": [str(record["sentence_id"]) for record in records],
    }


def _safe_average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _compute_sequence_confidence(labels: list[str], token_confidences: list[float]) -> float:
    # 优先看实体 token 的边际概率；若整句都预测为 O，再退回整句平均概率。
    entity_confidences = [
        confidence for label, confidence in zip(labels, token_confidences, strict=True) if label != "O"
    ]
    focus_confidences = entity_confidences or token_confidences
    return round(sum(focus_confidences) / len(focus_confidences), 6) if focus_confidences else 0.0


def _compute_sequence_confidences_from_model(
    model: Any,
    feature_batch: list[list[dict[str, Any]]],
    pred_sequences: list[list[str]],
) -> list[float] | None:
    if len(feature_batch) == 0 or len(pred_sequences) == 0:
        return []
    if hasattr(model, "predict_marginals"):
        marginals_batch = model.predict_marginals(feature_batch)
    elif hasattr(model, "predict_marginals_single"):
        marginals_batch = [model.predict_marginals_single(features) for features in feature_batch]
    else:
        return None

    sequence_confidences: list[float] = []
    for marginals, labels in zip(marginals_batch, pred_sequences, strict=True):
        token_confidences = [
            float(token_probs.get(label, 0.0)) for token_probs, label in zip(marginals, labels, strict=True)
        ]
        sequence_confidences.append(_compute_sequence_confidence(labels, token_confidences))
    return sequence_confidences


def _remember_prediction_confidences(
    *,
    sentence_ids: list[str] | None,
    pred_sequences: list[list[str]],
    sequence_confidences: list[float] | None,
    source: str,
) -> None:
    global _LAST_PREDICTION_CONFIDENCE_CONTEXT

    if sequence_confidences is None:
        return
    _LAST_PREDICTION_CONFIDENCE_CONTEXT = {
        "sentence_ids": sentence_ids,
        "pred_sequences": [tuple(sequence) for sequence in pred_sequences],
        "sequence_confidences": sequence_confidences,
        "source": source,
    }


def _patch_crf_predict_for_confidence_capture(sklearn_crfsuite: Any) -> None:
    global _SKLEARN_CRF_PREDICT_PATCHED

    if _SKLEARN_CRF_PREDICT_PATCHED:
        return

    original_predict = sklearn_crfsuite.CRF.predict

    def patched_predict(model: Any, xseqs: list[list[dict[str, Any]]]) -> list[list[str]]:
        predictions = original_predict(model, xseqs)
        sentence_ids: list[str] | None = None
        try:
            if (
                _LAST_FEATURE_BATCH_CONTEXT is not None
                and _LAST_FEATURE_BATCH_CONTEXT.get("feature_batch_id") == id(xseqs)
            ):
                sentence_ids = list(_LAST_FEATURE_BATCH_CONTEXT["sentence_ids"])
            sequence_confidences = _compute_sequence_confidences_from_model(model, xseqs, predictions)
            _remember_prediction_confidences(
                sentence_ids=sentence_ids,
                pred_sequences=predictions,
                sequence_confidences=sequence_confidences,
                source="prediction_marginal",
            )
        except Exception:
            # 置信度是评估增强信息，采集失败时不能影响主预测流程。
            pass
        return predictions

    sklearn_crfsuite.CRF.predict = patched_predict
    _SKLEARN_CRF_PREDICT_PATCHED = True


def _resolve_sequence_confidences(
    records: list[dict[str, Any]],
    pred_sequences: list[list[str]],
) -> tuple[list[float | None], str]:
    sentence_ids = [str(record["sentence_id"]) for record in records]
    prediction_signature = [tuple(sequence) for sequence in pred_sequences]
    if _LAST_PREDICTION_CONFIDENCE_CONTEXT is not None:
        cached_sentence_ids = _LAST_PREDICTION_CONFIDENCE_CONTEXT.get("sentence_ids")
        cached_predictions = _LAST_PREDICTION_CONFIDENCE_CONTEXT.get("pred_sequences")
        if cached_sentence_ids == sentence_ids and cached_predictions == prediction_signature:
            return list(_LAST_PREDICTION_CONFIDENCE_CONTEXT["sequence_confidences"]), str(
                _LAST_PREDICTION_CONFIDENCE_CONTEXT["source"]
            )

    record_confidences: list[float | None] = []
    has_any_record_confidence = False
    for record in records:
        raw_value = record.get("weak_label_confidence")
        if isinstance(raw_value, (int, float)):
            record_confidences.append(float(raw_value))
            has_any_record_confidence = True
        else:
            record_confidences.append(None)
    if has_any_record_confidence:
        return record_confidences, "weak_label_record"
    return [None] * len(records), "unavailable"


def _confidence_to_bucket(confidence: float) -> str:
    if confidence >= CONFIDENCE_BUCKET_RULES[0][1]:
        return "high"
    if confidence >= CONFIDENCE_BUCKET_RULES[1][1]:
        return "medium"
    return "low"


def build_confidence_bucket_metrics(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
    sequence_confidences: list[float | None],
    *,
    source: str,
) -> dict[str, Any]:
    bucket_indices: dict[str, list[int]] = {bucket_name: [] for bucket_name, _, _ in CONFIDENCE_BUCKET_RULES}
    bucket_confidences: dict[str, list[float]] = {bucket_name: [] for bucket_name, _, _ in CONFIDENCE_BUCKET_RULES}
    missing_confidence_record_count = 0

    for index, confidence in enumerate(sequence_confidences):
        if confidence is None:
            missing_confidence_record_count += 1
            continue
        bucket_name = _confidence_to_bucket(confidence)
        bucket_indices[bucket_name].append(index)
        bucket_confidences[bucket_name].append(confidence)

    buckets: dict[str, Any] = {}
    for bucket_name, _, _ in CONFIDENCE_BUCKET_RULES:
        indices = bucket_indices[bucket_name]
        gold_subset = [gold_sequences[index] for index in indices]
        pred_subset = [pred_sequences[index] for index in indices]
        buckets[bucket_name] = {
            "record_count": len(indices),
            "average_confidence": _safe_average(bucket_confidences[bucket_name]),
            "metrics": compute_entity_metrics(gold_subset, pred_subset),
            "sample_sentence_ids": [records[index]["sentence_id"] for index in indices[:10]],
        }

    return {
        "source": source,
        "thresholds": {
            "high": {"min_inclusive": 0.85, "max_inclusive": 1.0},
            "medium": {"min_inclusive": 0.65, "max_exclusive": 0.85},
            "low": {"min_inclusive": 0.0, "max_exclusive": 0.65},
        },
        "known_confidence_record_count": len(sequence_confidences) - missing_confidence_record_count,
        "missing_confidence_record_count": missing_confidence_record_count,
        "buckets": buckets,
    }


def build_dataset_features(
    records: list[dict[str, Any]],
    feature_config: FeatureConfig,
    matcher: MaxForwardDictionaryMatcher | None,
) -> tuple[list[list[dict[str, Any]]], list[list[str]]]:
    ensure_feature_dependencies(feature_config)
    x_values: list[list[dict[str, Any]]] = []
    y_values: list[list[str]] = []
    for record in records:
        features = build_sentence_features(record=record, config=feature_config, matcher=matcher)
        labels = legalize_bio_labels(list(record["labels"]))
        if len(features) != len(labels):
            raise ValueError(f"特征长度与标签长度不一致: {record['sentence_id']}")
        x_values.append(features)
        y_values.append(labels)
    _remember_feature_batch(records, x_values)
    return x_values, y_values


def train_crf_model(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    feature_config: FeatureConfig,
    training_config: CrfTrainingConfig,
    matcher: MaxForwardDictionaryMatcher | None,
) -> tuple[Any, dict[str, Any]]:
    sklearn_crfsuite = require_sklearn_crfsuite()
    x_train, y_train = build_dataset_features(train_records, feature_config, matcher)
    x_dev, y_dev = build_dataset_features(dev_records, feature_config, matcher)

    model = sklearn_crfsuite.CRF(
        algorithm=training_config.algorithm,
        c1=training_config.c1,
        c2=training_config.c2,
        max_iterations=training_config.max_iterations,
        all_possible_transitions=training_config.all_possible_transitions,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_dev)
    sequence_confidences = _compute_sequence_confidences_from_model(model, x_dev, predictions)
    evaluation = evaluate_predictions(
        dev_records,
        y_dev,
        predictions,
        sequence_confidences=sequence_confidences,
        confidence_source="prediction_marginal",
    )
    return model, evaluation


def _flatten_span_keys(labels: list[str]) -> set[tuple[int, int, str]]:
    return set(labels_to_spans(legalize_bio_labels(labels)))


def compute_entity_metrics(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, Any]:
    gold_total = 0
    pred_total = 0
    correct_total = 0

    for gold_labels, pred_labels in zip(gold_sequences, pred_sequences, strict=True):
        gold_spans = _flatten_span_keys(gold_labels)
        pred_spans = _flatten_span_keys(pred_labels)
        gold_total += len(gold_spans)
        pred_total += len(pred_spans)
        correct_total += len(gold_spans & pred_spans)

    precision = correct_total / pred_total if pred_total else 0.0
    recall = correct_total / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gold_entity_count": gold_total,
        "pred_entity_count": pred_total,
        "correct_entity_count": correct_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_per_type_metrics(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, Any]:
    """
    按实体类型统计精确匹配指标。

    这里仍然使用 span+type 全匹配口径，避免与总体指标出现两套标准。
    """

    per_type_counts = {
        entity_type: {
            "gold_entity_count": 0,
            "pred_entity_count": 0,
            "correct_entity_count": 0,
        }
        for entity_type in ENTITY_TYPES
    }

    for gold_labels, pred_labels in zip(gold_sequences, pred_sequences, strict=True):
        gold_spans = labels_to_spans(legalize_bio_labels(gold_labels))
        pred_spans = labels_to_spans(legalize_bio_labels(pred_labels))
        gold_span_set = set(gold_spans)
        pred_span_set = set(pred_spans)
        for _, _, entity_type in gold_spans:
            per_type_counts[entity_type]["gold_entity_count"] += 1
        for _, _, entity_type in pred_spans:
            per_type_counts[entity_type]["pred_entity_count"] += 1
        for _, _, entity_type in gold_span_set & pred_span_set:
            per_type_counts[entity_type]["correct_entity_count"] += 1

    metrics_by_type: dict[str, Any] = {}
    for entity_type, counts in per_type_counts.items():
        gold_total = counts["gold_entity_count"]
        pred_total = counts["pred_entity_count"]
        correct_total = counts["correct_entity_count"]
        precision = correct_total / pred_total if pred_total else 0.0
        recall = correct_total / gold_total if gold_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics_by_type[entity_type] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics_by_type


def build_type_confusion_summary(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    """
    统计同一位置或重叠位置上的类型混淆。

    这里只关心“识别到了某段实体，但类型不一致”的错误，便于观察核心类型混淆。
    """

    confusion_counts: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for record, gold_labels, pred_labels in zip(records, gold_sequences, pred_sequences, strict=True):
        gold_spans = labels_to_spans(legalize_bio_labels(gold_labels))
        pred_spans = labels_to_spans(legalize_bio_labels(pred_labels))
        for gold_start, gold_end, gold_type in gold_spans:
            same_boundary_candidates = [
                span for span in pred_spans if span[0] == gold_start and span[1] == gold_end and span[2] != gold_type
            ]
            overlap_candidates = [
                span
                for span in pred_spans
                if span[2] != gold_type and not (span[1] <= gold_start or span[0] >= gold_end)
            ]
            candidate = same_boundary_candidates[0] if same_boundary_candidates else (
                overlap_candidates[0] if overlap_candidates else None
            )
            if candidate is None:
                continue
            confusion_key = f"{gold_type}->{candidate[2]}"
            confusion_counts[confusion_key] = confusion_counts.get(confusion_key, 0) + 1
            if len(examples) < 20:
                examples.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "gold_span": {
                            "start": gold_start,
                            "end": gold_end,
                            "type": gold_type,
                        },
                        "pred_span": {
                            "start": candidate[0],
                            "end": candidate[1],
                            "type": candidate[2],
                        },
                    }
                )

    focus_pairs = ("CONCEPT->MACHINE", "MACHINE->CONCEPT", "WORK->AWARD", "AWARD->WORK", "ORG->LOC", "LOC->ORG")
    focus_confusions = {pair: confusion_counts.get(pair, 0) for pair in focus_pairs}
    sorted_counts = dict(sorted(confusion_counts.items(), key=lambda item: (-item[1], item[0])))
    return {
        "counts": sorted_counts,
        "focus_pairs": focus_confusions,
        "examples": examples,
    }


def _collect_error_counts_and_examples(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    error_counts = {
        "missed": 0,
        "boundary_short": 0,
        "boundary_long": 0,
        "type_error": 0,
        "spurious": 0,
    }
    examples: list[dict[str, Any]] = []

    for record, gold_labels, pred_labels in zip(records, gold_sequences, pred_sequences, strict=True):
        gold_spans = labels_to_spans(legalize_bio_labels(gold_labels))
        pred_spans = labels_to_spans(legalize_bio_labels(pred_labels))

        for gold_start, gold_end, gold_type in gold_spans:
            exact_match = (gold_start, gold_end, gold_type)
            if exact_match in pred_spans:
                continue

            overlap_predictions = [
                item
                for item in pred_spans
                if not (item[1] <= gold_start or item[0] >= gold_end)
            ]
            matched_prediction: tuple[int, int, str] | None = None
            if not overlap_predictions:
                error_type = "missed"
            else:
                same_type_predictions = [item for item in overlap_predictions if item[2] == gold_type]
                if same_type_predictions:
                    matched_prediction = same_type_predictions[0]
                    if matched_prediction[0] >= gold_start and matched_prediction[1] <= gold_end:
                        error_type = "boundary_short"
                    else:
                        error_type = "boundary_long"
                else:
                    matched_prediction = overlap_predictions[0]
                    error_type = "type_error"
            error_counts[error_type] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "sentence_id": record["sentence_id"],
                        "text": record["text"],
                        "gold_labels": gold_labels,
                        "pred_labels": pred_labels,
                        "gold_span": {"start": gold_start, "end": gold_end, "type": gold_type},
                        "pred_span": (
                            {
                                "start": matched_prediction[0],
                                "end": matched_prediction[1],
                                "type": matched_prediction[2],
                            }
                            if matched_prediction is not None
                            else None
                        ),
                        "error_type": error_type,
                    }
                )

        gold_span_set = set(gold_spans)
        for pred_start, pred_end, pred_type in pred_spans:
            pred_span = (pred_start, pred_end, pred_type)
            if pred_span in gold_span_set:
                continue
            overlaps_gold = any(not (pred_end <= gold[0] or pred_start >= gold[1]) for gold in gold_spans)
            if not overlaps_gold:
                error_counts["spurious"] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "sentence_id": record["sentence_id"],
                            "text": record["text"],
                            "gold_labels": gold_labels,
                            "pred_labels": pred_labels,
                            "gold_span": None,
                            "pred_span": {"start": pred_start, "end": pred_end, "type": pred_type},
                            "error_type": "spurious",
                        }
                    )

    return error_counts, examples


def build_error_analysis(records: list[dict[str, Any]], gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict[str, Any]:
    error_counts, examples = _collect_error_counts_and_examples(records, gold_sequences, pred_sequences)
    return {
        "counts": error_counts,
        "examples": examples,
    }


def build_boundary_error_metrics(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    error_counts, examples = _collect_error_counts_and_examples(records, gold_sequences, pred_sequences)
    overall_metrics = compute_entity_metrics(gold_sequences, pred_sequences)
    gold_total = overall_metrics["gold_entity_count"]
    pred_total = overall_metrics["pred_entity_count"]
    return {
        "counts": error_counts,
        "rates": {
            "boundary_short_over_gold": error_counts["boundary_short"] / gold_total if gold_total else 0.0,
            "boundary_long_over_gold": error_counts["boundary_long"] / gold_total if gold_total else 0.0,
            "type_error_over_gold": error_counts["type_error"] / gold_total if gold_total else 0.0,
            "missed_over_gold": error_counts["missed"] / gold_total if gold_total else 0.0,
            "spurious_over_pred": error_counts["spurious"] / pred_total if pred_total else 0.0,
        },
        "examples": examples[:10],
    }


def _count_raw_illegal_bio_for_sequence(labels: list[str]) -> dict[str, int]:
    invalid_token_count = 0
    invalid_transition_count = 0
    for index, label in enumerate(labels):
        if label == "O":
            continue
        if "-" not in label:
            invalid_token_count += 1
            invalid_transition_count += 1
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            continue
        if prefix != "I":
            invalid_token_count += 1
            invalid_transition_count += 1
            continue
        if index == 0:
            invalid_transition_count += 1
            invalid_token_count += 1
            continue
        previous = labels[index - 1]
        if previous == "O" or "-" not in previous:
            invalid_transition_count += 1
            invalid_token_count += 1
            continue
        _, previous_entity_type = previous.split("-", 1)
        if previous_entity_type != entity_type:
            invalid_transition_count += 1
            invalid_token_count += 1
    return {
        "invalid_token_count": invalid_token_count,
        "invalid_transition_count": invalid_transition_count,
        "sequence_with_invalid_bio": 1 if invalid_token_count > 0 else 0,
    }


def build_raw_illegal_bio_counts(
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    gold_invalid_token_count = 0
    gold_invalid_transition_count = 0
    gold_sequence_count = 0
    pred_invalid_token_count = 0
    pred_invalid_transition_count = 0
    pred_sequence_count = 0

    for labels in gold_sequences:
        counts = _count_raw_illegal_bio_for_sequence(labels)
        gold_invalid_token_count += counts["invalid_token_count"]
        gold_invalid_transition_count += counts["invalid_transition_count"]
        gold_sequence_count += counts["sequence_with_invalid_bio"]

    for labels in pred_sequences:
        counts = _count_raw_illegal_bio_for_sequence(labels)
        pred_invalid_token_count += counts["invalid_token_count"]
        pred_invalid_transition_count += counts["invalid_transition_count"]
        pred_sequence_count += counts["sequence_with_invalid_bio"]

    return {
        "gold": {
            "sequence_count": len(gold_sequences),
            "invalid_sequence_count": gold_sequence_count,
            "invalid_token_count": gold_invalid_token_count,
            "invalid_transition_count": gold_invalid_transition_count,
        },
        "pred": {
            "sequence_count": len(pred_sequences),
            "invalid_sequence_count": pred_sequence_count,
            "invalid_token_count": pred_invalid_token_count,
            "invalid_transition_count": pred_invalid_transition_count,
        },
    }


def _record_matches_phrase(text: str, phrase: str) -> bool:
    return phrase.casefold() in text.casefold()


def build_targeted_slice_metrics(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
) -> dict[str, Any]:
    slice_metrics: dict[str, Any] = {}
    for slice_name, spec in TARGETED_SLICE_SPECS.items():
        matched_indices: list[int] = []
        matched_phrase_counts = {phrase: 0 for phrase in spec["phrases"]}
        for index, record in enumerate(records):
            matched_phrases = [phrase for phrase in spec["phrases"] if _record_matches_phrase(record["text"], phrase)]
            if not matched_phrases:
                continue
            matched_indices.append(index)
            for phrase in matched_phrases:
                matched_phrase_counts[phrase] += 1

        gold_subset = [gold_sequences[index] for index in matched_indices]
        pred_subset = [pred_sequences[index] for index in matched_indices]
        per_type_metrics = compute_per_type_metrics(gold_subset, pred_subset) if matched_indices else {}
        slice_metrics[slice_name] = {
            "record_count": len(matched_indices),
            "metrics": compute_entity_metrics(gold_subset, pred_subset),
            "focus_type_metrics": {
                entity_type: per_type_metrics.get(entity_type, {})
                for entity_type in spec["focus_types"]
            },
            "matched_phrase_counts": {phrase: count for phrase, count in matched_phrase_counts.items() if count > 0},
            "sample_sentence_ids": [records[index]["sentence_id"] for index in matched_indices[:10]],
        }
    return slice_metrics


def evaluate_predictions(
    records: list[dict[str, Any]],
    gold_sequences: list[list[str]],
    pred_sequences: list[list[str]],
    *,
    sequence_confidences: list[float | None] | None = None,
    confidence_source: str | None = None,
) -> dict[str, Any]:
    resolved_confidences = sequence_confidences
    resolved_confidence_source = confidence_source
    if resolved_confidences is None:
        resolved_confidences, resolved_confidence_source = _resolve_sequence_confidences(records, pred_sequences)
    else:
        resolved_confidence_source = resolved_confidence_source or "prediction_marginal"
    metrics = compute_entity_metrics(gold_sequences, pred_sequences)
    per_type_metrics = compute_per_type_metrics(gold_sequences, pred_sequences)
    type_confusions = build_type_confusion_summary(records, gold_sequences, pred_sequences)
    error_analysis = build_error_analysis(records, gold_sequences, pred_sequences)
    return {
        "metrics": metrics,
        "per_type_metrics": per_type_metrics,
        "type_confusions": type_confusions,
        "error_analysis": error_analysis,
        "boundary_error_metrics": build_boundary_error_metrics(records, gold_sequences, pred_sequences),
        "targeted_slice_metrics": build_targeted_slice_metrics(records, gold_sequences, pred_sequences),
        "raw_illegal_bio_counts": build_raw_illegal_bio_counts(gold_sequences, pred_sequences),
        "confidence_bucket_metrics": build_confidence_bucket_metrics(
            records,
            gold_sequences,
            pred_sequences,
            resolved_confidences,
            source=str(resolved_confidence_source),
        ),
    }


def save_training_artifacts(
    model: Any,
    output_dir: Path,
    feature_config: FeatureConfig,
    training_config: CrfTrainingConfig,
    evaluation: dict[str, Any],
) -> None:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 joblib，无法保存 CRF 模型。") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "crf_model.pkl")
    write_json(output_dir / "feature_config.json", feature_config.to_dict())
    write_json(
        output_dir / "training_config.json",
        {
            "algorithm": training_config.algorithm,
            "c1": training_config.c1,
            "c2": training_config.c2,
            "max_iterations": training_config.max_iterations,
            "all_possible_transitions": training_config.all_possible_transitions,
        },
    )
    write_json(output_dir / "eval_dev.json", evaluation)


def train_from_paths(
    train_path: Path,
    dev_path: Path,
    output_dir: Path,
    feature_config: FeatureConfig,
    training_config: CrfTrainingConfig,
    matcher: MaxForwardDictionaryMatcher | None,
) -> dict[str, Any]:
    train_records = read_jsonl(train_path)
    dev_records = read_jsonl(dev_path)
    model, evaluation = train_crf_model(
        train_records=train_records,
        dev_records=dev_records,
        feature_config=feature_config,
        training_config=training_config,
        matcher=matcher,
    )
    save_training_artifacts(
        model=model,
        output_dir=output_dir,
        feature_config=feature_config,
        training_config=training_config,
        evaluation=evaluation,
    )
    return evaluation
