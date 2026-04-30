from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.schemas import MentionRecord, normalize_mention_text
from kg_core.taxonomy import canonical_entity_type_from_mention_label

from .data import build_tokenized_record, read_json, read_jsonl, write_jsonl
from .decode import decode_mentions_from_labels, legalize_bio_labels
from .dictionary import DictionaryMatchSpan, MaxForwardDictionaryMatcher, normalize_token
from .features import FeatureConfig, build_sentence_features

DICTIONARY_FALLBACK_ENTITY_TYPES = {"WORK", "CONCEPT", "MACHINE", "AWARD"}


def load_model(model_path: Path) -> Any:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 joblib，无法加载 CRF 模型。") from exc
    return joblib.load(model_path)


def load_feature_config(config_path: Path) -> FeatureConfig:
    payload = read_json(config_path)
    return FeatureConfig(
        use_pos=bool(payload.get("use_pos", True)),
        use_dict=bool(payload.get("use_dict", True)),
        use_time_hint=bool(payload.get("use_time_hint", True)),
        window_size=int(payload.get("window_size", 2)),
    )


def _predict_token_confidences(model: Any, features: list[dict[str, Any]], labels: list[str]) -> list[float | None] | None:
    """
    从 sklearn-crfsuite 的边际概率中估计 token 置信度。

    若模型对象不支持 predict_marginals_single，则保留 None，避免伪造置信度。
    """

    if not hasattr(model, "predict_marginals_single"):
        return None
    marginals = model.predict_marginals_single(features)
    return [float(token_probs.get(label, 0.0)) for token_probs, label in zip(marginals, labels, strict=True)]


def _build_machine_alias_span_resolver(
    matcher: MaxForwardDictionaryMatcher | None,
) -> Any:
    if matcher is None:
        return None
    alias_type_map = matcher.resources.alias_type_map
    max_alias_len = matcher.resources.max_alias_len

    def resolve_machine_alias_span(tokens: list[str], token_start: int, token_end: int) -> tuple[int, int] | None:
        normalized_tokens = [normalize_token(token) for token in tokens]
        current_prefix = tuple(normalized_tokens[token_start:token_end])
        current_length = len(current_prefix)
        max_length = min(max_alias_len, len(tokens) - token_start)
        for candidate_length in range(max_length, current_length, -1):
            candidate = tuple(normalized_tokens[token_start : token_start + candidate_length])
            if alias_type_map.get(candidate) != "MACHINE":
                continue
            if candidate_length < 2:
                continue
            if candidate[:current_length] != current_prefix:
                continue
            return token_start, token_start + candidate_length
        return None

    return resolve_machine_alias_span


def _build_dictionary_mention_record(
    record: dict[str, Any],
    mention_id: str,
    match: DictionaryMatchSpan,
) -> dict[str, Any]:
    token_spans = record["token_spans"][match.start:match.end]
    char_start = token_spans[0][0]
    char_end = token_spans[-1][1]
    mention_text = record["text"][char_start:char_end]
    return MentionRecord(
        mention_id=mention_id,
        sentence_id=record["sentence_id"],
        doc_id=record["doc_id"],
        source_id=record["source_id"],
        text=mention_text,
        normalized_text=normalize_mention_text(mention_text),
        mention_type=canonical_entity_type_from_mention_label(match.entity_type),
        char_start=char_start,
        char_end=char_end,
        token_start=match.start,
        token_end=match.end,
        extractor="dictionary",
        confidence=None,
        recall_source="dictionary_fallback",
    ).to_dict()


def _spans_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return not (left_end <= right_start or left_start >= right_end)


def _collect_dictionary_fallback_mentions(
    record: dict[str, Any],
    matcher: MaxForwardDictionaryMatcher | None,
    existing_mentions: list[dict[str, Any]],
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    if matcher is None:
        return [], start_index

    existing_spans = [(mention["token_start"], mention["token_end"]) for mention in existing_mentions]
    fallback_mentions: list[dict[str, Any]] = []
    counter = start_index
    for match in matcher.find_matches(record["tokens"]):
        if match.entity_type not in DICTIONARY_FALLBACK_ENTITY_TYPES:
            continue
        if any(_spans_overlap(match.start, match.end, span_start, span_end) for span_start, span_end in existing_spans):
            continue
        fallback_mentions.append(
            _build_dictionary_mention_record(
                record=record,
                mention_id=f"ment_{counter:06d}",
                match=match,
            )
        )
        existing_spans.append((match.start, match.end))
        counter += 1
    return fallback_mentions, counter


def _dedupe_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for mention in mentions:
        dedupe_key = (
            mention["sentence_id"],
            mention["mention_type"],
            mention["char_start"],
            mention["char_end"],
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(mention)
    return deduped


def predict_mentions(
    sentences_path: Path,
    output_path: Path,
    model_path: Path,
    feature_config_path: Path,
    matcher: MaxForwardDictionaryMatcher | None,
) -> tuple[int, int]:
    records = read_jsonl(sentences_path)
    model = load_model(model_path)
    feature_config = load_feature_config(feature_config_path)
    machine_alias_span_resolver = _build_machine_alias_span_resolver(matcher)

    mention_records: list[dict[str, Any]] = []
    mention_counter = 1
    for record in records:
        tokenized_record = build_tokenized_record(record) if "tokens" not in record else record
        features = build_sentence_features(tokenized_record, feature_config, matcher)
        if not features:
            continue
        raw_labels = list(model.predict_single(features))
        labels = legalize_bio_labels(raw_labels)
        token_confidences = _predict_token_confidences(model, features, labels)
        mentions, mention_counter = decode_mentions_from_labels(
            record=tokenized_record,
            labels=labels,
            start_index=mention_counter,
            dictionary_contains_span=matcher.contains_span if matcher is not None else None,
            machine_alias_span_resolver=machine_alias_span_resolver,
            token_confidences=token_confidences,
            recall_source="crf_model",
        )
        fallback_mentions, mention_counter = _collect_dictionary_fallback_mentions(
            record=tokenized_record,
            matcher=matcher,
            existing_mentions=mentions,
            start_index=mention_counter,
        )
        mention_records.extend(_dedupe_mentions(mentions + fallback_mentions))

    write_jsonl(output_path, mention_records)
    return len(records), len(mention_records)
