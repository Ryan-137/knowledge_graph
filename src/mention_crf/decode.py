from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kg_core.schemas import MentionRecord, normalize_mention_text
from kg_core.taxonomy import canonical_entity_type_from_mention_label

from .dictionary import MaxForwardDictionaryMatcher
from .features import looks_like_meaningless_single_token

MERGE_BRIDGE_TOKENS = {"of", "and", "in", "for"}
MERGEABLE_MENTION_TYPES = {"WORK", "CONCEPT"}


def legalize_bio_labels(labels: list[str]) -> list[str]:
    """把少量非法 BIO 输出纠正成可解码的合法序列。"""

    if not labels:
        return []

    normalized = list(labels)
    for index, label in enumerate(normalized):
        if label == "O":
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            continue
        if index == 0:
            normalized[index] = f"B-{entity_type}"
            continue
        previous = normalized[index - 1]
        if previous == "O":
            normalized[index] = f"B-{entity_type}"
            continue
        _, previous_entity_type = previous.split("-", 1)
        if previous_entity_type != entity_type:
            normalized[index] = f"B-{entity_type}"
    return normalized


def labels_to_spans(labels: list[str]) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    entity_type: str | None = None

    for index, label in enumerate(labels):
        if label == "O":
            if start is not None and entity_type is not None:
                spans.append((start, index, entity_type))
                start = None
                entity_type = None
            continue

        prefix, current_type = label.split("-", 1)
        if prefix == "B":
            if start is not None and entity_type is not None:
                spans.append((start, index, entity_type))
            start = index
            entity_type = current_type
            continue

        if start is None or entity_type != current_type:
            if start is not None and entity_type is not None:
                spans.append((start, index, entity_type))
            start = index
            entity_type = current_type
            continue

    if start is not None and entity_type is not None:
        spans.append((start, len(labels), entity_type))
    return spans


def _span_confidence(token_confidences: list[float | None] | None, token_start: int, token_end: int) -> float | None:
    if token_confidences is None:
        return None
    values = [item for item in token_confidences[token_start:token_end] if item is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_mention_record(
    record: dict[str, Any],
    mention_id: str,
    token_start: int,
    token_end: int,
    entity_type: str,
    token_confidences: list[float | None] | None = None,
    recall_source: str = "crf_model",
) -> dict[str, Any]:
    token_spans = record["token_spans"][token_start:token_end]
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
        mention_type=canonical_entity_type_from_mention_label(entity_type),
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
        extractor="crf",
        confidence=_span_confidence(token_confidences, token_start, token_end),
        recall_source=recall_source,
    ).to_dict()


def _expand_machine_span_with_dictionary(
    record: dict[str, Any],
    token_start: int,
    token_end: int,
    entity_type: str,
    dictionary_matcher: MaxForwardDictionaryMatcher | None,
) -> int:
    """
    只对高置信机器 alias 做前缀扩展。

    这里不做通用 span 扩张，避免把其他类型的边界错误“修出来”。
    """

    if entity_type != "MACHINE" or dictionary_matcher is None:
        return token_end
    match = dictionary_matcher.longest_match_at(record["tokens"], token_start)
    if match is None or match.entity_type != "MACHINE":
        return token_end
    predicted_tokens = tuple(token.casefold() for token in record["tokens"][token_start:token_end])
    if predicted_tokens != match.tokens[: len(predicted_tokens)]:
        return token_end
    return max(token_end, match.end)


def merge_work_and_concept_mentions(
    mentions: list[dict[str, Any]],
    record: dict[str, Any],
    dictionary_contains_span: Callable[[list[str], str], bool] | None = None,
    token_confidences: list[float | None] | None = None,
    recall_source: str = "crf_model",
) -> list[dict[str, Any]]:
    if len(mentions) <= 1:
        return mentions

    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(mentions):
        current = mentions[index]
        current_type = current["mention_type"]
        if current_type not in MERGEABLE_MENTION_TYPES or index + 1 >= len(mentions):
            merged.append(current)
            index += 1
            continue

        next_item = mentions[index + 1]
        if next_item["mention_type"] != current_type:
            merged.append(current)
            index += 1
            continue

        gap_tokens = record["tokens"][current["token_end"] : next_item["token_start"]]
        if not gap_tokens:
            merged.append(current)
            index += 1
            continue

        lower_gap_tokens = [token.lower() for token in gap_tokens]
        if len(lower_gap_tokens) > 2 or any(token not in MERGE_BRIDGE_TOKENS for token in lower_gap_tokens):
            merged.append(current)
            index += 1
            continue

        merged_tokens = record["tokens"][current["token_start"] : next_item["token_end"]]
        if dictionary_contains_span is not None and not dictionary_contains_span(merged_tokens, current_type):
            merged.append(current)
            index += 1
            continue

        merged.append(
            _build_mention_record(
                record=record,
                mention_id=current["mention_id"],
                token_start=current["token_start"],
                token_end=next_item["token_end"],
                entity_type=current_type,
                token_confidences=token_confidences,
                recall_source=recall_source,
            )
        )
        index += 2
    return merged


def decode_mentions_from_labels(
    record: dict[str, Any],
    labels: list[str],
    start_index: int,
    dictionary_contains_span: Callable[[list[str], str], bool] | None = None,
    machine_alias_span_resolver: Callable[[list[str], int, int], tuple[int, int] | None] | None = None,
    token_confidences: list[float | None] | None = None,
    recall_source: str = "crf_model",
    dictionary_matcher: MaxForwardDictionaryMatcher | None = None,
) -> tuple[list[dict[str, Any]], int]:
    legalized_labels = legalize_bio_labels(labels)
    spans = labels_to_spans(legalized_labels)
    mentions: list[dict[str, Any]] = []
    counter = start_index
    span_contains = dictionary_contains_span
    if span_contains is None and dictionary_matcher is not None:
        span_contains = dictionary_matcher.contains_span

    for token_start, token_end, entity_type in spans:
        if entity_type == "MACHINE" and machine_alias_span_resolver is not None:
            resolved_span = machine_alias_span_resolver(record["tokens"], token_start, token_end)
            if resolved_span is not None:
                token_start, token_end = resolved_span
        else:
            token_end = _expand_machine_span_with_dictionary(
                record=record,
                token_start=token_start,
                token_end=token_end,
                entity_type=entity_type,
                dictionary_matcher=dictionary_matcher,
            )
        tokens = record["tokens"][token_start:token_end]
        if len(tokens) == 1 and looks_like_meaningless_single_token(tokens[0]):
            continue
        mentions.append(
            _build_mention_record(
                record=record,
                mention_id=f"ment_{counter:06d}",
                token_start=token_start,
                token_end=token_end,
                entity_type=entity_type,
                token_confidences=token_confidences,
                recall_source=recall_source,
            )
        )
        counter += 1

    merged_mentions = merge_work_and_concept_mentions(
        mentions=mentions,
        record=record,
        dictionary_contains_span=span_contains,
        token_confidences=token_confidences,
        recall_source=recall_source,
    )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for mention in merged_mentions:
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
    return deduped, counter
