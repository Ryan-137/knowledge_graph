from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"[_\-/]+", " ", normalized)
    normalized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_for_similarity(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    return {
        token
        for token in normalized.split()
        if token and token not in STOPWORDS and len(token) > 1
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def name_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.86
    token_score = jaccard(tokenize_for_similarity(left_norm), tokenize_for_similarity(right_norm))
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(token_score, sequence_score)


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result
