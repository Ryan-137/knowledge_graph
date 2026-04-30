from __future__ import annotations

from .entity_catalog import normalize_alias_text


PRONOUN_TEXTS = {
    normalize_alias_text(text)
    for text in (
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
    )
}

GENERIC_MENTION_TEXTS = {
    normalize_alias_text(text)
    for text in (
        "the mathematician",
        "the scientist",
        "the school",
        "the laboratory",
        "the machine",
        "the university",
        "the model",
        "the paper",
        "the theory",
        "the concept",
        "the method",
        "this model",
        "this theory",
        "this paper",
        "that model",
        "the work",
        "the result",
        "the idea",
        "the problem",
    )
}


def is_pronoun_mention(text: str) -> bool:
    return normalize_alias_text(text) in PRONOUN_TEXTS


def is_generic_mention(text: str) -> bool:
    return normalize_alias_text(text) in GENERIC_MENTION_TEXTS


def classify_low_information_mention(text: str) -> str | None:
    normalized = normalize_alias_text(text)
    if not normalized:
        return "SKIPPED_LOW_INFORMATION"
    if normalized in PRONOUN_TEXTS:
        return "SKIPPED_PRONOUN"
    if normalized in GENERIC_MENTION_TEXTS:
        return "SKIPPED_GENERIC"
    if len(normalized) <= 1:
        return "SKIPPED_LOW_INFORMATION"
    return None
