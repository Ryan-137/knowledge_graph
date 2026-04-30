from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RELATION_PATTERNS: dict[str, list[str]] = {
    "BORN_IN": ["born in", "born at", "birthplace"],
    "DIED_IN": ["died in", "death in", "passed away in"],
    "STUDIED_AT": ["studied at", "educated at", "attended", "graduated from", "went to", "study"],
    "WORKED_AT": ["worked at", "worked for", "joined", "employed by", "served at", "appointment"],
    "AUTHORED": ["wrote", "authored", "published", "paper by"],
    "PROPOSED": ["proposed", "introduced", "outlined", "formulated", "devised"],
    "DESIGNED": ["designed", "built", "developed", "invented"],
    "AWARDED": ["awarded", "received", "won", "elected fellow"],
    "LOCATED_IN": ["located in", "based in", "situated in"],
}

PREDICATE_ALIASES = {
    "BORN_IN": "BORN_IN",
    "BORN_AT": "BORN_IN",
    "DIED_IN": "DIED_IN",
    "DIED_AT": "DIED_IN",
    "EDUCATED_AT": "STUDIED_AT",
    "STUDIED_AT": "STUDIED_AT",
    "WORKED_AT": "WORKED_AT",
    "EMPLOYED_BY": "WORKED_AT",
    "AUTHORED": "AUTHORED",
    "CREATED": "DESIGNED",
    "KNOWN_FOR": "PROPOSED",
    "PROPOSED": "PROPOSED",
    "DESIGNED": "DESIGNED",
    "AWARD_RECEIVED": "AWARDED",
    "AWARDED": "AWARDED",
    "LOCATED_IN": "LOCATED_IN",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_GENERIC_SINGLE_TOKEN_TRIGGERS = {"in", "at", "to", "of", "by", "for"}


def canonicalize_predicate(raw_predicate: str) -> str:
    normalized = raw_predicate.strip().upper().replace("-", "_").replace(" ", "_")
    return PREDICATE_ALIASES.get(normalized, normalized)


def _normalize_token(token: str) -> str:
    normalized = token.casefold().strip()
    normalized = re.sub(r"(^[^\w]+|[^\w]+$)", "", normalized)
    if len(normalized) > 4 and normalized.endswith("ied"):
        return normalized[:-3] + "y"
    for suffix in ("ing", "ed", "s"):
        if len(normalized) > len(suffix) + 3 and normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _tokenize_text(text: str) -> list[str]:
    return [_normalize_token(match.group(0)) for match in _TOKEN_PATTERN.finditer(text)]


def load_relation_patterns(patterns_path: str | Path | None) -> dict[str, list[str]]:
    patterns = {predicate: list(triggers) for predicate, triggers in DEFAULT_RELATION_PATTERNS.items()}
    if patterns_path is None:
        return patterns
    path = Path(patterns_path)
    if not path.exists():
        return patterns
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return patterns
    for raw_predicate, raw_config in payload.items():
        predicate = canonicalize_predicate(str(raw_predicate))
        raw_triggers = raw_config.get("triggers") if isinstance(raw_config, dict) else raw_config
        if not isinstance(raw_triggers, list):
            continue
        merged = list(patterns.get(predicate, []))
        for trigger in raw_triggers:
            trigger_text = str(trigger).strip()
            if _is_generic_single_token_trigger(trigger_text):
                continue
            if trigger_text and trigger_text not in merged:
                merged.append(trigger_text)
        patterns[predicate] = merged
    return patterns


def _is_generic_single_token_trigger(trigger_text: str) -> bool:
    trigger_tokens = _tokenize_text(trigger_text)
    return len(trigger_tokens) == 1 and trigger_tokens[0] in _GENERIC_SINGLE_TOKEN_TRIGGERS


def _contains_ordered_subsequence(tokens: list[str], phrase_tokens: list[str]) -> bool:
    if not phrase_tokens:
        return False
    cursor = 0
    for token in tokens:
        if token == phrase_tokens[cursor]:
            cursor += 1
            if cursor == len(phrase_tokens):
                return True
    return False


def _subject_object_distance(candidate: dict[str, Any]) -> int | None:
    subject_span = list(candidate.get("subject_token_span") or [])
    object_span = list(candidate.get("object_token_span") or [])
    if len(subject_span) != 2:
        subject_span = [candidate.get("subject_token_start"), candidate.get("subject_token_end")]
    if len(object_span) != 2:
        object_span = [candidate.get("object_token_start"), candidate.get("object_token_end")]
    if any(value is None for value in (*subject_span, *object_span)):
        return None
    subject_start, subject_end = int(subject_span[0]), int(subject_span[1])
    object_start, object_end = int(object_span[0]), int(object_span[1])
    if subject_end <= object_start:
        return object_start - subject_end
    if object_end <= subject_start:
        return subject_start - object_end
    return 0


def match_pattern_signals(
    candidate: dict[str, Any],
    *,
    relation_patterns: dict[str, list[str]],
    max_token_distance: int = 24,
) -> dict[str, Any]:
    predicate = canonicalize_predicate(str(candidate.get("predicate", "")))
    text = str(candidate.get("text") or "")
    tokens = [_normalize_token(token) for token in list(candidate.get("tokens") or [])]
    if not tokens:
        tokens = _tokenize_text(text)
    distance = _subject_object_distance(candidate)
    if distance is not None and distance > max_token_distance:
        return {"matched": False, "hits": [], "token_distance": distance, "score": 0.0}

    hits: list[str] = []
    normalized_text = " ".join(tokens)
    for trigger in relation_patterns.get(predicate, []):
        trigger_tokens = _tokenize_text(str(trigger))
        if not trigger_tokens:
            continue
        trigger_text = " ".join(trigger_tokens)
        if trigger_text in normalized_text or _contains_ordered_subsequence(tokens, trigger_tokens):
            hits.append(str(trigger))

    return {
        "matched": bool(hits),
        "hits": sorted(set(hits)),
        "token_distance": distance,
        "score": 0.35 if hits else 0.0,
    }
