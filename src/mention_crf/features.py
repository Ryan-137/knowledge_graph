from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .dictionary import MaxForwardDictionaryMatcher, normalize_token


PREPOSITION_TOKENS = {"at", "in", "of", "for", "to", "from", "with"}
MERGEABLE_TITLE_CONNECTORS = {"and", "of", "the", "for", "in", "on", "to"}
QUOTE_TOKENS = {'"', "'", "“", "”", "‘", "’"}
OPEN_PAREN_TOKENS = {"(", "[", "{"}
CLOSE_PAREN_TOKENS = {")", "]", "}"}
COLON_TOKENS = {":", "："}
MACHINE_TRIGGER_TOKENS = {"machine", "computer", "engine", "cipher", "digital", "pilot", "mark"}
CONCEPT_TRIGGER_TOKENS = {"machine", "test", "thesis", "problem", "game", "logic", "theory"}
WORK_TRIGGER_TOKENS = {"paper", "article", "manual", "report", "intelligence", "logic"}
AWARD_TRIGGER_TOKENS = {"award", "order", "prize", "medal", "fellowship", "obe"}
ROMAN_NUMERAL_RE = re.compile(r"^(?=[MDCLXVI])(M|CM|D|CD|C|XC|L|XL|X|IX|V|IV|I)+$", re.IGNORECASE)
UPPER_ABBREVIATION_RE = re.compile(r"^[A-Z][A-Z0-9\-]{1,7}$")


@dataclass(frozen=True)
class FeatureConfig:
    use_pos: bool = True
    use_dict: bool = True
    use_time_hint: bool = True
    window_size: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_nltk() -> Any:
    try:
        import nltk
    except ImportError as exc:  # pragma: no cover - 依赖缺失时直接给出操作建议
        raise RuntimeError(
            "缺少 nltk。请先在 knowgraph 环境中安装 nltk，并下载 averaged_perceptron_tagger_eng 资源。"
        ) from exc
    return nltk


def build_pos_tags(tokens: list[str]) -> list[str]:
    nltk = require_nltk()
    try:
        return [item[1] for item in nltk.pos_tag(tokens, lang="eng")]
    except LookupError as exc:  # pragma: no cover
        raise RuntimeError(
            "nltk 已安装，但缺少英文词性标注资源。请执行 nltk.download('averaged_perceptron_tagger_eng')。"
        ) from exc


def word_shape(token: str) -> str:
    chars: list[str] = []
    for char in token:
        if char.isupper():
            chars.append("X")
        elif char.islower():
            chars.append("x")
        elif char.isdigit():
            chars.append("d")
        else:
            chars.append(char)
    return "".join(chars)


def _is_roman_numeral(token: str) -> bool:
    return bool(token) and bool(ROMAN_NUMERAL_RE.fullmatch(token))


def _is_upper_abbreviation(token: str) -> bool:
    return bool(UPPER_ABBREVIATION_RE.fullmatch(token))


def _has_model_number_pattern(token: str) -> bool:
    has_digit = any(char.isdigit() for char in token)
    has_alpha = any(char.isalpha() for char in token)
    return has_digit and has_alpha


def _is_title_style_token(token: str) -> bool:
    cleaned = token.strip(".,:;!?()[]{}\"'")
    if not cleaned:
        return False
    return cleaned.istitle() or _is_upper_abbreviation(cleaned) or _is_roman_numeral(cleaned)


def _compute_parenthesis_depths(tokens: list[str]) -> list[int]:
    depths: list[int] = []
    depth = 0
    for token in tokens:
        depths.append(depth)
        if token in OPEN_PAREN_TOKENS:
            depth += 1
        elif token in CLOSE_PAREN_TOKENS and depth > 0:
            depth -= 1
    return depths


def _compute_inside_quotes(tokens: list[str]) -> list[bool]:
    inside_flags: list[bool] = []
    inside_quote = False
    for token in tokens:
        inside_flags.append(inside_quote and token not in QUOTE_TOKENS)
        if token in QUOTE_TOKENS:
            inside_quote = not inside_quote
    return inside_flags


def _compute_title_run_lengths(tokens: list[str]) -> list[int]:
    run_lengths = [0] * len(tokens)
    index = 0
    while index < len(tokens):
        lower_token = tokens[index].lower()
        if not (_is_title_style_token(tokens[index]) or lower_token in MERGEABLE_TITLE_CONNECTORS):
            index += 1
            continue
        start = index
        title_like_count = 1 if _is_title_style_token(tokens[index]) else 0
        index += 1
        while index < len(tokens):
            lower_token = tokens[index].lower()
            if _is_title_style_token(tokens[index]):
                title_like_count += 1
                index += 1
                continue
            if lower_token in MERGEABLE_TITLE_CONNECTORS:
                index += 1
                continue
            break
        run_length = index - start if title_like_count >= 2 else 0
        for cursor in range(start, index):
            run_lengths[cursor] = run_length
    return run_lengths


def _default_alias_feature() -> dict[str, Any]:
    return {
        "alias_exact_match": False,
        "alias_entity_type": "O",
        "alias_length": 0,
        "alias_is_single": False,
        "alias_is_start": False,
        "alias_is_end": False,
        "alias_token_offset": -1,
        "alias_tokens_from_end": -1,
    }


def _build_dictionary_match_features(
    tokens: list[str],
    matcher: MaxForwardDictionaryMatcher | None,
) -> tuple[list[str] | None, list[dict[str, Any]]]:
    alias_features = [_default_alias_feature() for _ in tokens]
    if matcher is None:
        return None, alias_features

    normalized_tokens = [normalize_token(token) for token in tokens]
    dict_tags = ["O"] * len(tokens)
    cursor = 0
    while cursor < len(tokens):
        matched = False
        max_len = min(matcher.resources.max_alias_len, len(tokens) - cursor)
        for length in range(max_len, 0, -1):
            candidate = tuple(normalized_tokens[cursor : cursor + length])
            entity_type = matcher.resources.alias_type_map.get(candidate)
            if entity_type is None:
                continue
            dict_tags[cursor] = f"B-{entity_type}"
            for offset in range(length):
                token_index = cursor + offset
                if offset > 0:
                    dict_tags[token_index] = f"I-{entity_type}"
                alias_features[token_index] = {
                    "alias_exact_match": True,
                    "alias_entity_type": entity_type,
                    "alias_length": length,
                    "alias_is_single": length == 1,
                    "alias_is_start": offset == 0,
                    "alias_is_end": offset == length - 1,
                    "alias_token_offset": offset,
                    "alias_tokens_from_end": length - offset - 1,
                }
            cursor += length
            matched = True
            break
        if not matched:
            cursor += 1
    return dict_tags, alias_features


def token_features(
    tokens: list[str],
    index: int,
    pos_tags: list[str] | None,
    dict_tags: list[str] | None,
    alias_feature: dict[str, Any],
    sentence_has_time: bool,
    sentence_index_in_doc: int,
    parenthesis_depths: list[int],
    inside_quotes: list[bool],
    title_run_lengths: list[int],
    config: FeatureConfig,
) -> dict[str, Any]:
    token = tokens[index]
    lower_token = token.lower()
    prev_token = tokens[index - 1] if index > 0 else ""
    next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
    features: dict[str, Any] = {
        "bias": 1.0,
        "token": token,
        "lower": lower_token,
        "shape": word_shape(token),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "is_digit": token.isdigit(),
        "has_hyphen": "-" in token,
        "has_dot": "." in token,
        "prefix_2": lower_token[:2],
        "prefix_3": lower_token[:3],
        "suffix_2": lower_token[-2:],
        "suffix_3": lower_token[-3:],
        "token_len": len(token),
        "is_sentence_start": index == 0,
        "is_sentence_end": index == len(tokens) - 1,
        "is_doc_leading_sentence": sentence_index_in_doc == 1,
        "is_preposition": lower_token in PREPOSITION_TOKENS,
        "is_quote_like": token in QUOTE_TOKENS,
        "inside_quotes": inside_quotes[index],
        "adjacent_left_quote": prev_token in QUOTE_TOKENS if index > 0 else False,
        "adjacent_right_quote": next_token in QUOTE_TOKENS if index + 1 < len(tokens) else False,
        "inside_parentheses": parenthesis_depths[index] > 0,
        "is_open_parenthesis": token in OPEN_PAREN_TOKENS,
        "is_close_parenthesis": token in CLOSE_PAREN_TOKENS,
        "after_colon": prev_token in COLON_TOKENS if index > 0 else False,
        "before_colon": next_token in COLON_TOKENS if index + 1 < len(tokens) else False,
        "title_run_len": title_run_lengths[index],
        "in_title_like_span": title_run_lengths[index] >= 2,
        "sentence_has_time": sentence_has_time if config.use_time_hint else False,
        "is_roman_numeral": _is_roman_numeral(token),
        "has_model_number_pattern": _has_model_number_pattern(token),
        "is_upper_abbreviation": _is_upper_abbreviation(token),
        "machine_trigger": lower_token in MACHINE_TRIGGER_TOKENS,
        "concept_trigger": lower_token in CONCEPT_TRIGGER_TOKENS,
        "work_trigger": lower_token in WORK_TRIGGER_TOKENS,
        "award_trigger": lower_token in AWARD_TRIGGER_TOKENS,
    }

    if config.use_pos and pos_tags is not None:
        features["pos"] = pos_tags[index]
        features["prev_pos"] = pos_tags[index - 1] if index > 0 else "__BOS__"
        features["next_pos"] = pos_tags[index + 1] if index + 1 < len(pos_tags) else "__EOS__"

    if config.use_dict and dict_tags is not None:
        features["dict_tag"] = dict_tags[index]
        features["alias_exact_match"] = alias_feature["alias_exact_match"]
        features["alias_entity_type"] = alias_feature["alias_entity_type"]
        features["alias_length"] = alias_feature["alias_length"]
        features["alias_is_single"] = alias_feature["alias_is_single"]
        features["alias_is_start"] = alias_feature["alias_is_start"]
        features["alias_is_end"] = alias_feature["alias_is_end"]
        features["alias_token_offset"] = alias_feature["alias_token_offset"]
        features["alias_tokens_from_end"] = alias_feature["alias_tokens_from_end"]

    for step in range(1, config.window_size + 1):
        left_index = index - step
        right_index = index + step

        if left_index >= 0:
            left_token = tokens[left_index]
            features[f"-{step}:lower"] = left_token.lower()
            features[f"-{step}:shape"] = word_shape(left_token)
            features[f"-{step}:is_title"] = left_token.istitle()
            features[f"-{step}:is_digit"] = left_token.isdigit()
            features[f"-{step}:is_upper_abbrev"] = _is_upper_abbreviation(left_token)
        else:
            features[f"-{step}:BOS"] = True

        if right_index < len(tokens):
            right_token = tokens[right_index]
            features[f"+{step}:lower"] = right_token.lower()
            features[f"+{step}:shape"] = word_shape(right_token)
            features[f"+{step}:is_title"] = right_token.istitle()
            features[f"+{step}:is_digit"] = right_token.isdigit()
            features[f"+{step}:is_upper_abbrev"] = _is_upper_abbreviation(right_token)
        else:
            features[f"+{step}:EOS"] = True
    return features


def build_sentence_features(
    record: dict[str, Any],
    config: FeatureConfig,
    matcher: MaxForwardDictionaryMatcher | None = None,
) -> list[dict[str, Any]]:
    tokens = list(record["tokens"])
    if not tokens:
        return []

    pos_tags = build_pos_tags(tokens) if config.use_pos else None
    dict_tags, alias_features = _build_dictionary_match_features(tokens, matcher) if config.use_dict else (None, [])
    sentence_has_time = bool(record.get("normalized_time"))
    sentence_index_in_doc = int(record.get("sentence_index_in_doc", 0))
    parenthesis_depths = _compute_parenthesis_depths(tokens)
    inside_quotes = _compute_inside_quotes(tokens)
    title_run_lengths = _compute_title_run_lengths(tokens)
    if not alias_features:
        alias_features = [_default_alias_feature() for _ in tokens]
    return [
        token_features(
            tokens=tokens,
            index=index,
            pos_tags=pos_tags,
            dict_tags=dict_tags,
            alias_feature=alias_features[index],
            sentence_has_time=sentence_has_time,
            sentence_index_in_doc=sentence_index_in_doc,
            parenthesis_depths=parenthesis_depths,
            inside_quotes=inside_quotes,
            title_run_lengths=title_run_lengths,
            config=config,
        )
        for index in range(len(tokens))
    ]


def ensure_feature_dependencies(config: FeatureConfig) -> None:
    if config.use_pos:
        require_nltk()


def looks_like_meaningless_single_token(token: str) -> bool:
    if len(token) <= 1 and not token.isalnum():
        return True
    if re.fullmatch(r"[_\-=/\\]+", token):
        return True
    return False
