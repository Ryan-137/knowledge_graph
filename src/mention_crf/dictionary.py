from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from kg_core.taxonomy import ENTITY_TYPES

from .data import tokenize_sentence_text


ENTITY_TYPE_TO_TAG = {
    "Person": "PER",
    "Organization": "ORG",
    "Place": "LOC",
    "Work": "WORK",
    "Concept": "CONCEPT",
    "Machine": "MACHINE",
    "Award": "AWARD",
}

ORG_ALIAS_STOP_TOKENS = {
    "school",
    "college",
    "university",
    "institute",
    "laboratory",
    "centre",
    "center",
    "museum",
}


def normalize_token(token: str) -> str:
    return token.lower().strip()


@dataclass(frozen=True)
class DictionaryResources:
    alias_type_map: dict[tuple[str, ...], str]
    max_alias_len: int


@dataclass(frozen=True)
class DictionaryMatchSpan:
    start: int
    end: int
    entity_type: str
    tokens: tuple[str, ...]


def _is_probable_location_suffix(tokens: tuple[str, ...], entity_type: str) -> bool:
    """
    判断机构别名逗号后的尾部是否更像地点补语。

    这里只做保守裁剪：仅处理机构别名、逗号后 1-3 个词，尾部包含明显机构词时不裁剪。
    """

    if entity_type != "ORG":
        return False
    if not 1 <= len(tokens) <= 3:
        return False
    if any(token in ORG_ALIAS_STOP_TOKENS for token in tokens):
        return False
    return all(token.replace(".", "").isalpha() for token in tokens)


def build_alias_token_tuples(name: str, entity_type: str) -> list[tuple[str, ...]]:
    """
    基于别名文本生成用于词典匹配的 token 序列。

    对“机构名 + 逗号 + 地点”只在内存里做裁剪，避免把地点尾巴错误并入机构 span。
    """

    token_spans = tokenize_sentence_text(name)
    token_tuple = tuple(normalize_token(item.text) for item in token_spans if item.text.strip())
    if not token_tuple:
        return []

    candidate_tuples = [token_tuple]
    if entity_type == "ORG" and "," in token_tuple:
        comma_index = token_tuple.index(",")
        prefix = token_tuple[:comma_index]
        suffix = token_tuple[comma_index + 1 :]
        if prefix and _is_probable_location_suffix(suffix, entity_type):
            candidate_tuples = [prefix]
    return candidate_tuples


def load_dictionary_resources(entities_csv_path: Path, aliases_csv_path: Path) -> DictionaryResources:
    """
    从结构化抓取结果中构建实体词典。

    这里采用最大前向匹配，避免额外依赖 Aho-Corasick 扩展库。
    """

    entity_id_to_type: dict[str, str] = {}
    entity_id_to_names: dict[str, set[str]] = {}

    with entities_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            entity_type = ENTITY_TYPE_TO_TAG.get((row.get("entity_type") or "").strip())
            if entity_type is None:
                continue
            entity_id = row["entity_id"].strip()
            entity_id_to_type[entity_id] = entity_type
            names = {
                (row.get("canonical_name") or "").strip(),
                (row.get("label_en") or "").strip(),
                (row.get("label_zh") or "").strip(),
            }
            entity_id_to_names.setdefault(entity_id, set()).update(name for name in names if name)

    with aliases_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            entity_id = row["entity_id"].strip()
            alias = (row.get("alias") or "").strip()
            if entity_id not in entity_id_to_type or not alias:
                continue
            entity_id_to_names.setdefault(entity_id, set()).add(alias)

    alias_type_map: dict[tuple[str, ...], str] = {}
    ambiguous_aliases: set[tuple[str, ...]] = set()
    max_alias_len = 1
    for entity_id, names in entity_id_to_names.items():
        entity_type = entity_id_to_type[entity_id]
        for name in names:
            for token_tuple in build_alias_token_tuples(name, entity_type):
                if token_tuple in ambiguous_aliases:
                    continue
                existing_type = alias_type_map.get(token_tuple)
                # 多类型别名歧义较大，直接放弃该词条，避免为 CRF 注入错误边界信号。
                if existing_type is not None and existing_type != entity_type:
                    alias_type_map.pop(token_tuple, None)
                    ambiguous_aliases.add(token_tuple)
                    continue
                alias_type_map[token_tuple] = entity_type
                max_alias_len = max(max_alias_len, len(token_tuple))
    return DictionaryResources(alias_type_map=alias_type_map, max_alias_len=max_alias_len)


class MaxForwardDictionaryMatcher:
    def __init__(self, resources: DictionaryResources) -> None:
        self.resources = resources

    def find_matches(self, tokens: list[str]) -> list[DictionaryMatchSpan]:
        normalized_tokens = [normalize_token(token) for token in tokens]
        matches: list[DictionaryMatchSpan] = []
        cursor = 0
        while cursor < len(tokens):
            match = self.longest_match_at(tokens, cursor, normalized_tokens=normalized_tokens)
            if match is None:
                cursor += 1
                continue
            matches.append(match)
            cursor = match.end
        return matches

    def longest_match_at(
        self,
        tokens: list[str],
        start_index: int,
        *,
        normalized_tokens: list[str] | None = None,
    ) -> DictionaryMatchSpan | None:
        if start_index < 0 or start_index >= len(tokens):
            return None
        normalized = normalized_tokens if normalized_tokens is not None else [normalize_token(token) for token in tokens]
        max_len = min(self.resources.max_alias_len, len(tokens) - start_index)
        for length in range(max_len, 0, -1):
            candidate = tuple(normalized[start_index : start_index + length])
            entity_type = self.resources.alias_type_map.get(candidate)
            if entity_type is None:
                continue
            return DictionaryMatchSpan(
                start=start_index,
                end=start_index + length,
                entity_type=entity_type,
                tokens=candidate,
            )
        return None

    def match(self, tokens: list[str]) -> list[str]:
        dict_tags = ["O"] * len(tokens)
        for match in self.find_matches(tokens):
            dict_tags[match.start] = f"B-{match.entity_type}"
            for index in range(match.start + 1, match.end):
                dict_tags[index] = f"I-{match.entity_type}"
        return dict_tags

    def contains_span(self, tokens: list[str], entity_type: str) -> bool:
        normalized_tokens = tuple(normalize_token(token) for token in tokens)
        return self.resources.alias_type_map.get(normalized_tokens) == entity_type


def filter_supported_entity_type(entity_type: str) -> bool:
    return entity_type in ENTITY_TYPES
