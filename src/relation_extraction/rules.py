from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from kg_core.taxonomy import normalize_entity_type

DEFAULT_RELATION_NAMES: tuple[str, ...] = (
    "BORN_IN",
    "STUDIED_AT",
    "WORKED_AT",
    "AUTHORED",
    "PROPOSED",
    "LOCATED_IN",
)
SUPPORTED_RELATION_NAMES: tuple[str, ...] = DEFAULT_RELATION_NAMES + (
    "DIED_IN",
    "DESIGNED",
    "AWARDED",
)

_TRIM_PATTERN = re.compile(r"(^[^\w]+|[^\w]+$)")
_WORDNET_LEMMATIZER = WordNetLemmatizer()
_WORDNET_READY = False


@dataclass(frozen=True, slots=True)
class RelationRule:
    """关系抽取规则与类型约束的统一视图。"""

    name: str
    ontology_domain: str
    ontology_range: str
    allowed_subject_types: tuple[str, ...]
    allowed_object_types: tuple[str, ...]
    trigger_lemmas: tuple[str, ...]
    trigger_phrases: tuple[str, ...]

    def matches_types(self, subject_type: str, object_type: str) -> bool:
        normalized_subject = normalize_entity_type(subject_type)
        normalized_object = normalize_entity_type(object_type)
        return normalized_subject in self.allowed_subject_types and normalized_object in self.allowed_object_types


TRIGGER_CONFIGS: dict[str, dict[str, tuple[str, ...]]] = {
    "BORN_IN": {
        "lemmas": ("born", "birth"),
        "phrases": ("be born in", "be born at", "born in", "born at"),
    },
    "DIED_IN": {
        "lemmas": ("die", "death", "dead"),
        "phrases": ("die in", "died in", "pass away in", "death in"),
    },
    "STUDIED_AT": {
        "lemmas": ("study", "attend", "educate", "graduate", "matriculate"),
        "phrases": ("study at", "studied at", "attend", "educated at", "graduate from"),
    },
    "WORKED_AT": {
        "lemmas": ("work", "join", "serve", "teach", "research", "employ", "appointment"),
        "phrases": ("work at", "worked at", "work for", "worked for", "join", "serve at", "employed by"),
    },
    "AUTHORED": {
        "lemmas": ("author", "write", "publish", "coauthor"),
        "phrases": ("write", "wrote", "author", "authored", "publish", "published"),
    },
    "PROPOSED": {
        "lemmas": ("propose", "introduce", "devise", "formulate", "present"),
        "phrases": ("propose", "proposed", "introduce", "introduced", "formulate", "devised"),
    },
    "DESIGNED": {
        "lemmas": ("design", "build", "develop", "invent"),
        "phrases": ("design", "designed", "build", "built", "invent", "invented"),
    },
    "AWARDED": {
        "lemmas": ("award", "receive", "win", "honor", "elect"),
        "phrases": ("be awarded", "was awarded", "receive", "received", "win", "won", "elect fellow"),
    },
    "LOCATED_IN": {
        "lemmas": ("locate", "base", "situate"),
        "phrases": ("locate in", "located in", "based in", "situated in"),
    },
}


def _ensure_wordnet_ready() -> None:
    """只在真正执行 trigger 规则时检查 wordnet，避免模块导入即失败。"""

    global _WORDNET_READY
    if _WORDNET_READY:
        return
    try:
        wordnet.ensure_loaded()
    except LookupError as exc:  # pragma: no cover - 依赖环境资源，测试环境可能缺失
        raise RuntimeError(
            "缺少 NLTK wordnet 语料，无法执行关系 trigger 词形还原。"
            "请先在 knowgraph 环境安装该资源，例如执行："
            "python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
        ) from exc
    _WORDNET_READY = True


def _normalize_trigger_token(token: str) -> str:
    lowered = token.casefold().strip()
    trimmed = _TRIM_PATTERN.sub("", lowered)
    if not trimmed:
        return ""
    _ensure_wordnet_ready()
    # 这里优先按动词还原，再按名词收口，兼顾 studied/published/awards 这类常见形态。
    lemma = _WORDNET_LEMMATIZER.lemmatize(trimmed, pos="v")
    lemma = _WORDNET_LEMMATIZER.lemmatize(lemma, pos="n")
    return lemma


@lru_cache(maxsize=256)
def _normalize_trigger_phrase(phrase: str) -> tuple[str, ...]:
    return tuple(token for token in (_normalize_trigger_token(part) for part in phrase.split()) if token)


def normalize_trigger_tokens(tokens: Sequence[str]) -> list[str]:
    return [token for token in (_normalize_trigger_token(token) for token in tokens) if token]


def _coerce_primary_entity_type(raw_type: Any) -> str:
    if isinstance(raw_type, (list, tuple)):
        for item in raw_type:
            normalized_item = normalize_entity_type(item)
            if normalized_item:
                return normalized_item
        return "CONCEPT"
    return normalize_entity_type(raw_type)


def _load_ontology_relation_specs(ontology: dict[str, Any]) -> dict[str, dict[str, str]]:
    relation_specs: dict[str, dict[str, str]] = {}
    for relation in ontology.get("relations", []):
        relation_name = str(relation.get("name", "")).strip().upper()
        if not relation_name:
            continue
        relation_specs[relation_name] = {
            "domain": _coerce_primary_entity_type(relation.get("domain")),
            "range": _coerce_primary_entity_type(relation.get("range")),
        }
    return relation_specs


def _collect_claim_type_overrides(
    claims: Sequence[dict[str, Any]],
    entity_index: dict[str, dict[str, Any]],
) -> dict[str, dict[str, set[str]]]:
    overrides: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"subjects": set(), "objects": set()})
    for claim in claims:
        predicate = str(claim.get("predicate", "")).strip().upper()
        if not predicate:
            continue
        subject_id = str(claim.get("subject_id", "")).strip()
        object_id = str(claim.get("object_id", "")).strip()
        subject_row = entity_index.get(subject_id, {})
        object_row = entity_index.get(object_id, {})
        subject_type = normalize_entity_type(subject_row.get("entity_type"))
        object_type = normalize_entity_type(object_row.get("entity_type"))
        overrides[predicate]["subjects"].add(subject_type)
        overrides[predicate]["objects"].add(object_type)
    return overrides


def build_relation_rules(
    *,
    ontology: dict[str, Any],
    claims: Sequence[dict[str, Any]],
    entity_index: dict[str, dict[str, Any]],
    relation_names: Sequence[str] | None = None,
) -> dict[str, RelationRule]:
    """构建关系规则。

    这里同时吸收 ontology 的权威约束，以及 claims 中真实出现过的类型组合。
    这样可以兜住当前结构化库里 `PROPOSED -> MACHINE` 这类比 ontology 更细的实际情况。
    """

    normalized_relation_names = tuple(
        name.strip().upper() for name in (relation_names or DEFAULT_RELATION_NAMES) if name and name.strip()
    )
    ontology_specs = _load_ontology_relation_specs(ontology)
    claim_type_overrides = _collect_claim_type_overrides(claims, entity_index)
    relation_rules: dict[str, RelationRule] = {}
    for relation_name in normalized_relation_names:
        if relation_name not in ontology_specs and relation_name not in claim_type_overrides:
            raise ValueError(f"关系 {relation_name} 既不在 ontology 中，也没有 claims 支撑，无法构建规则。")
        ontology_domain = ontology_specs.get(relation_name, {}).get("domain", "CONCEPT")
        ontology_range = ontology_specs.get(relation_name, {}).get("range", "CONCEPT")
        trigger_config = TRIGGER_CONFIGS.get(relation_name, {"lemmas": (), "phrases": ()})
        allowed_subject_types = tuple(
            sorted(
                {ontology_domain, *claim_type_overrides.get(relation_name, {}).get("subjects", set())}
            )
        )
        allowed_object_types = tuple(
            sorted(
                {ontology_range, *claim_type_overrides.get(relation_name, {}).get("objects", set())}
            )
        )
        relation_rules[relation_name] = RelationRule(
            name=relation_name,
            ontology_domain=ontology_domain,
            ontology_range=ontology_range,
            allowed_subject_types=allowed_subject_types,
            allowed_object_types=allowed_object_types,
            trigger_lemmas=tuple(trigger_config.get("lemmas", ())),
            trigger_phrases=tuple(trigger_config.get("phrases", ())),
        )
    return relation_rules


def infer_candidate_relations(
    *,
    subject_type: str,
    object_type: str,
    relation_rules: dict[str, RelationRule],
) -> list[str]:
    matched_relations = [
        relation_name
        for relation_name, relation_rule in relation_rules.items()
        if relation_rule.matches_types(subject_type, object_type)
    ]
    return sorted(matched_relations)


def match_relation_triggers(
    *,
    tokens: Sequence[str],
    relation_rule: RelationRule,
) -> list[str]:
    normalized_tokens = normalize_trigger_tokens(tokens)
    if not normalized_tokens:
        return []
    hits: list[str] = []
    normalized_token_set = set(normalized_tokens)
    for trigger_lemma in relation_rule.trigger_lemmas:
        normalized_lemma = _normalize_trigger_token(trigger_lemma)
        if normalized_lemma and normalized_lemma in normalized_token_set:
            hits.append(normalized_lemma)
    for trigger_phrase in relation_rule.trigger_phrases:
        normalized_phrase_tokens = _normalize_trigger_phrase(trigger_phrase)
        if not normalized_phrase_tokens:
            continue
        window_size = len(normalized_phrase_tokens)
        for start in range(0, len(normalized_tokens) - window_size + 1):
            if tuple(normalized_tokens[start : start + window_size]) == normalized_phrase_tokens:
                hits.append(" ".join(normalized_phrase_tokens))
                break
    return sorted(set(hits))


def build_sentence_trigger_map(
    *,
    tokens: Sequence[str],
    relation_rules: dict[str, RelationRule],
) -> dict[str, list[str]]:
    trigger_map: dict[str, list[str]] = {}
    for relation_name, relation_rule in relation_rules.items():
        hits = match_relation_triggers(tokens=tokens, relation_rule=relation_rule)
        if hits:
            trigger_map[relation_name] = hits
    return trigger_map
