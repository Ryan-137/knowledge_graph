from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
import unicodedata
from pathlib import Path
from typing import Any

from .io import read_csv_records
from .taxonomy import normalize_entity_type


UPPER_ABBREVIATION_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,9}$")
GENERIC_PLACE_SUFFIXES = {
    "city",
    "county",
    "district",
    "borough",
    "park",
    "school",
    "college",
    "university",
    "laboratory",
    "museum",
    "institute",
}

CURATED_ENTITIES = [
    {
        "entity_id": "LOCAL_BLETCHLEY_PARK",
        "canonical_name": "Bletchley Park",
        "label_en": "Bletchley Park",
        "label_zh": "布莱切利园",
        "description_en": "wartime codebreaking centre associated with Alan Turing",
        "description_zh": "图灵参与战时破译工作的地点",
        "wikipedia_title_en": "Bletchley Park",
        "wikipedia_summary_en": "Bletchley Park was the principal centre of Allied codebreaking during the Second World War.",
        "entity_type": "Place",
        "external_ids": "{}",
        "source_name": "curated_local",
        "source_record_id": "LOCAL_BLETCHLEY_PARK",
        "retrieved_at": "",
        "confidence": "1.0",
        "raw_payload_json": "{}",
    },
    {
        "entity_id": "LOCAL_ENIGMA_MACHINE",
        "canonical_name": "Enigma machine",
        "label_en": "Enigma machine",
        "label_zh": "恩尼格玛机",
        "description_en": "cipher machine central to Turing's wartime cryptanalysis",
        "description_zh": "图灵战时密码分析相关的密码机",
        "wikipedia_title_en": "Enigma machine",
        "wikipedia_summary_en": "The Enigma machine was a cipher device used to protect military communication.",
        "entity_type": "Machine",
        "external_ids": "{}",
        "source_name": "curated_local",
        "source_record_id": "LOCAL_ENIGMA_MACHINE",
        "retrieved_at": "",
        "confidence": "1.0",
        "raw_payload_json": "{}",
    },
    {
        "entity_id": "LOCAL_ALONZO_CHURCH",
        "canonical_name": "Alonzo Church",
        "label_en": "Alonzo Church",
        "label_zh": "阿隆佐·邱奇",
        "description_en": "mathematician and Turing's doctoral supervisor at Princeton",
        "description_zh": "图灵在普林斯顿时期的导师",
        "wikipedia_title_en": "Alonzo Church",
        "wikipedia_summary_en": "Alonzo Church was an American mathematician and logician.",
        "entity_type": "Person",
        "external_ids": "{}",
        "source_name": "curated_local",
        "source_record_id": "LOCAL_ALONZO_CHURCH",
        "retrieved_at": "",
        "confidence": "1.0",
        "raw_payload_json": "{}",
    },
    {
        "entity_id": "LOCAL_MAX_NEWMAN",
        "canonical_name": "Max Newman",
        "label_en": "Max Newman",
        "label_zh": "马克斯·纽曼",
        "description_en": "mathematician connected with Turing at Cambridge and Manchester",
        "description_zh": "与图灵在剑桥和曼彻斯特相关的数学家",
        "wikipedia_title_en": "Max Newman",
        "wikipedia_summary_en": "Max Newman was a British mathematician and computer scientist.",
        "entity_type": "Person",
        "external_ids": "{}",
        "source_name": "curated_local",
        "source_record_id": "LOCAL_MAX_NEWMAN",
        "retrieved_at": "",
        "confidence": "1.0",
        "raw_payload_json": "{}",
    },
    {
        "entity_id": "LOCAL_MANCHESTER_MARK_1",
        "canonical_name": "Manchester Mark 1",
        "label_en": "Manchester Mark 1",
        "label_zh": "曼彻斯特一号",
        "description_en": "early stored-program computer associated with Manchester and Turing",
        "description_zh": "与曼彻斯特和图灵相关的早期存储程序计算机",
        "wikipedia_title_en": "Manchester Mark 1",
        "wikipedia_summary_en": "The Manchester Mark 1 was an early stored-program computer.",
        "entity_type": "Machine",
        "external_ids": "{}",
        "source_name": "curated_local",
        "source_record_id": "LOCAL_MANCHESTER_MARK_1",
        "retrieved_at": "",
        "confidence": "1.0",
        "raw_payload_json": "{}",
    },
]

CURATED_ALIASES = [
    {"entity_id": "LOCAL_BLETCHLEY_PARK", "alias": "Bletchley Park", "alias_lang": "en", "normalized_alias": "bletchley park"},
    {"entity_id": "LOCAL_BLETCHLEY_PARK", "alias": "Bletchley", "alias_lang": "en", "normalized_alias": "bletchley"},
    {"entity_id": "LOCAL_ENIGMA_MACHINE", "alias": "Enigma", "alias_lang": "en", "normalized_alias": "enigma"},
    {"entity_id": "LOCAL_ENIGMA_MACHINE", "alias": "Enigma machine", "alias_lang": "en", "normalized_alias": "enigma machine"},
    {"entity_id": "LOCAL_ENIGMA_MACHINE", "alias": "Enigma cipher machine", "alias_lang": "en", "normalized_alias": "enigma cipher machine"},
    {"entity_id": "LOCAL_ENIGMA_MACHINE", "alias": "Enigma cipher machines", "alias_lang": "en", "normalized_alias": "enigma cipher machines"},
    {"entity_id": "LOCAL_ALONZO_CHURCH", "alias": "Alonzo Church", "alias_lang": "en", "normalized_alias": "alonzo church"},
    {"entity_id": "LOCAL_MAX_NEWMAN", "alias": "Max Newman", "alias_lang": "en", "normalized_alias": "max newman"},
    {"entity_id": "LOCAL_MANCHESTER_MARK_1", "alias": "Manchester Mark 1", "alias_lang": "en", "normalized_alias": "manchester mark 1"},
    {"entity_id": "LOCAL_MANCHESTER_MARK_1", "alias": "Manchester Mk 1", "alias_lang": "en", "normalized_alias": "manchester mk 1"},
    {"entity_id": "LOCAL_MANCHESTER_MARK_1", "alias": "Mark 1", "alias_lang": "en", "normalized_alias": "mark 1"},
]


@dataclass(frozen=True)
class EntityCatalog:
    """结构化实体库的共享内存视图。"""

    entities: dict[str, dict[str, Any]]
    aliases: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    exact_alias_index: dict[str, list[dict[str, Any]]]
    normalized_alias_index: dict[str, list[dict[str, Any]]]
    person_surname_index: dict[str, list[dict[str, Any]]]
    organization_abbreviation_index: dict[str, list[dict[str, Any]]]
    full_short_name_index: dict[str, list[dict[str, Any]]]
    place_variant_index: dict[str, list[dict[str, Any]]]
    alias_surface_index_by_entity: dict[str, list[dict[str, Any]]]
    claims_adjacency: dict[str, set[str]]
    retrieval_text_by_entity: dict[str, str]

    @property
    def aliases_by_text(self) -> dict[str, list[dict[str, Any]]]:
        return self.normalized_alias_index


def normalize_alias_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("’", "'").casefold().strip()
    normalized = re.sub(r"(?<=\w)'s$", "", normalized)
    return " ".join(normalized.split())


def normalize_exact_alias_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).replace("’", "'").strip().split())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = normalize_alias_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(" ".join(str(value).strip().split()))
    return deduped


def _is_abbreviation(alias: str) -> bool:
    return bool(UPPER_ABBREVIATION_RE.fullmatch(alias.strip()))


def _extract_person_surname(alias: str) -> str | None:
    tokens = [token for token in alias.replace(".", " ").split() if token]
    if len(tokens) < 2:
        return None
    surname = tokens[-1]
    return surname if surname.isalpha() and len(surname) > 2 else None


def _extract_org_short_name(alias: str) -> str | None:
    if "," in alias:
        prefix = alias.split(",", 1)[0].strip()
        if prefix:
            return prefix
    if "(" in alias and ")" in alias:
        prefix = alias.split("(", 1)[0].strip()
        if prefix:
            return prefix
    return None


def _extract_place_variant(alias: str) -> str | None:
    if "," in alias:
        prefix = alias.split(",", 1)[0].strip()
        if prefix:
            return prefix
    tokens = alias.split()
    if len(tokens) >= 2 and tokens[-1].casefold() in GENERIC_PLACE_SUFFIXES:
        prefix = " ".join(tokens[:-1]).strip()
        if prefix:
            return prefix
    return None


def _domain_terms(entity_row: dict[str, Any]) -> list[str]:
    raw_parts = [
        str(entity_row.get("entity_type") or "").strip(),
        str(entity_row.get("description_en") or "").strip(),
        str(entity_row.get("description_zh") or "").strip(),
        str(entity_row.get("wikipedia_summary_en") or "").strip(),
        str(entity_row.get("canonical_name") or "").strip(),
    ]
    tokens: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        tokens.extend(re.findall(r"[A-Za-z][A-Za-z\-]+", part))
    return _dedupe_keep_order(tokens[:24])


def _build_retrieval_text(entity_row: dict[str, Any], alias_rows: list[dict[str, Any]]) -> str:
    alias_texts = [str(item.get("alias") or "").strip() for item in alias_rows]
    parts = [
        str(entity_row.get("canonical_name") or "").strip(),
        str(entity_row.get("label_en") or "").strip(),
        str(entity_row.get("label_zh") or "").strip(),
        str(entity_row.get("description_en") or "").strip(),
        str(entity_row.get("description_zh") or "").strip(),
        str(entity_row.get("wikipedia_summary_en") or "").strip(),
        " ".join(_domain_terms(entity_row)),
        " ".join(alias_texts),
    ]
    return " ".join(part for part in parts if part).strip()


def _build_claims_adjacency(claims: list[dict[str, Any]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in claims:
        subject_id = str(row.get("subject_id") or "").strip()
        object_id = str(row.get("object_id") or "").strip()
        if not subject_id or not object_id:
            continue
        adjacency[subject_id].add(object_id)
        adjacency[object_id].add(subject_id)
    return {key: set(value) for key, value in adjacency.items()}


def _append_index(
    index: dict[str, list[dict[str, Any]]],
    key: str,
    alias_row: dict[str, Any],
) -> None:
    if not key:
        return
    existing = index.setdefault(key, [])
    dedupe_key = (
        str(alias_row.get("entity_id") or "").strip(),
        str(alias_row.get("normalized_alias") or "").strip(),
        str(alias_row.get("alias_type") or "").strip(),
    )
    if any(
        (
            str(item.get("entity_id") or "").strip(),
            str(item.get("normalized_alias") or "").strip(),
            str(item.get("alias_type") or "").strip(),
        )
        == dedupe_key
        for item in existing
    ):
        return
    existing.append(alias_row)


def load_entity_catalog(
    entities_csv_path: str | Path,
    aliases_csv_path: str | Path,
    claims_csv_path: str | Path | None = None,
) -> EntityCatalog:
    entities: dict[str, dict[str, Any]] = {}
    for row in read_csv_records(entities_csv_path):
        entity_id = str(row.get("entity_id", "")).strip()
        if not entity_id:
            raise ValueError(f"{entities_csv_path} 存在空 entity_id")
        row["entity_type"] = normalize_entity_type(row.get("entity_type"))
        entities[entity_id] = dict(row)
    for row in CURATED_ENTITIES:
        entity_id = str(row["entity_id"])
        if entity_id in entities:
            continue
        curated_row = dict(row)
        curated_row["entity_type"] = normalize_entity_type(curated_row.get("entity_type"))
        entities[entity_id] = curated_row

    alias_rows_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases: list[dict[str, Any]] = []
    exact_alias_index: dict[str, list[dict[str, Any]]] = {}
    normalized_alias_index: dict[str, list[dict[str, Any]]] = {}
    person_surname_index: dict[str, list[dict[str, Any]]] = {}
    organization_abbreviation_index: dict[str, list[dict[str, Any]]] = {}
    full_short_name_index: dict[str, list[dict[str, Any]]] = {}
    place_variant_index: dict[str, list[dict[str, Any]]] = {}
    alias_surface_index_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in [*read_csv_records(aliases_csv_path), *CURATED_ALIASES]:
        entity_id = str(row.get("entity_id", "")).strip()
        alias = normalize_exact_alias_text(str(row.get("alias", "")))
        if not entity_id or not alias:
            raise ValueError(f"{aliases_csv_path} 存在空 entity_id 或 alias")
        entity_row = entities.get(entity_id)
        if entity_row is None:
            continue
        entity_type = normalize_entity_type(row.get("entity_type") or entity_row.get("entity_type"))
        normalized_alias = normalize_alias_text(row.get("normalized_alias") or alias)
        alias_type = "alias"
        alias_weight = 0.9
        surname = _extract_person_surname(alias) if entity_type == "PERSON" else None
        short_name = _extract_org_short_name(alias) if entity_type == "ORGANIZATION" else None
        place_variant = _extract_place_variant(alias) if entity_type == "PLACE" else None
        if _is_abbreviation(alias) and entity_type == "ORGANIZATION":
            alias_type = "abbreviation"
            alias_weight = 0.96
        elif surname is not None:
            alias_type = "surname"
            alias_weight = 0.88
        elif short_name is not None and normalize_alias_text(short_name) != normalized_alias:
            alias_type = "short_name"
            alias_weight = 0.92
        elif place_variant is not None and normalize_alias_text(place_variant) != normalized_alias:
            alias_type = "place_variant"
            alias_weight = 0.9

        normalized_row = {
            **dict(row),
            "entity_id": entity_id,
            "alias": alias,
            "normalized_alias": normalized_alias,
            "entity_type": entity_type,
            "alias_type": alias_type,
            "alias_weight": alias_weight,
        }
        aliases.append(normalized_row)
        alias_rows_by_entity[entity_id].append(normalized_row)
        _append_index(exact_alias_index, alias, normalized_row)
        _append_index(normalized_alias_index, normalized_alias, normalized_row)
        if surname is not None:
            surname_row = {**normalized_row, "alias": surname, "normalized_alias": normalize_alias_text(surname), "alias_type": "surname", "alias_weight": 0.88}
            _append_index(person_surname_index, normalize_alias_text(surname), surname_row)
        if _is_abbreviation(alias) and entity_type == "ORGANIZATION":
            _append_index(organization_abbreviation_index, normalize_alias_text(alias), normalized_row)
        if short_name is not None:
            short_name_row = {
                **normalized_row,
                "alias": short_name,
                "normalized_alias": normalize_alias_text(short_name),
                "alias_type": "short_name",
                "alias_weight": 0.92,
            }
            _append_index(full_short_name_index, normalize_alias_text(short_name), short_name_row)
        if place_variant is not None:
            place_variant_row = {
                **normalized_row,
                "alias": place_variant,
                "normalized_alias": normalize_alias_text(place_variant),
                "alias_type": "place_variant",
                "alias_weight": 0.9,
            }
            _append_index(place_variant_index, normalize_alias_text(place_variant), place_variant_row)
        alias_surface_index_by_entity[entity_id].append(
            {
                "surface": alias,
                "normalized_surface": normalized_alias,
                "alias_type": alias_type,
                "alias_weight": alias_weight,
            }
        )

    for entity_id, entity_row in entities.items():
        built_in_aliases = _dedupe_keep_order(
            [
                str(entity_row.get("canonical_name") or "").strip(),
                str(entity_row.get("label_en") or "").strip(),
                str(entity_row.get("label_zh") or "").strip(),
                str(entity_row.get("wikipedia_title_en") or "").strip(),
            ]
        )
        for alias in built_in_aliases:
            normalized_alias = normalize_alias_text(alias)
            surface_entry = {
                "surface": alias,
                "normalized_surface": normalized_alias,
                "alias_type": "canonical" if alias == str(entity_row.get("canonical_name") or "").strip() else "builtin",
                "alias_weight": 1.0 if alias == str(entity_row.get("canonical_name") or "").strip() else 0.95,
            }
            if all(item["normalized_surface"] != normalized_alias for item in alias_surface_index_by_entity[entity_id]):
                alias_surface_index_by_entity[entity_id].append(surface_entry)
            builtin_row = {
                "entity_id": entity_id,
                "alias": alias,
                "alias_lang": "",
                "normalized_alias": normalized_alias,
                "entity_type": normalize_entity_type(entity_row.get("entity_type")),
                "alias_type": surface_entry["alias_type"],
                "alias_weight": surface_entry["alias_weight"],
            }
            _append_index(exact_alias_index, alias, builtin_row)
            _append_index(normalized_alias_index, normalized_alias, builtin_row)
            if entity_row.get("entity_type") == "PERSON":
                surname = _extract_person_surname(alias)
                if surname is not None:
                    surname_row = {
                        **builtin_row,
                        "alias": surname,
                        "normalized_alias": normalize_alias_text(surname),
                        "alias_type": "surname",
                        "alias_weight": 0.88,
                    }
                    _append_index(person_surname_index, normalize_alias_text(surname), surname_row)
            if entity_row.get("entity_type") == "ORGANIZATION":
                if _is_abbreviation(alias):
                    _append_index(organization_abbreviation_index, normalize_alias_text(alias), builtin_row)
                short_name = _extract_org_short_name(alias)
                if short_name is not None:
                    short_name_row = {
                        **builtin_row,
                        "alias": short_name,
                        "normalized_alias": normalize_alias_text(short_name),
                        "alias_type": "short_name",
                        "alias_weight": 0.92,
                    }
                    _append_index(full_short_name_index, normalize_alias_text(short_name), short_name_row)
            if entity_row.get("entity_type") == "PLACE":
                place_variant = _extract_place_variant(alias)
                if place_variant is not None:
                    place_variant_row = {
                        **builtin_row,
                        "alias": place_variant,
                        "normalized_alias": normalize_alias_text(place_variant),
                        "alias_type": "place_variant",
                        "alias_weight": 0.9,
                    }
                    _append_index(place_variant_index, normalize_alias_text(place_variant), place_variant_row)

    claims = read_csv_records(claims_csv_path) if claims_csv_path and Path(claims_csv_path).exists() else []
    retrieval_text_by_entity = {
        entity_id: _build_retrieval_text(entity_row, alias_rows_by_entity.get(entity_id, []))
        for entity_id, entity_row in entities.items()
    }
    return EntityCatalog(
        entities=entities,
        aliases=aliases,
        claims=claims,
        exact_alias_index=exact_alias_index,
        normalized_alias_index=normalized_alias_index,
        person_surname_index=person_surname_index,
        organization_abbreviation_index=organization_abbreviation_index,
        full_short_name_index=full_short_name_index,
        place_variant_index=place_variant_index,
        alias_surface_index_by_entity={key: value for key, value in alias_surface_index_by_entity.items()},
        claims_adjacency=_build_claims_adjacency(claims),
        retrieval_text_by_entity=retrieval_text_by_entity,
    )
