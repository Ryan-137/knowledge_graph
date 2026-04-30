from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def aggregate_entity_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """把 Wikidata 多行 instance/label 查询结果聚合成单实体。"""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        entity_uri = get_binding_value(row, "entity")
        if not entity_uri:
            continue
        entity_id = entity_uri.rsplit("/", 1)[-1]
        bucket = grouped.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "label_en": None,
                "label_zh": None,
                "description_en": None,
                "description_zh": None,
                "wikipedia_title_en": None,
                "birth_date": None,
                "death_date": None,
                "instance_of_ids": [],
                "instance_of_labels": [],
                "rows": [],
            },
        )
        bucket["label_en"] = bucket["label_en"] or get_binding_value(row, "entityLabelEn")
        bucket["label_zh"] = bucket["label_zh"] or get_binding_value(row, "entityLabelZh")
        bucket["description_en"] = bucket["description_en"] or get_binding_value(row, "descriptionEn")
        bucket["description_zh"] = bucket["description_zh"] or get_binding_value(row, "descriptionZh")
        bucket["wikipedia_title_en"] = bucket["wikipedia_title_en"] or get_binding_value(row, "wikipediaTitleEn")
        bucket["birth_date"] = bucket["birth_date"] or get_binding_value(row, "birthDate")
        bucket["death_date"] = bucket["death_date"] or get_binding_value(row, "deathDate")
        instance_uri = get_binding_value(row, "instanceOf")
        if instance_uri:
            bucket["instance_of_ids"].append(instance_uri.rsplit("/", 1)[-1])
        instance_label = get_binding_value(row, "instanceOfLabel")
        if instance_label:
            bucket["instance_of_labels"].append(instance_label)
        bucket["rows"].append(row)
    return grouped


def aggregate_alias_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """把别名查询结果按实体和语言聚合。"""
    grouped: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        entity_uri = get_binding_value(row, "entity")
        alias = get_binding_value(row, "alias")
        alias_lang = get_binding_value(row, "aliasLang")
        if not entity_uri or not alias:
            continue
        entity_id = entity_uri.rsplit("/", 1)[-1]
        bucket = grouped.setdefault(entity_id, {"en": [], "zh": []})
        if alias_lang in ("en", "zh"):
            bucket[alias_lang].append(alias)
    return grouped


def build_entity_record(
    grouped: dict[str, Any],
    aliases_by_lang: dict[str, list[str]],
    entity_type: str,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """把聚合后的 Wikidata 实体转换成 entities/aliases 两类落库记录。"""
    label_en = grouped["label_en"]
    label_zh = grouped["label_zh"]
    canonical_name = choose_canonical_name(label_zh, label_en, grouped["entity_id"])
    alias_records = build_alias_records(
        canonical_name=canonical_name,
        label_en=label_en,
        label_zh=label_zh,
        aliases_en=aliases_by_lang.get("en", []),
        aliases_zh=aliases_by_lang.get("zh", []),
    )
    external_ids = {
        "wikidata_qid": grouped["entity_id"],
        "wikipedia_title_en": grouped["wikipedia_title_en"],
    }
    if grouped["wikipedia_title_en"]:
        external_ids["dbpedia_uri"] = build_dbpedia_uri(grouped["wikipedia_title_en"])
    entity = {
        "entity_id": grouped["entity_id"],
        "canonical_name": canonical_name,
        "label_en": label_en,
        "label_zh": label_zh,
        "description_en": grouped["description_en"],
        "description_zh": grouped["description_zh"],
        "wikipedia_title_en": grouped["wikipedia_title_en"],
        "wikipedia_summary_en": None,
        "entity_type": entity_type,
        "external_ids": external_ids,
        "source_name": "wikidata_sparql",
        "source_record_id": grouped["entity_id"],
        "retrieved_at": utc_now_text(),
        "confidence": confidence,
        "raw_payload_json": {
            "instance_of_ids": dedupe_preserve_order(grouped["instance_of_ids"]),
            "instance_of_labels": dedupe_preserve_order(grouped["instance_of_labels"]),
            "birth_date_raw": grouped["birth_date"],
            "death_date_raw": grouped["death_date"],
        },
    }
    return entity, alias_records


def choose_canonical_name(label_zh: str | None, label_en: str | None, fallback: str) -> str:
    return (label_zh or label_en or fallback).strip()


def build_alias_records(
    canonical_name: str,
    label_en: str | None,
    label_zh: str | None,
    aliases_en: list[str],
    aliases_zh: list[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for alias, alias_lang in (
        *((item, "en") for item in aliases_en),
        *((item, "zh") for item in aliases_zh),
        *((item, "en") for item in [label_en] if item),
        *((item, "zh") for item in [label_zh] if item),
    ):
        normalized = normalize_alias(alias)
        if not normalized or normalized == normalize_alias(canonical_name) or normalized in seen:
            continue
        seen.add(normalized)
        records.append(
            {
                "alias": alias.strip(),
                "alias_lang": alias_lang,
                "normalized_alias": normalized,
            }
        )
    return records


def normalize_alias(alias: str | None) -> str:
    if not alias:
        return ""
    alias = re.sub(r"\s+", " ", alias.strip())
    return alias.casefold()


def build_dbpedia_uri(title: str) -> str:
    safe_title = title.replace(" ", "_")
    return f"http://dbpedia.org/resource/{safe_title}"


def utc_now_text() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def get_binding_value(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if not value:
        return None
    return value.get("value")


def extract_qid(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rsplit("/", 1)[-1]


def simplify_binding_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value.get("value") for key, value in row.items()}


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
