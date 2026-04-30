from __future__ import annotations

ENTITY_TYPES = (
    "PER",
    "ORG",
    "LOC",
    "WORK",
    "CONCEPT",
    "MACHINE",
    "AWARD",
)

CANONICAL_ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "PLACE",
    "WORK",
    "CONCEPT",
    "MACHINE",
    "AWARD",
    "EVENT",
)

BIO_LABELS = tuple(["O"] + [f"{prefix}-{entity_type}" for entity_type in ENTITY_TYPES for prefix in ("B", "I")])
VALID_LABEL_SET = set(BIO_LABELS)

MENTION_LABEL_TO_ENTITY_TYPE = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "PLACE",
    "WORK": "WORK",
    "CONCEPT": "CONCEPT",
    "MACHINE": "MACHINE",
    "AWARD": "AWARD",
}

MENTION_TYPE_TO_ENTITY_TYPE = {
    **MENTION_LABEL_TO_ENTITY_TYPE,
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "ORGANISATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "PLACE": "PLACE",
    "LOCATION": "PLACE",
    "LOC": "PLACE",
    "WORK": "WORK",
    "CONCEPT": "CONCEPT",
    "MACHINE": "MACHINE",
    "AWARD": "AWARD",
    "EVENT": "EVENT",
}

ENTITY_TYPE_TO_MENTION_TYPE = {
    "PERSON": "PER",
    "ORGANIZATION": "ORG",
    "PLACE": "LOC",
    "WORK": "WORK",
    "CONCEPT": "CONCEPT",
    "MACHINE": "MACHINE",
    "AWARD": "AWARD",
    "EVENT": "CONCEPT",
}

# 兼容现有调用名称，但语义改为“结构化实体类型 -> mention 标签类型”。
ENTITY_TYPE_TO_BIO_TAG = dict(ENTITY_TYPE_TO_MENTION_TYPE)


def normalize_mention_type(raw_type: str | None) -> str:
    if not raw_type:
        return "CONCEPT"
    normalized = raw_type.strip().upper()
    return MENTION_TYPE_TO_ENTITY_TYPE.get(normalized, "CONCEPT")


def normalize_entity_type(raw_type: str | None) -> str:
    if not raw_type:
        return "CONCEPT"
    normalized = raw_type.strip().upper()
    if normalized in CANONICAL_ENTITY_TYPES:
        return normalized
    return MENTION_TYPE_TO_ENTITY_TYPE.get(normalized, "CONCEPT")


def canonical_entity_type_from_mention_label(raw_type: str | None) -> str:
    if not raw_type:
        return "CONCEPT"
    normalized = raw_type.strip().upper()
    return MENTION_LABEL_TO_ENTITY_TYPE.get(normalized, normalize_entity_type(normalized))
