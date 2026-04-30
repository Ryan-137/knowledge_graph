from __future__ import annotations


INSTANCE_TYPE_MAP = {
    "Q5": "Person",
    "Q43229": "Organization",
    "Q2385804": "Organization",
    "Q17334923": "Place",
    "Q2221906": "Place",
    "Q618123": "Place",
    "Q571": "Work",
    "Q7725634": "Work",
    "Q386724": "Work",
    "Q151885": "Concept",
    "Q7184903": "Concept",
    "Q11012": "Machine",
    "Q170978": "Machine",
    "Q618779": "Award",
}

ENTITY_KEYWORD_TYPE_HINTS = {
    "person": "Person",
    "human": "Person",
    "organization": "Organization",
    "agency": "Organization",
    "headquarters": "Organization",
    "institution": "Organization",
    "university": "Organization",
    "college": "Organization",
    "school": "Organization",
    "laboratory": "Organization",
    "lab": "Organization",
    "museum": "Organization",
    "intelligence service": "Organization",
    "intelligence agency": "Organization",
    "residential district": "Place",
    "district": "Place",
    "municipality": "Place",
    "borough": "Place",
    "civil parish": "Place",
    "market town": "Place",
    "town": "Place",
    "village": "Place",
    "hamlet": "Place",
    "park": "Place",
    "city": "Place",
    "country": "Place",
    "place": "Place",
    "county town": "Place",
    "building": "Place",
    "nursing home": "Place",
    "work": "Work",
    "paper": "Work",
    "book": "Work",
    "report": "Work",
    "proof": "Work",
    "concept": "Concept",
    "theory": "Concept",
    "machine": "Machine",
    "computer": "Machine",
    "medal": "Award",
    "order of chivalry": "Award",
    "honour": "Award",
    "honor": "Award",
    "rank of the order": "Award",
    "award": "Award",
    "prize": "Award",
}

TITLE_KEYWORD_TYPE_HINTS = {
    "fellow": "Award",
    "prize": "Award",
    "award": "Award",
    "medal": "Award",
    "order of": "Award",
    "machine": "Machine",
    "engine": "Machine",
    "computer": "Machine",
    "test": "Concept",
    "theory": "Concept",
    "proof": "Work",
    "report": "Work",
    "school": "Organization",
    "university": "Organization",
    "college": "Organization",
    "laboratory": "Organization",
    "headquarters": "Organization",
    "museum": "Organization",
    "district": "Place",
    "borough": "Place",
    "municipality": "Place",
    "town": "Place",
    "village": "Place",
    "city": "Place",
    "park": "Place",
}

DESCRIPTION_KEYWORD_TYPE_HINTS = {
    "fellow of the royal society": "Award",
    "elected fellow": "Award",
    "honorary fellow": "Award",
    "foreign fellow": "Award",
    "royal fellow": "Award",
    "order of chivalry": "Award",
    "rank of the order": "Award",
    "award": "Award",
    "prize": "Award",
    "medal": "Award",
    "report": "Work",
    "paper": "Work",
    "book": "Work",
    "proof": "Work",
    "device": "Machine",
    "machine": "Machine",
    "computer": "Machine",
    "school": "Organization",
    "university": "Organization",
    "college": "Organization",
    "agency": "Organization",
    "headquarters": "Organization",
    "museum": "Organization",
    "laboratory": "Organization",
    "district": "Place",
    "municipality": "Place",
    "borough": "Place",
    "civil parish": "Place",
    "town": "Place",
    "village": "Place",
    "city": "Place",
    "county town": "Place",
    "nursing home": "Place",
    "building": "Place",
    "theory": "Concept",
    "concept": "Concept",
    "test of": "Concept",
    "statistical technique": "Concept",
}


def infer_entity_type(
    instance_of_ids: list[str],
    instance_of_labels: list[str],
    expected_type: str | None,
    description_en: str | None = None,
    description_zh: str | None = None,
    label_en: str | None = None,
    label_zh: str | None = None,
) -> str:
    """按 Wikidata instance、关键词、人工期望类型依次推断实体类型。"""
    for instance_id in instance_of_ids:
        if instance_id in INSTANCE_TYPE_MAP:
            return INSTANCE_TYPE_MAP[instance_id]

    instance_hint_text = build_entity_type_hint_text(instance_of_labels=instance_of_labels)
    entity_type = match_keyword_type(instance_hint_text, ENTITY_KEYWORD_TYPE_HINTS)
    if entity_type is not None:
        return entity_type

    title_hint_text = build_entity_type_hint_text(label_en=label_en, label_zh=label_zh)
    entity_type = match_keyword_type(title_hint_text, TITLE_KEYWORD_TYPE_HINTS)
    if entity_type is not None:
        return entity_type

    description_hint_text = build_entity_type_hint_text(
        description_en=description_en,
        description_zh=description_zh,
    )
    entity_type = match_keyword_type(description_hint_text, DESCRIPTION_KEYWORD_TYPE_HINTS)
    if entity_type is not None:
        return entity_type

    if expected_type:
        return expected_type
    return "Concept"


def build_entity_type_hint_text(
    instance_of_labels: list[str] | None = None,
    description_en: str | None = None,
    description_zh: str | None = None,
    label_en: str | None = None,
    label_zh: str | None = None,
) -> str:
    instance_of_labels = instance_of_labels or []
    return " ".join(
        part.strip().lower()
        for part in (
            *instance_of_labels,
            description_en or "",
            description_zh or "",
            label_en or "",
            label_zh or "",
        )
        if isinstance(part, str) and part.strip()
    )


def match_keyword_type(hint_text: str, keyword_map: dict[str, str]) -> str | None:
    for keyword, entity_type in keyword_map.items():
        if keyword in hint_text:
            return entity_type
    return None
