from __future__ import annotations

from kg_core.entity_catalog import EntityCatalog

from .features import safe_float
from .models import EntityProfile


def build_entity_profiles(entity_catalog: EntityCatalog) -> dict[str, EntityProfile]:
    profiles: dict[str, EntityProfile] = {}
    for entity_id, row in entity_catalog.entities.items():
        alias_surfaces = entity_catalog.alias_surface_index_by_entity.get(entity_id, [])
        alias_values = []
        seen: set[str] = set()
        for surface in alias_surfaces:
            value = str(surface.get("surface") or "").strip()
            if not value or value.casefold() in seen:
                continue
            seen.add(value.casefold())
            alias_values.append(value)
        description = " ".join(
            value
            for value in [
                str(row.get("description_en") or "").strip(),
                str(row.get("description_zh") or "").strip(),
                str(row.get("wikipedia_summary_en") or "").strip(),
            ]
            if value
        )
        profiles[entity_id] = EntityProfile(
            entity_id=entity_id,
            canonical_name=str(row.get("canonical_name") or "").strip(),
            entity_type=str(row.get("entity_type") or "").strip(),
            aliases=alias_values,
            description=description,
            source_name=str(row.get("source_name") or "").strip() or None,
            confidence=safe_float(row.get("confidence"), default=0.75),
        )
    return profiles
