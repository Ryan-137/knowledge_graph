from __future__ import annotations

from pathlib import Path

from kg_core.schemas import SourceRecord

try:
    import yaml
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
    raise RuntimeError(
        "缺少 PyYAML，请先安装 requirements.txt 中的依赖。"
    ) from exc


def load_sources(config_path: Path) -> list[SourceRecord]:
    """读取非结构化来源清单，并转成强类型记录。"""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    items = payload.get("sources", [])
    sources: list[SourceRecord] = []
    for item in items:
        sources.append(
            SourceRecord(
                source_id=item["source_id"],
                title=item["title"],
                tier=int(item["tier"]),
                authority_level=item["authority_level"],
                source_type=item["source_type"],
                original_url=item.get("original_url", ""),
                raw_path=item["raw_path"],
                organization=item.get("organization", ""),
                verification_status=item.get("verification_status", ""),
                notes=item.get("notes", ""),
            )
        )
    return sources
