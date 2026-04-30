from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class FetchConfig:
    """结构化种子库构建配置。"""

    database_path: Path
    csv_export_dir: Path
    log_path: Path
    seed_file: Path
    wikidata_endpoint: str
    wikipedia_summary_api: str
    sleep_seconds: float
    request_timeout_seconds: int
    max_retries: int
    backoff_base_seconds: float
    page_size: int
    entity_batch_size: int
    summary_batch_size: int
    user_agent: str
    default_confidence: float
    relations: list[str]

    @classmethod
    def load(cls, config_path: str | Path) -> "FetchConfig":
        """保留旧调用方式：FetchConfig.load(path)。"""
        return load_config(config_path)

    def ensure_directories(self) -> None:
        """在正式写入 SQLite、CSV、日志前保证目录存在。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_export_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def load_seed_entities(self) -> list[dict[str, Any]]:
        """保留旧调用方式：config.load_seed_entities()。"""
        return read_seed_entities(self.seed_file)


def load_config(config_path: str | Path) -> FetchConfig:
    """从 JSON 配置读取结构化种子库构建参数。"""
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parents[2].resolve()
    return FetchConfig(
        database_path=_resolve_path(base_dir, payload["database_path"]),
        csv_export_dir=_resolve_path(base_dir, payload["csv_export_dir"]),
        log_path=_resolve_path(base_dir, payload["log_path"]),
        seed_file=_resolve_path(base_dir, payload["seed_file"]),
        wikidata_endpoint=payload["wikidata_endpoint"],
        wikipedia_summary_api=payload["wikipedia_summary_api"],
        sleep_seconds=float(payload["sleep_seconds"]),
        request_timeout_seconds=int(payload["request_timeout_seconds"]),
        max_retries=int(payload["max_retries"]),
        backoff_base_seconds=float(payload["backoff_base_seconds"]),
        page_size=int(payload["page_size"]),
        entity_batch_size=int(payload["entity_batch_size"]),
        summary_batch_size=int(payload["summary_batch_size"]),
        user_agent=str(payload["user_agent"]),
        default_confidence=float(payload["default_confidence"]),
        relations=list(payload["relations"]),
    )


def read_seed_entities(seed_file: str | Path) -> list[dict[str, Any]]:
    """读取人工维护的种子实体名单。"""
    return json.loads(Path(seed_file).read_text(encoding="utf-8"))


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
