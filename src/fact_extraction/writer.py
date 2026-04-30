from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from kg_core.io import write_jsonl


def write_fact_jsonl(output_path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """事实层统一 JSONL 写入口。"""

    write_jsonl(output_path, records)
