from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(file_path: str | Path) -> Any:
    path = Path(file_path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(file_path: str | Path, payload: Any) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path.as_posix()} 第 {line_number} 行不是 JSON 对象")
        records.append(record)
    return records


def write_jsonl(file_path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"第 {index} 条记录不是对象，无法写入 JSONL")
        lines.append(json.dumps(record, ensure_ascii=False))
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def read_csv_records(file_path: str | Path) -> list[dict[str, str]]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_records(file_path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
