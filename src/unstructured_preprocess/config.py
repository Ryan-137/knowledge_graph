from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """统一生成 UTC 时间戳，保持旧 documents/sentences 报告字段行为。"""

    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(text: str) -> str:
    """统一清理空白字符；HTML/PDF 正文抽取都依赖这个语义。"""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_json(file_path: Path, payload: Any) -> None:
    """统一 JSON 输出格式，避免不同流程各写一套。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(file_path: Path) -> Any:
    """统一读取 JSON，避免各流水线重复处理编码和 loads。"""

    return json.loads(file_path.read_text(encoding="utf-8"))


def write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    """统一写出 JSONL 记录流，并强制每行都是一个对象记录。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_lines: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"JSONL 写入失败：第 {index} 条记录不是对象，而是 {type(record).__name__}")
        serialized_lines.append(json.dumps(record, ensure_ascii=False))

    content = "\n".join(serialized_lines)
    if serialized_lines:
        content += "\n"
    file_path.write_text(content, encoding="utf-8")


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """统一读取 JSONL 记录流，并在逐行解析时校验对象结构。"""

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue

        try:
            payload = json.loads(stripped_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path.as_posix()} 第 {line_number} 行不是合法 JSON: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"{file_path.as_posix()} 第 {line_number} 行不是对象记录")
        records.append(payload)

    return records
