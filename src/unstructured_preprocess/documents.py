from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import utc_now_iso, write_json, write_jsonl
from .extractors.html import extract_html_text
from .extractors.pdf import extract_pdf_text
from .source_registry import load_sources


def infer_language(text: str) -> str:
    """按旧逻辑粗分中文、英文和未知语言。"""

    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    alpha_count = sum(1 for char in text if char.isalpha())
    if chinese_count > 0 and chinese_count >= alpha_count * 0.2:
        return "zh"
    if alpha_count > 0:
        return "en"
    return "unknown"


def extract_text(file_path: Path, source_type: str) -> str:
    """根据来源类型分发到对应正文抽取器。"""

    normalized_type = source_type.lower()
    if normalized_type == "html":
        return extract_html_text(file_path)
    if normalized_type == "pdf":
        return extract_pdf_text(file_path)
    raise ValueError(f"暂不支持的 source_type: {source_type}")


def build_documents(repo_root: Path, config_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构建 documents.jsonl 记录，字段与旧 preprocess_unstructured 输出保持一致。"""

    sources = load_sources(config_path)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        raw_file = repo_root / source.raw_path
        if not raw_file.exists():
            errors.append(
                {
                    "source_id": source.source_id,
                    "raw_path": source.raw_path,
                    "error": "原始文件不存在",
                }
            )
            continue

        try:
            clean_text = extract_text(raw_file, source.source_type)
            if not clean_text:
                raise ValueError("正文提取结果为空")

            documents.append(
                {
                    "doc_id": f"doc_{index:04d}",
                    "source_id": source.source_id,
                    "title": source.title,
                    "tier": source.tier,
                    "authority_level": source.authority_level,
                    "source_type": source.source_type,
                    "original_url": source.original_url,
                    "raw_path": source.raw_path,
                    "organization": source.organization,
                    "verification_status": source.verification_status,
                    "language": infer_language(clean_text),
                    "clean_text": clean_text,
                    "char_count": len(clean_text),
                    "paragraph_count": len([part for part in clean_text.split("\n\n") if part.strip()]),
                    "processed_at": utc_now_iso(),
                    "notes": source.notes,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 这里需要保留原始错误信息
            errors.append(
                {
                    "source_id": source.source_id,
                    "raw_path": source.raw_path,
                    "error": str(exc),
                }
            )

    return documents, errors


def run_document_preprocess(
    repo_root: Path,
    config_path: Path,
    output_path: Path,
    report_path: Path,
    strict: bool = False,
) -> tuple[int, int]:
    """执行文档级预处理，并写出 documents.jsonl 与报告。"""

    documents, errors = build_documents(repo_root=repo_root, config_path=config_path)

    write_jsonl(output_path, documents)
    write_json(
        report_path,
        {
            "generated_at": utc_now_iso(),
            "document_count": len(documents),
            "error_count": len(errors),
            "errors": errors,
        },
    )

    if strict and errors:
        raise RuntimeError(
            f"文档级预处理存在 {len(errors)} 个错误，请先查看 {report_path.as_posix()}。"
        )

    return len(documents), len(errors)
