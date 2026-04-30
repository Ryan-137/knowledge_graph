from __future__ import annotations

from pathlib import Path

from ..config import normalize_whitespace


def extract_pdf_text(file_path: Path) -> str:
    """抽取 PDF 每页文本，并沿用旧流程的页内空白归一化和页间空行拼接。"""

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
        raise RuntimeError(
            "检测到 PDF 来源，但当前环境缺少 pypdf。请先安装 requirements.txt 中的依赖。"
        ) from exc

    reader = PdfReader(str(file_path))
    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = normalize_whitespace(text)
        if text:
            page_texts.append(text)
    return "\n\n".join(page_texts).strip()
