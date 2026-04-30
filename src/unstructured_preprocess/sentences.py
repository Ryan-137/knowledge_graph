from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from kg_core.schemas import DocumentRecord, SentenceRecord, TimeMentionRecord

from .config import read_jsonl, utc_now_iso, write_json, write_jsonl
from .sentence_splitter import split_document_sentences
from .time_mentions import extract_time_mentions


def load_documents(documents_path: Path) -> list[DocumentRecord]:
    """从 documents.jsonl 读取句子预处理所需字段。"""

    payload = read_jsonl(documents_path)
    documents: list[DocumentRecord] = []
    for item in payload:
        documents.append(
            DocumentRecord(
                doc_id=item["doc_id"],
                source_id=item["source_id"],
                title=item.get("title", ""),
                tier=int(item.get("tier", 0)),
                language=item.get("language", "unknown"),
                clean_text=item.get("clean_text", ""),
            )
        )
    return documents


def build_sentences(
    documents_path: Path,
) -> tuple[list[SentenceRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    """构建 sentences.jsonl 记录，字段与旧 preprocess_unstructured 输出保持一致。"""

    documents = load_documents(documents_path)
    sentences: list[SentenceRecord] = []
    errors: list[dict[str, Any]] = []
    doc_sentence_counts: list[dict[str, Any]] = []
    sentence_counter = 1

    for document in documents:
        if not document.clean_text.strip():
            errors.append(
                {
                    "doc_id": document.doc_id,
                    "source_id": document.source_id,
                    "error": "clean_text 为空，无法进行句子级预处理",
                }
            )
            continue

        sentence_index_in_doc = 1
        for sentence_text, absolute_start, absolute_end in split_document_sentences(document.clean_text):
            time_mentions = extract_time_mentions(sentence_text)
            sentences.append(
                SentenceRecord(
                    sentence_id=f"sent_{sentence_counter:06d}",
                    doc_id=document.doc_id,
                    source_id=document.source_id,
                    sentence_index_in_doc=sentence_index_in_doc,
                    text=sentence_text,
                    offset_start=absolute_start,
                    offset_end=absolute_end,
                    normalized_time=[item.normalized for item in time_mentions],
                    time_mentions=time_mentions,
                )
            )
            sentence_counter += 1
            sentence_index_in_doc += 1

        doc_sentence_count = sentence_index_in_doc - 1
        doc_sentence_counts.append(
            {
                "doc_id": document.doc_id,
                "source_id": document.source_id,
                "sentence_count": doc_sentence_count,
            }
        )

        if doc_sentence_count == 0:
            errors.append(
                {
                    "doc_id": document.doc_id,
                    "source_id": document.source_id,
                    "error": "句子切分结果为空",
                }
            )

    return sentences, doc_sentence_counts, errors


def run_sentence_preprocess(
    documents_path: Path,
    output_path: Path,
    report_path: Path,
    strict: bool = False,
) -> tuple[int, int]:
    """执行句子级预处理，并写出 sentences.jsonl 与报告。"""

    sentences, doc_sentence_counts, errors = build_sentences(documents_path=documents_path)
    write_jsonl(output_path, [asdict(sentence) for sentence in sentences])
    write_json(
        report_path,
        {
            "generated_at": utc_now_iso(),
            "document_count": len(doc_sentence_counts),
            "sentence_count": len(sentences),
            "error_count": len(errors),
            "doc_sentence_counts": doc_sentence_counts,
            "errors": errors,
        },
    )

    if strict and errors:
        raise RuntimeError(
            f"句子级预处理存在 {len(errors)} 个错误，请先查看 {report_path.as_posix()}。"
        )

    return len(sentences), len(errors)
