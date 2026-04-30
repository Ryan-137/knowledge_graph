from __future__ import annotations

import re
from pathlib import Path

try:
    from bs4 import BeautifulSoup, Comment, Tag
except ImportError as exc:  # pragma: no cover - 依赖缺失时直接报错
    raise RuntimeError(
        "缺少 beautifulsoup4，请先安装 requirements.txt 中的依赖。"
    ) from exc

from configs.rules.document import (
    BLOCK_TAGS,
    CONTENT_SELECTORS,
    NOISE_KEYWORDS,
    NOISE_SELECTORS,
    NOISE_TAGS,
    PROTECTED_CONTENT_TAGS,
    TAIL_SECTION_TITLES,
)
from ..config import normalize_whitespace


LEADING_REFERENCE_PATTERN = re.compile(
    r"^(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:[A-Z\"'“‘(]|\Z))"
)
TRAILING_REFERENCE_PATTERN = re.compile(
    r"([,.;:!?%)\]}\"'”’])\s*(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:\s|$))"
)
PURE_REFERENCE_BLOCK_PATTERN = re.compile(r"^(?:\[\s*\d+(?:\s*[-,–]\s*\d+)?\s*\]\s*)+$")
TOC_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")


def _normalize_line(text: str) -> str:
    text = normalize_whitespace(text)
    text = LEADING_REFERENCE_PATTERN.sub("", text)
    text = TRAILING_REFERENCE_PATTERN.sub(r"\1", text)
    text = re.sub(r"\s+([,.;:!?%)\]}\u201d\u2019])", r"\1", text)
    text = re.sub(r"([(\[{\u201c\u2018])\s+", r"\1", text)
    text = text.strip(" -|")
    return text


def _contains_noise_keyword(value: str) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in NOISE_KEYWORDS)


def _is_visually_hidden(tag: Tag) -> bool:
    style = tag.get("style")
    if not isinstance(style, str):
        return False

    normalized_style = re.sub(r"\s+", "", style.lower())
    hidden_tokens = (
        "display:none",
        "visibility:hidden",
        "opacity:0",
    )
    return any(token in normalized_style for token in hidden_tokens)


def _looks_like_content_container(tag: Tag) -> bool:
    if tag.name in PROTECTED_CONTENT_TAGS:
        return True

    paragraph_count = len(tag.find_all("p"))
    heading_count = len(tag.find_all(("h1", "h2", "h3")))
    return paragraph_count >= 3 or (paragraph_count >= 1 and heading_count >= 1)


def _prune_html_noise(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()

    for selector in NOISE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    for tag in soup.find_all(True):
        # BeautifulSoup 在父节点被 decompose 后，预先缓存的子节点可能仍会被遍历到。
        if getattr(tag, "attrs", None) is None:
            continue

        attributes = []
        for attr_name in ("id", "class", "role", "aria-label"):
            attr_value = tag.get(attr_name)
            if isinstance(attr_value, list):
                attributes.extend(
                    value for value in attr_value if isinstance(value, str) and value.strip()
                )
            elif isinstance(attr_value, str):
                attributes.append(attr_value)

        if any(_contains_noise_keyword(value) for value in attributes):
            if _looks_like_content_container(tag):
                continue
            tag.decompose()
            continue

        if (
            tag.get("hidden") is not None
            or tag.get("aria-hidden") == "true"
            or _is_visually_hidden(tag)
        ):
            tag.decompose()


def _normalize_heading_text(text: str) -> str:
    text = normalize_whitespace(text).lower()
    text = re.sub(r"^[^a-z\u4e00-\u9fff]+|[^a-z\u4e00-\u9fff]+$", "", text)
    return text


def _looks_like_toc_block(text: str) -> bool:
    toc_token_count = len(TOC_TOKEN_PATTERN.findall(text))
    if toc_token_count >= 2:
        punctuation_count = len(re.findall(r"[.!?;:,，。！？；：]", text))
        return punctuation_count <= 2

    if re.match(r"^\d+(?:\.\d+)*\s+\S", text):
        return len(text) <= 160 and not re.search(r"[.!?;:,，。！？；：]", text)

    return False


def _is_noise_block(text: str) -> bool:
    if not text:
        return True

    if PURE_REFERENCE_BLOCK_PATTERN.fullmatch(text):
        return True

    if _looks_like_toc_block(text):
        return True

    if text.startswith(("Jump up to:", "Archived from the original", "Retrieved ")):
        return True

    return False


def _extract_blocks(container: Tag) -> list[str]:
    blocks: list[str] = []
    for tag in container.find_all(BLOCK_TAGS):
        text = _normalize_line(tag.get_text(" ", strip=True))
        if not text:
            continue
        if _is_noise_block(text):
            continue
        if tag.name == "li" and len(text) < 24:
            continue
        blocks.append(text)

    deduped: list[str] = []
    for text in blocks:
        if not deduped or deduped[-1] != text:
            deduped.append(text)
    return deduped


def _trim_trailing_reference_sections(blocks: list[str]) -> list[str]:
    for index, block in enumerate(blocks):
        normalized_block = _normalize_heading_text(block)
        if normalized_block not in TAIL_SECTION_TITLES:
            continue

        if index < 2:
            continue

        preceding_blocks = blocks[:index]
        has_meaningful_body = any(
            len(item) >= 40 or re.search(r"[.!?;:,，。！？；：]", item)
            for item in preceding_blocks
        )
        if not has_meaningful_body:
            continue

        # 尾部出现参考文献、外链等章节时，后续内容基本不再属于正文。
        return blocks[:index]

    return blocks


def _trim_leading_blocks_for_structured_article(blocks: list[str]) -> list[str]:
    numbered_heading_indexes = [
        index
        for index, block in enumerate(blocks)
        if re.match(r"^\d+\.\s+\S", block)
    ]
    if len(numbered_heading_indexes) < 5:
        return blocks

    first_numbered_heading_index = numbered_heading_indexes[0]
    if first_numbered_heading_index < 8:
        return blocks

    # 当前导块过多且正文已经呈现出稳定的编号章节结构时，应从正文首个编号章节开始截断。
    return blocks[first_numbered_heading_index:]


def _score_blocks(blocks: list[str]) -> int:
    if not blocks:
        return 0
    text = "\n".join(blocks)
    punctuation_bonus = len(re.findall(r"[.!?;:,，。！？；：]", text))
    long_block_bonus = sum(1 for block in blocks if len(block) >= 80) * 8
    return len(text) + punctuation_bonus * 6 + long_block_bonus


def extract_html_text(file_path: Path) -> str:
    """抽取 HTML 正文，保留旧流程的噪声剪枝、候选容器打分和尾部裁剪策略。"""

    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")
    _prune_html_noise(soup)

    candidates: list[Tag] = []
    for selector in CONTENT_SELECTORS:
        candidates.extend(soup.select(selector))

    body = soup.body
    if body is not None:
        candidates.append(body)

    seen: set[int] = set()
    unique_candidates: list[Tag] = []
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id not in seen:
            seen.add(candidate_id)
            unique_candidates.append(candidate)

    best_text = ""
    best_score = -1
    for candidate in unique_candidates:
        blocks = _extract_blocks(candidate)
        blocks = _trim_trailing_reference_sections(blocks)
        blocks = _trim_leading_blocks_for_structured_article(blocks)
        score = _score_blocks(blocks)
        if score > best_score:
            best_score = score
            best_text = "\n\n".join(blocks)

    if not best_text:
        fallback_text = normalize_whitespace(soup.get_text("\n"))
        return fallback_text

    return normalize_whitespace(best_text)
