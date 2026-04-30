from __future__ import annotations

import re
from dataclasses import dataclass

from configs.rules.sentence import (
    ABBREVIATIONS,
    BLOCK_PATTERN,
    LINE_PATTERN,
    SECTION_PREFIX_PATTERN,
    TOC_TOKEN_PATTERN,
)


TRAILING_CLOSERS = "\"'”’)]}"
COMMON_SINGLE_LETTER_WORDS = {"a", "A", "i", "I"}
COMMON_SINGLE_LETTER_SYMBOLS = {"A", "B", "C", "I", "Q", "X", "Y"}
SENTENCE_ENDING_PATTERN = re.compile(r"[.!?;:。！？；：][\"'”’)\]}]*$")
PURE_REFERENCE_SENTENCE_PATTERN = re.compile(r"^(?:\[\s*\d+(?:\s*[-,–]\s*\d+)?\s*\]\s*)+$")
LEADING_REFERENCE_PATTERN = re.compile(
    r"^(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:[A-Z\"'“‘(]|\Z))"
)
TRAILING_REFERENCE_PATTERN = re.compile(
    r"([,.;:!?%)\]}\"'”’])\s*(?:\[\s*\d+(?:\s*[-–]\s*\d+)?\s*\]\s*)+(?=(?:\s|$))"
)


@dataclass(frozen=True)
class NormalizedSegment:
    """保存归一化后的段落文本，以及每个字符对应的原始绝对偏移。"""

    text: str
    offsets: list[int]


def _iter_blocks(text: str) -> list[tuple[str, int, int]]:
    blocks: list[tuple[str, int, int]] = []
    for match in BLOCK_PATTERN.finditer(text):
        blocks.append((match.group(0), match.start(), match.end()))
    return blocks


def _looks_like_toc_block(block_text: str) -> bool:
    section_token_count = len(TOC_TOKEN_PATTERN.findall(block_text))
    if section_token_count < 2:
        return False

    punctuation_count = len(re.findall(r"[.!?;:。！？；：]", block_text))
    return punctuation_count <= 2


def _split_toc_block(block_text: str, block_start: int) -> list[tuple[str, int, int]]:
    matches = list(re.finditer(r"(?:^|\s)(\d+(?:\.\d+)+|\d+)\s+[A-Z]", block_text))
    if len(matches) < 2:
        return [(block_text, block_start, block_start + len(block_text))]

    segments: list[tuple[str, int, int]] = []
    starts = [match.start(1) for match in matches]
    starts.append(len(block_text))
    for index in range(len(starts) - 1):
        segment_start = starts[index]
        segment_end = starts[index + 1]
        segment_text = block_text[segment_start:segment_end].strip()
        if not segment_text:
            continue
        absolute_start = block_start + segment_start
        segments.append((segment_text, absolute_start, absolute_start + len(segment_text)))
    return segments


def _split_block_into_segments(block_text: str, block_start: int) -> list[tuple[str, int, int]]:
    if _looks_like_toc_block(block_text):
        return _split_toc_block(block_text, block_start)

    lines = [(match.group(0), match.start(), match.end()) for match in LINE_PATTERN.finditer(block_text)]
    meaningful_lines = [line for line in lines if line[0].strip()]
    if len(meaningful_lines) <= 1:
        return [(block_text, block_start, block_start + len(block_text))]

    soft_wrap_count = 0
    for index in range(len(meaningful_lines) - 1):
        current_line = meaningful_lines[index][0].strip()
        next_line = meaningful_lines[index + 1][0].strip()
        if not current_line or not next_line:
            continue

        current_tail = current_line[-1]
        next_head = next_line[0]
        if current_tail in ",-–(":
            soft_wrap_count += 1
            continue

        if current_tail not in ".!?;:。！？；：" and next_head.islower():
            soft_wrap_count += 1

    if soft_wrap_count >= max(1, len(meaningful_lines) // 2):
        return [(block_text, block_start, block_start + len(block_text))]

    average_length = sum(len(line[0]) for line in meaningful_lines) / len(meaningful_lines)
    if average_length > 80:
        return [(block_text, block_start, block_start + len(block_text))]

    # PDF 常把“章节标题 + 正文首句”压在同一个 block 里，先剥离前导标题行。
    leading_heading_count = 0
    has_wrapped_prefix = False
    for index, (line_text, _, relative_end) in enumerate(meaningful_lines):
        if not _looks_like_heading_block(line_text):
            break
        next_line_text = meaningful_lines[index + 1][0] if index + 1 < len(meaningful_lines) else ""
        separator_text = (
            block_text[relative_end : meaningful_lines[index + 1][1]]
            if index + 1 < len(meaningful_lines)
            else ""
        )
        if _looks_like_wrapped_prefix(line_text, next_line_text, separator_text):
            has_wrapped_prefix = True
            break
        leading_heading_count += 1

    if has_wrapped_prefix:
        return [(block_text, block_start, block_start + len(block_text))]

    if 0 < leading_heading_count < len(meaningful_lines):
        segments: list[tuple[str, int, int]] = []
        for line_text, relative_start, relative_end in meaningful_lines[:leading_heading_count]:
            absolute_start = block_start + relative_start
            segments.append((line_text, absolute_start, block_start + relative_end))

        body_start = meaningful_lines[leading_heading_count][1]
        body_end = meaningful_lines[-1][2]
        segments.append((block_text[body_start:body_end], block_start + body_start, block_start + body_end))
        return segments

    segments: list[tuple[str, int, int]] = []
    for line_text, relative_start, relative_end in meaningful_lines:
        absolute_start = block_start + relative_start
        segments.append((line_text, absolute_start, block_start + relative_end))
    return segments


def _looks_like_heading_block(block_text: str) -> bool:
    normalized_text = re.sub(r"\s+", " ", block_text).strip()
    normalized_text = SECTION_PREFIX_PATTERN.sub("", normalized_text).strip()
    if not normalized_text:
        return False

    if len(normalized_text) > 100:
        return False

    if re.search(r"[.!?;:。！？；：]", normalized_text):
        return False

    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff&+'’-]+", normalized_text)
    if not words or len(words) > 12:
        return False

    # 标题通常以短语形态出现，这里故意放宽判断，宁可少吃正文句子。
    return True


def _looks_like_wrapped_prefix(line_text: str, next_line_text: str, separator_text: str) -> bool:
    """
    识别被 PDF 强行断开的句首片段，例如 “Alan\n Mathison ...”。

    这类文本在视觉上像标题 + 正文，但其实只是一个句子的前缀，不能被当成 heading 拆开。
    """

    if not next_line_text or "\n" not in separator_text:
        return False

    words = re.findall(r"[A-Za-z0-9\u4e00-\u9fff&+'’-]+", line_text.strip())
    # 只有换行本身说明是正常新行；换行后仍残留空白，通常是 PDF 强制折行。
    return len(words) == 1 and bool(separator_text.replace("\n", ""))


def _collapse_whitespace_with_offsets(text: str, absolute_start: int) -> NormalizedSegment:
    chars: list[str] = []
    offsets: list[int] = []
    pending_space = False
    pending_space_offset = absolute_start

    for relative_index, char in enumerate(text):
        absolute_index = absolute_start + relative_index
        if char.isspace():
            if chars and not pending_space:
                pending_space = True
                pending_space_offset = absolute_index
            continue

        if pending_space:
            previous_char = chars[-1] if chars else ""
            if previous_char not in "\"'“‘([{" and char not in ",.;:!?%)]}\"'”’":
                chars.append(" ")
                offsets.append(pending_space_offset)
            pending_space = False

        chars.append(char)
        offsets.append(absolute_index)

    return NormalizedSegment(text="".join(chars), offsets=offsets)


def _next_significant_char(text: str, start_index: int) -> str:
    cursor = start_index
    while cursor < len(text):
        char = text[cursor]
        if char.isspace() or char in TRAILING_CLOSERS:
            cursor += 1
            continue
        return char
    return ""


def _last_ascii_token(text: str, end_index: int) -> str:
    window = text[max(0, end_index - 24) : end_index + 1]
    match = re.search(r"([A-Za-z][A-Za-z.\-]{0,23})[.!?;:]?$", window)
    if not match:
        return ""
    return match.group(1).strip(".").lower()


def _consume_trailing_closers(text: str, index: int) -> int:
    cursor = index
    while cursor + 1 < len(text) and text[cursor + 1] in TRAILING_CLOSERS:
        cursor += 1
    return cursor


def _is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char in "。！？；!?;":
        return True

    if char != ".":
        return False

    previous_char = text[index - 1] if index > 0 else ""
    next_char = text[index + 1] if index + 1 < len(text) else ""

    if previous_char.isdigit() and next_char.isdigit():
        return False

    if previous_char == "." or next_char == ".":
        return False

    token = _last_ascii_token(text, index)
    if token in ABBREVIATIONS:
        return False

    next_significant_char = _next_significant_char(text, index + 1)
    if not next_significant_char:
        return True

    if previous_char.isupper() and next_significant_char.isupper():
        return False

    if re.match(r"\s*[A-Za-z]\.", text[index + 1 :]):
        return False

    if next_significant_char.islower():
        return False

    return True


def _normalize_sentence_text(text: str) -> str:
    text = SECTION_PREFIX_PATTERN.sub("", text).strip()
    text = LEADING_REFERENCE_PATTERN.sub("", text)
    text = TRAILING_REFERENCE_PATTERN.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text)
    # 一些 PDF 提取会把整段引文前后的空格吃掉。
    # 这里按“完整引号块”补空格，避免把 opening / closing quote 分别修坏。
    text = re.sub(r"(?<=\w)(\"[^\"]+\")", r" \1", text)
    text = re.sub(r"(\"[^\"]+\")(?=\w)", r"\1 ", text)
    text = re.sub(r"([,.;:!?])(\"[^\"]+\")", r"\1 \2", text)
    text = re.sub(r"([“‘([{])\s+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?%)\]}”’])", r"\1", text)
    text = re.sub(r"\s+([\"'])(?=\s|$|[,.;:!?%)\]}])", r"\1", text)

    tokens = text.split(" ")
    merged_tokens: list[str] = []
    index = 0
    while index < len(tokens):
        current_token = tokens[index]
        if index + 1 < len(tokens) and re.fullmatch(r"[A-Za-z]", current_token):
            next_token = tokens[index + 1]
            if re.fullmatch(r"[a-z]+", next_token):
                if current_token.islower() and current_token not in COMMON_SINGLE_LETTER_WORDS:
                    merged_tokens.append(current_token + next_token)
                    index += 2
                    continue

                if current_token.isupper() and current_token not in COMMON_SINGLE_LETTER_SYMBOLS:
                    if len(next_token) == 1 or len(next_token) >= 4:
                        merged_tokens.append(current_token + next_token)
                        index += 2
                        continue

        merged_tokens.append(current_token)
        index += 1

    text = " ".join(token for token in merged_tokens if token)
    return text.strip()


def _complete_heading_sentence(text: str) -> str:
    if not text:
        return text

    if SENTENCE_ENDING_PATTERN.search(text):
        return text

    if re.search(r"[\u4e00-\u9fff]", text):
        return f"{text}。"
    return f"{text}."


def _extract_heading_piece(block_text: str, block_start: int, block_end: int) -> tuple[str, int, int] | None:
    left_trim = len(block_text) - len(block_text.lstrip())
    right_trim = len(block_text) - len(block_text.rstrip())
    trimmed_text = block_text.strip()
    if not trimmed_text:
        return None

    absolute_start = block_start + left_trim
    absolute_end = block_end - right_trim

    prefix_match = SECTION_PREFIX_PATTERN.match(trimmed_text)
    if prefix_match:
        prefix_length = prefix_match.end()
        trimmed_text = trimmed_text[prefix_length:]
        absolute_start += prefix_length

    normalized_text = _normalize_sentence_text(trimmed_text)
    if not normalized_text:
        return None

    return normalized_text, absolute_start, absolute_end


def _build_heading_sentence(heading_blocks: list[tuple[str, int, int]]) -> tuple[str, int, int] | None:
    heading_parts: list[str] = []
    absolute_start: int | None = None
    absolute_end: int | None = None

    for block_text, block_start, block_end in heading_blocks:
        heading_piece = _extract_heading_piece(block_text, block_start, block_end)
        if heading_piece is None:
            continue

        piece_text, piece_start, piece_end = heading_piece
        heading_parts.append(piece_text)
        if absolute_start is None:
            absolute_start = piece_start
        absolute_end = piece_end

    if not heading_parts or absolute_start is None or absolute_end is None:
        return None

    heading_text = _normalize_sentence_text(" ".join(heading_parts))
    heading_text = _complete_heading_sentence(heading_text)
    if _is_discardable_sentence(heading_text):
        return None

    return heading_text, absolute_start, absolute_end


def _is_discardable_sentence(text: str) -> bool:
    if not text or len(text) < 4:
        return True

    if PURE_REFERENCE_SENTENCE_PATTERN.fullmatch(text):
        return True

    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text):
        return True

    if re.fullmatch(r"(?:\d+(?:\.\d+)+|\d+)", text):
        return True

    if len(TOC_TOKEN_PATTERN.findall(text)) >= 3 and not re.search(r"[.!?;:。！？；：]", text):
        return True

    if text.startswith(("Retrieved ", "Archived from the original", "^")):
        return True

    if text.startswith(("Author:", "Useful Links", "Related articles", "Other on-line publications")):
        return True

    if len(text) <= 80 and any(marker in text for marker in ("ISBN", "doi", "Vol.", "pp.", "BBC News", "The Guardian")):
        return True

    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    if not re.search(r"[.!?;:。！？；：]", text) and words:
        if len(text) <= 60 and len(words) <= 8 and all(word[:1].isupper() for word in words if word):
            return True

    if len(text) <= 36 and ":" in text and not re.search(r"\d{4}", text):
        return True

    if re.fullmatch(r"[A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*){1,3}", text):
        return True

    return False


def _split_segment_into_sentences(segment_text: str, segment_start: int) -> list[tuple[str, int, int]]:
    if "\n" in segment_text:
        line_segments = _split_block_into_segments(segment_text, segment_start)
        should_split_lines = len(line_segments) > 1 and any(
            _looks_like_heading_block(line_text) for line_text, _, _ in line_segments[:-1]
        )
        if should_split_lines:
            sentences: list[tuple[str, int, int]] = []
            for line_text, line_start, _ in line_segments:
                sentences.extend(_split_segment_into_sentences(line_text, line_start))
            return sentences

    normalized_segment = _collapse_whitespace_with_offsets(segment_text, segment_start)
    normalized_text = normalized_segment.text
    offsets = normalized_segment.offsets
    if not normalized_text:
        return []

    sentences: list[tuple[str, int, int]] = []
    sentence_start = 0
    index = 0

    while index < len(normalized_text):
        if not _is_sentence_boundary(normalized_text, index):
            index += 1
            continue

        boundary_end = _consume_trailing_closers(normalized_text, index)
        while sentence_start <= boundary_end and normalized_text[sentence_start].isspace():
            sentence_start += 1

        if sentence_start > boundary_end:
            sentence_start = boundary_end + 1
            index = boundary_end + 1
            continue

        sentence_text = _normalize_sentence_text(normalized_text[sentence_start : boundary_end + 1])
        absolute_start = offsets[sentence_start]
        absolute_end = offsets[boundary_end] + 1
        if not _is_discardable_sentence(sentence_text):
            sentences.append((sentence_text, absolute_start, absolute_end))

        sentence_start = boundary_end + 1
        index = boundary_end + 1

    while sentence_start < len(normalized_text) and normalized_text[sentence_start].isspace():
        sentence_start += 1

    if sentence_start < len(normalized_text):
        tail_end = len(normalized_text) - 1
        while tail_end >= sentence_start and normalized_text[tail_end].isspace():
            tail_end -= 1

        if tail_end >= sentence_start:
            sentence_text = _normalize_sentence_text(normalized_text[sentence_start : tail_end + 1])
            absolute_start = offsets[sentence_start]
            absolute_end = offsets[tail_end] + 1
            if _looks_like_heading_block(sentence_text):
                sentence_text = _complete_heading_sentence(sentence_text)
            if not _is_discardable_sentence(sentence_text):
                sentences.append((sentence_text, absolute_start, absolute_end))

    return sentences


def split_document_sentences(clean_text: str) -> list[tuple[str, int, int]]:
    """按旧句子流水线切分单篇文档，返回句子文本和 clean_text 内绝对偏移。"""

    sentences: list[tuple[str, int, int]] = []
    previous_sentence_text = ""

    blocks = _iter_blocks(clean_text)
    block_index = 0
    while block_index < len(blocks):
        block_text, block_start, _ = blocks[block_index]

        if _looks_like_heading_block(block_text):
            heading_end_index = block_index
            while heading_end_index + 1 < len(blocks) and _looks_like_heading_block(blocks[heading_end_index + 1][0]):
                heading_end_index += 1

            heading_sentence = _build_heading_sentence(blocks[block_index : heading_end_index + 1])
            if heading_sentence is not None:
                sentence_text, absolute_start, absolute_end = heading_sentence
                if sentence_text != previous_sentence_text:
                    sentences.append((sentence_text, absolute_start, absolute_end))
                    previous_sentence_text = sentence_text

            block_index = heading_end_index + 1
            continue

        for segment_text, segment_start, _ in _split_block_into_segments(block_text, block_start):
            for sentence_text, absolute_start, absolute_end in _split_segment_into_sentences(
                segment_text,
                segment_start,
            ):
                if sentence_text == previous_sentence_text:
                    continue

                sentences.append((sentence_text, absolute_start, absolute_end))
                previous_sentence_text = sentence_text
        block_index += 1

    return sentences
