from __future__ import annotations

import re

# 文档块按“空行”拆分，保证句子切分前保留原始段落结构。
BLOCK_PATTERN = re.compile(r"[^\n]+(?:\n(?!\n)[^\n]+)*", flags=re.MULTILINE)

# 行级拆分用于处理 PDF 断行、目录块和标题块。
LINE_PATTERN = re.compile(r"[^\n]+")

# 标题常带数字前缀，句子归一化时需要去掉。
SECTION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:chapter|section|part)?\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+)[.)]?\s+",
    flags=re.IGNORECASE,
)

# 用于识别目录块中的编号项。
TOC_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")

# 常见缩写，避免把句号误切成句子边界。
ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "al",
    "no",
    "vol",
    "pp",
    "inc",
    "jr",
    "sr",
    "u.s",
    "u.k",
}

MONTH_MAP = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

