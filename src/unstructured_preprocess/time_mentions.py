from __future__ import annotations

import re

from kg_core.schemas import TimeMentionRecord

from configs.rules.sentence import MONTH_MAP


def _expand_two_digit_year(start_year: str, end_year: str) -> str:
    if len(end_year) == 4:
        return end_year
    century_prefix = start_year[:2]
    return f"{century_prefix}{end_year}"


def _append_time_mention(
    mentions: list[TimeMentionRecord],
    occupied: list[tuple[int, int]],
    start: int,
    end: int,
    text: str,
    normalized: str,
    mention_type: str,
) -> None:
    for existing_start, existing_end in occupied:
        if not (end <= existing_start or start >= existing_end):
            return

    occupied.append((start, end))
    mentions.append(
        TimeMentionRecord(
            text=text,
            normalized=normalized,
            type=mention_type,
            offset_start=start,
            offset_end=end,
        )
    )


def extract_time_mentions(sentence_text: str) -> list[TimeMentionRecord]:
    """抽取句内时间表达，保持旧流程的优先级和重叠去重规则。"""

    mentions: list[TimeMentionRecord] = []
    occupied: list[tuple[int, int]] = []

    for match in re.finditer(
        r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        day = f"{int(match.group('day')):02d}"
        normalized = f"{match.group('year')}-{month}-{day}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "date",
        )

    for match in re.finditer(
        r"\b(?P<day>\d{1,2})\s+(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s+(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        day = f"{int(match.group('day')):02d}"
        normalized = f"{match.group('year')}-{month}-{day}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "date",
        )

    for match in re.finditer(
        r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+(?P<year>\d{4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        month = MONTH_MAP[match.group("month").lower()]
        normalized = f"{match.group('year')}-{month}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "month_year",
        )

    for match in re.finditer(
        r"\b(?P<start>\d{4})\s*(?:-|–|/|to)\s*(?P<end>\d{2,4})\b",
        sentence_text,
        flags=re.IGNORECASE,
    ):
        end_year = _expand_two_digit_year(match.group("start"), match.group("end"))
        normalized = f"{match.group('start')}-{end_year}"
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            normalized,
            "year_range",
        )

    for match in re.finditer(r"\b(?:c\.|ca\.|circa)\s*(?P<year>\d{4})\b", sentence_text, flags=re.IGNORECASE):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group("year"),
            "circa_year",
        )

    for match in re.finditer(r"\b(?P<year>\d{4})s\b", sentence_text):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group(0),
            "decade",
        )

    for match in re.finditer(r"\b(1[5-9]\d{2}|20\d{2})\b", sentence_text):
        _append_time_mention(
            mentions,
            occupied,
            match.start(),
            match.end(),
            match.group(0),
            match.group(0),
            "year",
        )

    return mentions
