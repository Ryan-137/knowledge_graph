from __future__ import annotations

from typing import Any


def mentions_by_sentence(resolved_mentions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 sentence_id 聚合已解析 mention，供模板抽取快速取论元。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for mention in resolved_mentions:
        sentence_id = str(mention.get("sentence_id") or "")
        if not sentence_id:
            continue
        grouped.setdefault(sentence_id, []).append(mention)
    return grouped


def relation_candidate_ids_by_sentence(relation_candidates: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for candidate in relation_candidates:
        sentence_id = str(candidate.get("sentence_id") or "")
        candidate_id = str(candidate.get("candidate_id") or candidate.get("claim_candidate_id") or "")
        if sentence_id and candidate_id:
            grouped.setdefault(sentence_id, []).append(candidate_id)
    return grouped
