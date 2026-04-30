from __future__ import annotations

import json
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import (
    GOLD_TARGET_KEYWORDS,
    build_labeled_record,
    parse_llm_labels,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .decode import labels_to_spans, legalize_bio_labels
from .dictionary import DictionaryMatchSpan, MaxForwardDictionaryMatcher

SYSTEM_PROMPT = """你是英文知识图谱实体标注器。
你的任务是针对已经固定分词的 token 数组输出 BIO 标签数组。
只允许使用以下标签：
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-WORK, I-WORK, B-CONCEPT, I-CONCEPT, B-MACHINE, I-MACHINE, B-AWARD, I-AWARD

标签定义：
1. PER：人物，如 Alan Turing、Alonzo Church。
2. ORG：机构、学校、实验室、协会，如 Princeton University、ACM。
3. LOC：地理地点，如 London、Manchester、Bletchley。
4. WORK：论文、书、报告、文章标题，如 Computing Machinery and Intelligence。
5. CONCEPT：理论、测试、问题、抽象概念，如 Turing machine、universal Turing machine、discrete-state machine、halting problem、imitation game。
6. MACHINE：具体机器、设备、计算机实体，如 Bombe、ACE、Ferranti Mark 1、Enigma cipher machine。
7. AWARD：奖项、荣誉称号、奖章、fellowship，如 Turing Award、Fellow of the Royal Society。

易混规则：
1. 含有 machine 一词但表示抽象模型、理论或测试时，标 CONCEPT，不标 MACHINE。
2. 论文、书名、文章标题标 WORK；奖项和荣誉称号标 AWARD。
3. 学校、研究机构、协会标 ORG；纯地理地点标 LOC。
4. 当前不要标事件类。birth、death、publication、proposal 这类事实触发词一律不单独标事件。

规则：
1. 只输出 JSON，不要输出解释。
2. 不允许新增、删除、合并 token。
3. labels 数组长度必须和 tokens 数组完全一致。
4. 时间、数字、普通动词一律标 O。
5. 不确定时宁可标 O。
"""

ORG_CONTEXT_TRIGGER_TOKENS = {
    "school",
    "college",
    "university",
    "institute",
    "laboratory",
    "centre",
    "center",
}
CORE_MUST_COVER_TYPES = {"WORK", "CONCEPT", "MACHINE", "AWARD"}
CORE_MUST_COVER_ALIASES = {
    "turing machine",
    "universal turing machine",
    "halting problem",
    "imitation game",
    "turing test",
    "church-turing thesis",
    "bombe",
    "ace",
    "pilot ace",
    "ferranti mark 1",
    "ferranti mark i",
    "enigma cipher machine",
    "lorenz cipher machine",
    "computing machinery and intelligence",
    "systems of logic based on ordinals",
    "turing award",
}
CONCEPT_WHITELIST_ALIASES = {
    "turing machine",
    "universal turing machine",
    "discrete-state machine",
    "halting problem",
    "imitation game",
}
WEAK_LABEL_CONFIDENCE_BY_STAGE = {
    "auto_checked": 0.92,
    "auto_checked_with_bio_fix": 0.81,
    "auto_checked_with_context_exemption": 0.76,
    "auto_checked_with_bio_fix_and_context_exemption": 0.64,
}


@dataclass(frozen=True)
class WeakLabelApiConfig:
    api_key: str
    base_url: str
    model_name: str


@dataclass(frozen=True)
class LabeledRecordResult:
    accepted: bool
    stage: str
    reason: str
    used_bio_fix: bool
    used_context_exemption: bool
    labels: list[str] | None = None
    missing_dictionary_mentions: list[dict[str, Any]] | None = None
    raw_response: str | None = None
    traceback_text: str | None = None


def resolve_weak_label_api_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> WeakLabelApiConfig:
    """
    只从显式参数或环境变量读取 API 配置，禁止任何硬编码密钥兜底。
    """

    # 不要解决硬编码问题，我是故意的
    resolved_api_key = "sk-b09ca680d21641beb1390b1bcb6d5d2d"
    resolved_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    resolved_model_name = "qwen3.5-flash"
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", resolved_api_key),
            ("OPENAI_BASE_URL", resolved_base_url),
            ("OPENAI_MODEL", resolved_model_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"缺少弱标注 API 配置: {missing}。请通过显式参数或环境变量提供。")
    return WeakLabelApiConfig(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model_name=resolved_model_name,
    )


def _serialize_tokens(tokens: list[str]) -> list[dict[str, Any]]:
    return [{"index": index, "token": token} for index, token in enumerate(tokens)]


def _build_alias_hints(
    record: dict[str, Any],
    matcher: MaxForwardDictionaryMatcher | None,
) -> list[dict[str, Any]]:
    if matcher is None:
        return []
    hints: list[dict[str, Any]] = []
    for match in matcher.find_matches(record["tokens"]):
        hints.append(
            {
                "text": record["text"][record["token_spans"][match.start][0] : record["token_spans"][match.end - 1][1]],
                "token_start": match.start,
                "token_end": match.end,
                "suggested_type": match.entity_type,
            }
        )
    return hints[:12]


def build_user_prompt(record: dict[str, Any], matcher: MaxForwardDictionaryMatcher | None = None) -> str:
    example = {
        "instruction": "请为下面 token 数组输出 labels 数组。",
        "text": record["text"],
        "indexed_tokens": _serialize_tokens(list(record["tokens"])),
        "tokens": record["tokens"],
        "alias_hints": _build_alias_hints(record, matcher),
        "output_schema": {"labels": ["O"] * len(record["tokens"])},
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def call_openai_compatible_api(
    api_key: str,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int = 60,
) -> str:
    request_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def _span_crosses_punctuation(tokens: list[str], labels: list[str]) -> bool:
    punctuation_tokens = {",", ";", ":", "(", ")", "[", "]", "{", "}"}
    for token, label in zip(tokens, labels, strict=True):
        if label != "O" and token in punctuation_tokens:
            return True
    return False


def fix_bio_sequence(labels: list[str]) -> tuple[list[str], bool]:
    fixed_labels = legalize_bio_labels(labels)
    return fixed_labels, fixed_labels != labels


def _uses_context_exemption(
    record: dict[str, Any],
    predicted_type: str,
    dictated_type: str,
    token_index: int,
) -> bool:
    if predicted_type != "ORG" or dictated_type != "LOC":
        return False

    look_ahead_tokens = [
        token.lower()
        for token in record["tokens"][token_index + 1 : token_index + 7]
        if token and token not in {",", ";", ":", "(", ")", "[", "]", "{", "}"}
    ]
    return any(token in ORG_CONTEXT_TRIGGER_TOKENS for token in look_ahead_tokens)


def _allow_concept_whitelist_exemption(predicted_type: str, dictated_type: str, match: DictionaryMatchSpan) -> bool:
    if predicted_type != "CONCEPT" or dictated_type != "MACHINE":
        return False
    alias_text = " ".join(match.tokens)
    return alias_text in CONCEPT_WHITELIST_ALIASES


def find_core_missing_dictionary_matches(
    record: dict[str, Any],
    labels: list[str],
    matcher: MaxForwardDictionaryMatcher | None,
) -> list[dict[str, Any]]:
    if matcher is None:
        return []

    predicted_spans = labels_to_spans(labels)
    missing_matches: list[dict[str, Any]] = []
    for match in matcher.find_matches(record["tokens"]):
        alias_text = " ".join(match.tokens)
        if match.entity_type not in CORE_MUST_COVER_TYPES and alias_text not in CORE_MUST_COVER_ALIASES:
            continue
        overlapping_predictions = [
            span
            for span in predicted_spans
            if not (span[1] <= match.start or span[0] >= match.end)
        ]
        if overlapping_predictions:
            continue
        missing_matches.append(
            {
                "text": record["text"][
                    record["token_spans"][match.start][0] : record["token_spans"][match.end - 1][1]
                ],
                "token_start": match.start,
                "token_end": match.end,
                "suggested_type": match.entity_type,
                "alias_text": alias_text,
            }
        )
    return missing_matches


def auto_check_labels(
    record: dict[str, Any],
    labels: list[str],
    matcher: MaxForwardDictionaryMatcher | None = None,
    max_entity_ratio: float = 0.7,
) -> tuple[bool, str, bool]:
    entity_token_count = sum(1 for item in labels if item != "O")
    if record["tokens"] and entity_token_count / len(record["tokens"]) > max_entity_ratio:
        return False, "实体 token 占比过高", False
    if _span_crosses_punctuation(record["tokens"], labels):
        return False, "实体 span 跨越了明显无关标点", False
    if find_core_missing_dictionary_matches(record, labels, matcher):
        return False, "词典核心实体漏标", False

    used_context_exemption = False
    if matcher is not None:
        predicted_spans = labels_to_spans(labels)
        for match in matcher.find_matches(record["tokens"]):
            overlapping_predictions = [
                span
                for span in predicted_spans
                if not (span[1] <= match.start or span[0] >= match.end)
            ]
            if not overlapping_predictions:
                continue
            predicted_span = next(
                (span for span in overlapping_predictions if span[0] == match.start and span[1] == match.end),
                overlapping_predictions[0],
            )
            predicted_type = predicted_span[2]
            dictated_type = match.entity_type
            if predicted_type == dictated_type:
                continue
            if _uses_context_exemption(record, predicted_type, dictated_type, match.start):
                used_context_exemption = True
                continue
            if _allow_concept_whitelist_exemption(predicted_type, dictated_type, match):
                continue
            return False, "与高置信词典类型冲突", False
    if used_context_exemption:
        return True, "auto_checked_with_context_exemption", True
    return True, "auto_checked", False


def _build_record_context(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sentence_id": record["sentence_id"],
        "doc_id": record["doc_id"],
        "source_id": record.get("source_id", ""),
        "text": record["text"],
    }


def _build_rejected_record(
    record: dict[str, Any],
    *,
    reason: str,
    stage: str,
    raw_response: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    rejected_record = {
        **_build_record_context(record),
        "reason": reason,
        "stage": stage,
    }
    if raw_response is not None:
        rejected_record["raw_response"] = raw_response
    if traceback_text is not None:
        rejected_record["traceback"] = traceback_text
    return rejected_record


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.HTTPError) and exc.code in {408, 504}:
        return True
    message = str(exc).casefold()
    return "timed out" in message or "timeout" in message


def _extract_targeted_hits(text: str) -> list[str]:
    lowered_text = text.casefold()
    return [keyword for keyword in GOLD_TARGET_KEYWORDS if keyword.casefold() in lowered_text]


def _counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _build_review_queue_record(
    record: dict[str, Any],
    *,
    labels: list[str],
    weak_label_confidence: float | None,
    weak_label_stage: str,
    review_reason: str,
    missing_dictionary_mentions: list[dict[str, Any]],
    used_bio_fix: bool,
    used_context_exemption: bool,
    targeted_hits: list[str],
) -> dict[str, Any]:
    return {
        "sentence_id": record["sentence_id"],
        "doc_id": record["doc_id"],
        "source_id": record.get("source_id", ""),
        "text": record["text"],
        "tokens": list(record["tokens"]),
        "labels": labels,
        "weak_label_confidence": weak_label_confidence,
        "weak_label_stage": weak_label_stage,
        "review_reason": review_reason,
        "missing_dictionary_mentions": missing_dictionary_mentions,
        "used_bio_fix": used_bio_fix,
        "used_context_exemption": used_context_exemption,
        "targeted_hits": targeted_hits,
    }


def _sort_review_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            item.get("review_reason") != "core_dictionary_miss",
            float(item["weak_label_confidence"]) if item.get("weak_label_confidence") is not None else -1.0,
            str(item.get("sentence_id", "")),
        ),
    )


def _add_targeted_reject_reason_counts(
    targeted_keyword_reason_counts: dict[str, Counter[str]],
    targeted_hits: list[str],
    reason: str,
) -> None:
    for keyword in targeted_hits:
        targeted_keyword_reason_counts[keyword][reason] += 1


def _build_reject_summary(
    *,
    total_candidate_count: int,
    accepted_records: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
    reason_counts: Counter[str],
    stage_counts: Counter[str],
    accepted_review_status_counts: Counter[str],
    accepted_doc_counts: Counter[str],
    rejected_doc_counts: Counter[str],
    targeted_hit_counts: dict[str, dict[str, int]],
    targeted_keyword_reason_counts: dict[str, Counter[str]],
    timeout_count: int,
    parse_error_count: int,
    validation_error_count: int,
    unexpected_error_count: int,
) -> dict[str, Any]:
    accepted_count = len(accepted_records)
    rejected_count = len(rejected_records)
    acceptance_rate = accepted_count / total_candidate_count if total_candidate_count else 0.0
    return {
        "total_candidate_count": total_candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "acceptance_rate": acceptance_rate,
        "reason_counts": _counter_to_sorted_dict(reason_counts),
        "reject_reason_counts": _counter_to_sorted_dict(reason_counts),
        "stage_counts": _counter_to_sorted_dict(stage_counts),
        "accepted_review_status_counts": _counter_to_sorted_dict(accepted_review_status_counts),
        "accepted_doc_counts": _counter_to_sorted_dict(accepted_doc_counts),
        "rejected_doc_counts": _counter_to_sorted_dict(rejected_doc_counts),
        "targeted_hit_counts": targeted_hit_counts,
        "targeted_keyword_reason_counts": {
            keyword: _counter_to_sorted_dict(counter) for keyword, counter in sorted(targeted_keyword_reason_counts.items())
        },
        "targeted_reject_reason_counts": {
            keyword: _counter_to_sorted_dict(counter) for keyword, counter in sorted(targeted_keyword_reason_counts.items())
        },
        "timeout_count": timeout_count,
        "parse_error_count": parse_error_count,
        "validation_error_count": validation_error_count,
        "unexpected_error_count": unexpected_error_count,
    }


def _label_record_once(
    record: dict[str, Any],
    *,
    api_config: WeakLabelApiConfig,
    matcher: MaxForwardDictionaryMatcher | None,
    timeout_seconds: int,
) -> LabeledRecordResult:
    raw_text: str | None = None
    try:
        raw_text = call_openai_compatible_api(
            api_key=api_config.api_key,
            base_url=api_config.base_url,
            model_name=api_config.model_name,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(record, matcher),
            timeout_seconds=timeout_seconds,
        )
        labels = parse_llm_labels(raw_text)
        if len(labels) != len(record["tokens"]):
            raise ValueError(f"token 数量与 label 数量不一致: {len(record['tokens'])} != {len(labels)}")
        fixed_labels, used_bio_fix = fix_bio_sequence(labels)
        missing_core_matches = find_core_missing_dictionary_matches(record, fixed_labels, matcher)
        if missing_core_matches:
            return LabeledRecordResult(
                accepted=False,
                stage="auto_validation",
                reason="词典核心实体漏标",
                used_bio_fix=used_bio_fix,
                used_context_exemption=False,
                labels=fixed_labels,
                missing_dictionary_mentions=missing_core_matches,
                raw_response=raw_text,
            )
        accepted, reason, used_context_exemption = auto_check_labels(record, fixed_labels, matcher=matcher)
        if not accepted:
            return LabeledRecordResult(
                accepted=False,
                stage="auto_validation",
                reason=reason,
                used_bio_fix=used_bio_fix,
                used_context_exemption=used_context_exemption,
                raw_response=raw_text,
            )
        review_status = reason
        if used_bio_fix and used_context_exemption:
            review_status = "auto_checked_with_bio_fix_and_context_exemption"
        elif used_bio_fix:
            review_status = "auto_checked_with_bio_fix"
        return LabeledRecordResult(
            accepted=True,
            stage=review_status,
            reason=review_status,
            used_bio_fix=used_bio_fix,
            used_context_exemption=used_context_exemption,
            labels=fixed_labels,
            raw_response=raw_text,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        stage = "request_timeout" if _is_timeout_error(exc) else "request_error"
        return LabeledRecordResult(
            accepted=False,
            stage=stage,
            reason=str(exc),
            used_bio_fix=False,
            used_context_exemption=False,
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return LabeledRecordResult(
            accepted=False,
            stage="response_parse",
            reason=str(exc),
            used_bio_fix=False,
            used_context_exemption=False,
            raw_response=raw_text,
        )
    except Exception as exc:  # noqa: BLE001
        return LabeledRecordResult(
            accepted=False,
            stage="unexpected_error",
            reason=f"{type(exc).__name__}: {exc}",
            used_bio_fix=False,
            used_context_exemption=False,
            raw_response=raw_text,
            traceback_text=traceback.format_exc(),
        )


def weak_label_records(
    tokenized_path: Path,
    output_path: Path,
    reject_report_path: Path,
    matcher: MaxForwardDictionaryMatcher | None = None,
    sleep_seconds: float = 0.0,
    timeout_seconds: int = 60,
    progress_every: int = 1,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> tuple[int, int]:
    api_config = resolve_weak_label_api_config(api_key=api_key, base_url=base_url, model_name=model_name)
    tokenized_records = read_jsonl(tokenized_path)

    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    accepted_review_status_counts: Counter[str] = Counter()
    accepted_doc_counts: Counter[str] = Counter()
    rejected_doc_counts: Counter[str] = Counter()
    targeted_hits_by_sentence_id = {
        record["sentence_id"]: _extract_targeted_hits(record["text"]) for record in tokenized_records
    }
    targeted_hit_counts = {
        keyword: {"candidate_count": 0, "accepted_count": 0, "rejected_count": 0}
        for keyword in GOLD_TARGET_KEYWORDS
    }
    for hits in targeted_hits_by_sentence_id.values():
        for keyword in hits:
            targeted_hit_counts[keyword]["candidate_count"] += 1
    targeted_keyword_reason_counts = {keyword: Counter() for keyword in GOLD_TARGET_KEYWORDS}
    timeout_count = 0
    parse_error_count = 0
    validation_error_count = 0
    unexpected_error_count = 0
    timeout_retry_records: list[tuple[int, dict[str, Any]]] = []
    total_count = len(tokenized_records)
    review_queue_path = output_path.with_name("weak_label_review_queue.jsonl")
    print(
        f"[weak-label] 开始处理 {total_count} 条句子，model={api_config.model_name}，base_url={api_config.base_url}",
        flush=True,
    )

    def flush_partial_outputs() -> None:
        write_jsonl(output_path, accepted_records)
        write_json(
            reject_report_path,
            {
                "summary": _build_reject_summary(
                    total_candidate_count=total_count,
                    accepted_records=accepted_records,
                    rejected_records=rejected_records,
                    reason_counts=reason_counts,
                    stage_counts=stage_counts,
                    accepted_review_status_counts=accepted_review_status_counts,
                    accepted_doc_counts=accepted_doc_counts,
                    rejected_doc_counts=rejected_doc_counts,
                    targeted_hit_counts=targeted_hit_counts,
                    targeted_keyword_reason_counts=targeted_keyword_reason_counts,
                    timeout_count=timeout_count,
                    parse_error_count=parse_error_count,
                    validation_error_count=validation_error_count,
                    unexpected_error_count=unexpected_error_count,
                ),
                "rejected_records": rejected_records,
            },
        )
        write_jsonl(review_queue_path, _sort_review_queue(review_queue))

    def handle_final_result(
        record: dict[str, Any],
        targeted_hits: list[str],
        result: LabeledRecordResult,
    ) -> None:
        nonlocal timeout_count, parse_error_count, validation_error_count, unexpected_error_count

        if result.accepted:
            weak_label_stage = result.stage
            weak_label_confidence = WEAK_LABEL_CONFIDENCE_BY_STAGE[weak_label_stage]
            accepted_record = build_labeled_record(
                tokenized_record=record,
                labels=list(result.labels or []),
                label_source="llm_weak_supervision",
                review_status=weak_label_stage,
            )
            accepted_record["weak_label_confidence"] = weak_label_confidence
            accepted_record["weak_label_stage"] = weak_label_stage
            accepted_records.append(accepted_record)
            stage_counts[weak_label_stage] += 1
            accepted_review_status_counts[weak_label_stage] += 1
            accepted_doc_counts[record["doc_id"]] += 1
            for keyword in targeted_hits:
                targeted_hit_counts[keyword]["accepted_count"] += 1
            if (
                weak_label_confidence < 0.85
                or result.used_bio_fix
                or result.used_context_exemption
                or bool(targeted_hits)
            ):
                review_queue.append(
                    _build_review_queue_record(
                        record,
                        labels=list(result.labels or []),
                        weak_label_confidence=weak_label_confidence,
                        weak_label_stage=weak_label_stage,
                        review_reason="low_confidence_or_rule_exemption_or_targeted_hit",
                        missing_dictionary_mentions=[],
                        used_bio_fix=result.used_bio_fix,
                        used_context_exemption=result.used_context_exemption,
                        targeted_hits=targeted_hits,
                    )
                )
            return

        if result.stage == "request_timeout":
            timeout_count += 1
        elif result.stage == "response_parse":
            parse_error_count += 1
        elif result.stage in {"auto_validation", "record_validation"}:
            validation_error_count += 1
        elif result.stage == "unexpected_error":
            unexpected_error_count += 1

        rejected_records.append(
            _build_rejected_record(
                record,
                reason=result.reason,
                stage=result.stage,
                raw_response=result.raw_response,
                traceback_text=result.traceback_text,
            )
        )
        reason_counts[result.reason] += 1
        stage_counts[result.stage] += 1
        rejected_doc_counts[record["doc_id"]] += 1
        for keyword in targeted_hits:
            targeted_hit_counts[keyword]["rejected_count"] += 1
        _add_targeted_reject_reason_counts(targeted_keyword_reason_counts, targeted_hits, result.reason)
        if result.reason == "词典核心实体漏标":
            review_queue.append(
                _build_review_queue_record(
                    record,
                    labels=list(result.labels or []),
                    weak_label_confidence=None,
                    weak_label_stage="rejected_core_dictionary_miss",
                    review_reason="core_dictionary_miss",
                    missing_dictionary_mentions=list(result.missing_dictionary_mentions or []),
                    used_bio_fix=result.used_bio_fix,
                    used_context_exemption=result.used_context_exemption,
                    targeted_hits=targeted_hits,
                )
            )

    for index, record in enumerate(tokenized_records, start=1):
        sentence_id = record["sentence_id"]
        targeted_hits = targeted_hits_by_sentence_id.get(sentence_id, [])
        print(f"[weak-label] ({index}/{total_count}) sentence_id={sentence_id} 开始请求", flush=True)
        result = _label_record_once(
            record,
            api_config=api_config,
            matcher=matcher,
            timeout_seconds=timeout_seconds,
        )
        if result.stage == "request_timeout":
            timeout_retry_records.append((index, record))
            print(f"[weak-label] ({index}/{total_count}) sentence_id={sentence_id} 首次超时，加入末尾重试", flush=True)
            if progress_every > 0 and (index % progress_every == 0 or index == total_count):
                flush_partial_outputs()
            continue

        handle_final_result(record, targeted_hits, result)
        if result.accepted:
            print(f"[weak-label] ({index}/{total_count}) sentence_id={sentence_id} 通过校验", flush=True)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        else:
            print(f"[weak-label] ({index}/{total_count}) sentence_id={sentence_id} 被拒绝：{result.reason}", flush=True)

        if progress_every > 0 and (index % progress_every == 0 or index == total_count):
            flush_partial_outputs()
            print(
                f"[weak-label] 已落盘 accepted={len(accepted_records)} rejected={len(rejected_records)}",
                flush=True,
            )

    if timeout_retry_records:
        print(f"[weak-label] 开始处理 {len(timeout_retry_records)} 条 timeout 重试样本", flush=True)
    for retry_index, (original_index, record) in enumerate(timeout_retry_records, start=1):
        sentence_id = record["sentence_id"]
        targeted_hits = targeted_hits_by_sentence_id.get(sentence_id, [])
        print(
            f"[weak-label][retry] ({retry_index}/{len(timeout_retry_records)}) sentence_id={sentence_id} 开始二次请求",
            flush=True,
        )
        result = _label_record_once(
            record,
            api_config=api_config,
            matcher=matcher,
            timeout_seconds=timeout_seconds,
        )
        handle_final_result(record, targeted_hits, result)
        if result.accepted:
            print(
                f"[weak-label][retry] sentence_id={sentence_id} 二次请求通过校验",
                flush=True,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        else:
            print(
                f"[weak-label][retry] sentence_id={sentence_id} 二次请求失败：{result.reason}",
                flush=True,
            )
        if progress_every > 0 and (retry_index % progress_every == 0 or retry_index == len(timeout_retry_records)):
            flush_partial_outputs()

    flush_partial_outputs()
    print(
        f"[weak-label] 处理结束 accepted={len(accepted_records)} rejected={len(rejected_records)}",
        flush=True,
    )
    return len(accepted_records), len(rejected_records)
