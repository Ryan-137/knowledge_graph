from __future__ import annotations

from collections import Counter
import json
import random
import re
from pathlib import Path
from typing import Any

from kg_core.schemas import TokenSpan
from kg_core.taxonomy import BIO_LABELS, VALID_LABEL_SET


# 统一使用本地 tokenizer，避免 LLM 自作主张改变切分方式导致 BIO 对齐失败。
TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z]+)*(?:'[sS])?|\d+(?:\.\d+)*|[^\w\s]",
    flags=re.UNICODE,
)

GOLD_TARGET_KEYWORDS = (
    "Turing machine",
    "Turing Award",
    "Bombe",
    "ACE",
    "Ferranti Mark 1",
    "Computing Machinery and Intelligence",
)

WEAK_LABEL_SECONDARY_KEYWORDS = (
    "Alan Turing",
    "Turing Test",
    "Halting Problem",
    "Church-Turing thesis",
    "Alonzo Church",
    "Princeton",
    "Bletchley Park",
    "ACM",
)


def read_json(file_path: Path) -> Any:
    # 兼容 Windows 下可能混入的 UTF-8 BOM，避免 JSON/JSONL 首行解析失败。
    return json.loads(file_path.read_text(encoding="utf-8-sig"))


def write_json(file_path: Path, payload: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        payload = json.loads(stripped_line)
        if not isinstance(payload, dict):
            raise ValueError(f"{file_path.as_posix()} 第 {line_number} 行不是对象记录")
        records.append(payload)
    return records


def write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"第 {index} 条记录不是对象，无法写入 JSONL")
        serialized.append(json.dumps(record, ensure_ascii=False))
    content = "\n".join(serialized)
    if content:
        content += "\n"
    file_path.write_text(content, encoding="utf-8")


def tokenize_sentence_text(text: str) -> list[TokenSpan]:
    """对英文句子做统一分词，并保留字符级偏移。"""

    return [TokenSpan(text=match.group(0), start=match.start(), end=match.end()) for match in TOKEN_PATTERN.finditer(text)]


def build_tokenized_record(sentence_payload: dict[str, Any]) -> dict[str, Any]:
    token_spans = tokenize_sentence_text(sentence_payload["text"])
    return {
        "sentence_id": sentence_payload["sentence_id"],
        "doc_id": sentence_payload["doc_id"],
        "source_id": sentence_payload.get("source_id", ""),
        "sentence_index_in_doc": int(sentence_payload.get("sentence_index_in_doc", 0)),
        "text": sentence_payload["text"],
        "tokens": [item.text for item in token_spans],
        "token_spans": [[item.start, item.end] for item in token_spans],
        "normalized_time": list(sentence_payload.get("normalized_time", [])),
        "time_mentions": list(sentence_payload.get("time_mentions", [])),
    }


def tokenize_sentences_file(sentences_path: Path, output_path: Path) -> int:
    payload = read_jsonl(sentences_path)
    records = [build_tokenized_record(item) for item in payload]
    write_jsonl(output_path, records)
    return len(records)


def _group_records_by_doc_id(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    records_by_doc_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_doc_id.setdefault(record["doc_id"], []).append(record)
    return records_by_doc_id


def _select_records_randomly(
    candidates: list[dict[str, Any]],
    limit: int,
    sampler: random.Random,
    picked_sentence_ids: set[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    shuffled = candidates[:]
    sampler.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    for record in shuffled:
        sentence_id = record["sentence_id"]
        if sentence_id in picked_sentence_ids:
            continue
        selected.append(record)
        picked_sentence_ids.add(sentence_id)
        if len(selected) >= limit:
            break
    return selected


def _allocate_doc_size_weighted_budget(
    available_counts: dict[str, int],
    remaining_budget: int,
) -> dict[str, int]:
    quotas = {doc_id: 0 for doc_id in available_counts}
    if remaining_budget <= 0:
        return quotas

    total_available = sum(available_counts.values())
    if total_available <= 0:
        return quotas

    assigned = 0
    remainders: list[tuple[float, int, str]] = []
    for doc_id, available_count in available_counts.items():
        exact_quota = remaining_budget * available_count / total_available
        assigned_quota = min(available_count, int(exact_quota))
        quotas[doc_id] = assigned_quota
        assigned += assigned_quota
        remainders.append((exact_quota - assigned_quota, available_count, doc_id))

    leftover = remaining_budget - assigned
    for _, _, doc_id in sorted(remainders, reverse=True):
        if leftover <= 0:
            break
        if quotas[doc_id] >= available_counts[doc_id]:
            continue
        quotas[doc_id] += 1
        leftover -= 1

    if leftover <= 0:
        return quotas

    docs_with_room = sorted(available_counts, key=lambda doc_id: available_counts[doc_id], reverse=True)
    while leftover > 0 and docs_with_room:
        advanced = False
        for doc_id in docs_with_room:
            if quotas[doc_id] >= available_counts[doc_id]:
                continue
            quotas[doc_id] += 1
            leftover -= 1
            advanced = True
            if leftover <= 0:
                break
        if not advanced:
            break
    return quotas


def _select_records_with_doc_dispersion(
    candidates: list[dict[str, Any]],
    limit: int,
    sampler: random.Random,
    picked_sentence_ids: set[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    shuffled = candidates[:]
    sampler.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_doc_ids: set[str] = set()

    # 先按文档去重抽样，优先把模板扩散到更多来源，避免人工校验只盯着单一文档。
    for record in shuffled:
        sentence_id = record["sentence_id"]
        doc_id = record["doc_id"]
        if sentence_id in picked_sentence_ids or doc_id in selected_doc_ids:
            continue
        selected.append(record)
        selected_doc_ids.add(doc_id)
        picked_sentence_ids.add(sentence_id)
        if len(selected) >= limit:
            return selected

    # 如果跨文档样本不够，再从剩余候选中补齐数量。
    for record in shuffled:
        sentence_id = record["sentence_id"]
        if sentence_id in picked_sentence_ids:
            continue
        selected.append(record)
        picked_sentence_ids.add(sentence_id)
        if len(selected) >= limit:
            return selected
    return selected


def sample_weak_label_candidates(
    tokenized_path: Path,
    base_sample_per_doc: int,
    sample_budget: int,
    targeted_topup_per_keyword: int,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    为弱标注阶段挑选候选句子。

    采样分三步：
    1. 每个文档先保底抽样，避免弱标注结果只覆盖头部文档；
    2. 剩余预算按文档可用句子数加权补齐；
    3. 对核心关键词和第二优先级关键词做额外定向补采。

    `sample_budget` 只约束前两步的基础预算，定向补采会在此基础上额外增加样本。
    """

    if base_sample_per_doc < 0:
        raise ValueError(f"base_sample_per_doc 不能小于 0，当前值为 {base_sample_per_doc}")
    if sample_budget < 0:
        raise ValueError(f"sample_budget 不能小于 0，当前值为 {sample_budget}")
    if targeted_topup_per_keyword < 0:
        raise ValueError(f"targeted_topup_per_keyword 不能小于 0，当前值为 {targeted_topup_per_keyword}")

    records = read_jsonl(tokenized_path)
    records_by_doc_id = _group_records_by_doc_id(records)
    sampler = random.Random(seed)
    picked_sentence_ids: set[str] = set()

    minimum_required_budget = sum(min(base_sample_per_doc, len(doc_records)) for doc_records in records_by_doc_id.values())
    if minimum_required_budget > sample_budget:
        raise ValueError(
            "sample_budget 小于文档保底采样所需数量，"
            f"minimum_required_budget={minimum_required_budget}，sample_budget={sample_budget}。"
            "请提高 sample_budget 或降低 base_sample_per_doc。"
        )

    selected_records: list[dict[str, Any]] = []

    # 先给每个文档保底，避免训练样本完全被长文档垄断。
    for doc_id in sorted(records_by_doc_id):
        doc_records = records_by_doc_id[doc_id]
        selected_records.extend(
            _select_records_randomly(
                candidates=doc_records,
                limit=min(base_sample_per_doc, len(doc_records)),
                sampler=sampler,
                picked_sentence_ids=picked_sentence_ids,
            )
        )
    base_selected_count = len(selected_records)

    # 再把剩余预算按文档剩余句子数加权分配，保留长文档的额外覆盖能力。
    remaining_budget = max(0, sample_budget - base_selected_count)
    available_counts_by_doc = {
        doc_id: sum(1 for record in doc_records if record["sentence_id"] not in picked_sentence_ids)
        for doc_id, doc_records in records_by_doc_id.items()
    }
    budget_quotas = _allocate_doc_size_weighted_budget(available_counts_by_doc, remaining_budget)
    budget_topup_count = 0
    for doc_id in sorted(records_by_doc_id):
        doc_records = records_by_doc_id[doc_id]
        added_records = _select_records_randomly(
            candidates=doc_records,
            limit=budget_quotas.get(doc_id, 0),
            sampler=sampler,
            picked_sentence_ids=picked_sentence_ids,
        )
        selected_records.extend(added_records)
        budget_topup_count += len(added_records)

    lowered_records = [(record, record["text"].casefold()) for record in records]
    targeted_keyword_counts: dict[str, int] = {}

    def collect_targeted_records(keywords: tuple[str, ...]) -> int:
        targeted_count = 0
        if targeted_topup_per_keyword <= 0:
            return targeted_count
        for keyword in keywords:
            keyword_casefold = keyword.casefold()
            candidates = [record for record, lowered_text in lowered_records if keyword_casefold in lowered_text]
            added_records = _select_records_with_doc_dispersion(
                candidates=candidates,
                limit=targeted_topup_per_keyword,
                sampler=sampler,
                picked_sentence_ids=picked_sentence_ids,
            )
            targeted_keyword_counts[keyword] = len(added_records)
            selected_records.extend(added_records)
            targeted_count += len(added_records)
        return targeted_count

    core_targeted_count = collect_targeted_records(GOLD_TARGET_KEYWORDS)
    secondary_targeted_count = collect_targeted_records(WEAK_LABEL_SECONDARY_KEYWORDS)

    selected_records = sorted(selected_records, key=lambda item: item["sentence_id"])
    selected_doc_ids = {record["doc_id"] for record in selected_records}
    summary = {
        "input_record_count": len(records),
        "doc_count": len(records_by_doc_id),
        "base_sample_per_doc": base_sample_per_doc,
        "sample_budget": sample_budget,
        "targeted_topup_per_keyword": targeted_topup_per_keyword,
        "base_selected_count": base_selected_count,
        "budget_topup_count": budget_topup_count,
        "core_targeted_count": core_targeted_count,
        "secondary_targeted_count": secondary_targeted_count,
        "selected_count": len(selected_records),
        "selected_doc_count": len(selected_doc_ids),
        "selected_doc_coverage": (len(selected_doc_ids) / len(records_by_doc_id)) if records_by_doc_id else 0.0,
        "targeted_keyword_counts": targeted_keyword_counts,
    }
    return selected_records, summary


def extract_gold_seed(tokenized_path: Path, output_path: Path, sample_size: int, seed: int = 42) -> int:
    """
    生成黄金测试集人工校验模板。

    这里故意不写 labels，避免误把模板文件当成已标注语料使用。
    """

    if sample_size <= 0:
        raise ValueError(f"sample_size 必须大于 0，当前值为 {sample_size}")

    records = read_jsonl(tokenized_path)
    sampler = random.Random(seed)
    picked_sentence_ids: set[str] = set()
    prioritized: list[dict[str, Any]] = []
    targeted_budget = min(sample_size, max(len(GOLD_TARGET_KEYWORDS), sample_size // 2))

    lowered_records = [(record, record["text"].casefold()) for record in records]
    for index, keyword in enumerate(GOLD_TARGET_KEYWORDS):
        remaining_target_keywords = len(GOLD_TARGET_KEYWORDS) - index
        remaining_target_slots = targeted_budget - len(prioritized)
        if remaining_target_slots <= 0:
            break
        quota_for_keyword = max(1, remaining_target_slots // remaining_target_keywords)
        keyword_casefold = keyword.casefold()
        candidates = [record for record, lowered_text in lowered_records if keyword_casefold in lowered_text]
        prioritized.extend(
            _select_records_with_doc_dispersion(
                candidates=candidates,
                limit=quota_for_keyword,
                sampler=sampler,
                picked_sentence_ids=picked_sentence_ids,
            )
        )

    remaining_candidates = [record for record in records if record["sentence_id"] not in picked_sentence_ids]
    remaining_needed = max(0, sample_size - len(prioritized))
    picked = prioritized + _select_records_with_doc_dispersion(
        candidates=remaining_candidates,
        limit=remaining_needed,
        sampler=sampler,
        picked_sentence_ids=picked_sentence_ids,
    )
    picked = sorted(picked[:sample_size], key=lambda item: item["sentence_id"])
    for item in picked:
        item["label_source"] = "human_gold_template"
        item["review_status"] = "pending_manual_review"
    write_jsonl(output_path, picked)
    return len(picked)


def strip_code_fence(raw_text: str) -> str:
    normalized = raw_text.strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return normalized


def parse_llm_labels(raw_text: str) -> list[str]:
    """
    解析 LLM 弱标注返回值。

    接受两种结构：
    1. 纯 JSON 数组：["B-PER", ...]
    2. JSON 对象：{"labels": ["B-PER", ...]}
    """

    normalized = strip_code_fence(raw_text)
    payload = json.loads(normalized)
    labels = payload.get("labels") if isinstance(payload, dict) else payload
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError("LLM 标注结果不是合法的 label 数组")
    return labels


def validate_bio_labels(tokens: list[str], labels: list[str]) -> list[str]:
    if len(tokens) != len(labels):
        raise ValueError(f"token 数量与 label 数量不一致: {len(tokens)} != {len(labels)}")
    if any(label not in VALID_LABEL_SET for label in labels):
        illegal_labels = sorted({label for label in labels if label not in VALID_LABEL_SET})
        raise ValueError(f"存在非法标签: {illegal_labels}")

    for index, label in enumerate(labels):
        if label == "O":
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "I":
            if index == 0:
                raise ValueError("句首不允许出现 I- 标签")
            previous = labels[index - 1]
            if previous == "O":
                raise ValueError("不允许出现 O -> I-X")
            previous_prefix, previous_type = previous.split("-", 1)
            if previous_prefix not in {"B", "I"} or previous_type != entity_type:
                raise ValueError("不允许出现跨类型 B/I 转移")
    return labels


def count_entity_spans(labels: list[str]) -> int:
    return sum(1 for label in labels if label.startswith("B-"))


def build_labeled_record(tokenized_record: dict[str, Any], labels: list[str], label_source: str, review_status: str) -> dict[str, Any]:
    validated_labels = validate_bio_labels(tokenized_record["tokens"], labels)
    return {
        **tokenized_record,
        "labels": validated_labels,
        "label_source": label_source,
        "review_status": review_status,
    }


def split_weak_and_gold_datasets(
    weak_labeled_path: Path,
    train_output_path: Path,
    dev_output_path: Path,
    dev_ratio: float,
    seed: int = 42,
) -> dict[str, Any]:
    weak_records = read_jsonl(weak_labeled_path)
    accepted_records = [item for item in weak_records if (item.get("review_status") or "").startswith("auto_checked")]
    if not 0 < dev_ratio < 1:
        raise ValueError(f"dev_ratio 必须在 0 和 1 之间，当前值为 {dev_ratio}")
    if not accepted_records:
        raise ValueError("弱标注训练集为空，无法切分 train/dev")
    if len(accepted_records) < 2:
        raise ValueError(f"accepted weak-labeled 样本至少需要 2 条，当前只有 {len(accepted_records)} 条")

    sampler = random.Random(seed)
    shuffled_records = accepted_records[:]
    sampler.shuffle(shuffled_records)
    target_dev_sentence_count = max(1, int(len(shuffled_records) * dev_ratio))
    target_dev_sentence_count = min(target_dev_sentence_count, len(shuffled_records) - 1)

    train_records = sorted(
        shuffled_records[target_dev_sentence_count:],
        key=lambda item: item["sentence_id"],
    )
    dev_records = sorted(
        shuffled_records[:target_dev_sentence_count],
        key=lambda item: item["sentence_id"],
    )
    write_jsonl(train_output_path, train_records)
    write_jsonl(dev_output_path, dev_records)
    weak_doc_ids = {record["doc_id"] for record in weak_records}
    accepted_doc_ids = {record["doc_id"] for record in accepted_records}
    train_doc_ids = {record["doc_id"] for record in train_records}
    dev_doc_ids = {record["doc_id"] for record in dev_records}
    weak_doc_count = len(weak_doc_ids)
    accepted_review_status_distribution = dict(
        sorted(
            Counter(str(record.get("review_status", "")) for record in accepted_records).items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    accepted_confidence_distribution = dict(
        sorted(
            Counter(f"{float(record['weak_label_confidence']):.2f}" for record in accepted_records if isinstance(record.get("weak_label_confidence"), (int, float))).items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return {
        "weak_record_count": len(weak_records),
        "accepted_count": len(accepted_records),
        "train_count": len(train_records),
        "dev_count": len(dev_records),
        "weak_doc_count": weak_doc_count,
        "accepted_doc_count": len(accepted_doc_ids),
        "train_doc_count": len(train_doc_ids),
        "dev_doc_count": len(dev_doc_ids),
        "accepted_doc_coverage": (len(accepted_doc_ids) / weak_doc_count) if weak_doc_count else 0.0,
        "train_doc_coverage": (len(train_doc_ids) / weak_doc_count) if weak_doc_count else 0.0,
        "dev_doc_coverage": (len(dev_doc_ids) / weak_doc_count) if weak_doc_count else 0.0,
        "accepted_review_status_distribution": accepted_review_status_distribution,
        "accepted_weak_label_confidence_distribution": accepted_confidence_distribution,
    }


def summarize_label_distribution(records_path: Path, output_path: Path) -> dict[str, Any]:
    records = read_jsonl(records_path)
    label_counter: dict[str, int] = {label: 0 for label in BIO_LABELS}
    total_entities = 0
    review_status_counter: Counter[str] = Counter()
    weak_label_confidence_counter: Counter[str] = Counter()
    doc_ids: set[str] = set()
    for record in records:
        labels = record.get("labels", [])
        total_entities += count_entity_spans(labels)
        if record.get("doc_id"):
            doc_ids.add(str(record["doc_id"]))
        if record.get("review_status"):
            review_status_counter[str(record["review_status"])] += 1
        if isinstance(record.get("weak_label_confidence"), (int, float)):
            weak_label_confidence_counter[f"{float(record['weak_label_confidence']):.2f}"] += 1
        for label in labels:
            label_counter[label] = label_counter.get(label, 0) + 1

    summary = {
        "record_count": len(records),
        "entity_span_count": total_entities,
        "label_distribution": label_counter,
        "doc_coverage_count": len(doc_ids),
        "doc_ids": sorted(doc_ids),
        "review_status_distribution": dict(
            sorted(review_status_counter.items(), key=lambda item: (-item[1], item[0]))
        ),
        "weak_label_confidence_distribution": dict(
            sorted(weak_label_confidence_counter.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    write_json(output_path, summary)
    return summary
