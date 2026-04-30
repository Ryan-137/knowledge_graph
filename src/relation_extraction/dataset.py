from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mention_crf.data import build_tokenized_record

from .config import NA_RELATION_LABEL, PAD_TOKEN, UNK_TOKEN, RelationExtractionConfig


@dataclass(frozen=True)
class RelationSentenceEvidence:
    """一个 bag 中的一条句子证据。"""

    sentence_id: str
    doc_id: str
    text: str
    tokens: list[str]
    token_spans: list[list[int]]
    subject_span: tuple[int, int]
    object_span: tuple[int, int]
    subject_mention_id: str
    object_mention_id: str
    source_id: str = ""
    sentence_index_in_doc: int | None = None
    candidate_id: str = ""
    predicate: str = ""
    subject_text: str = ""
    object_text: str = ""
    pair_source: str = ""
    exact_claim_match: bool = False
    matched_claim_ids: list[str] | None = None
    bridge_predicates: list[str] | None = None
    bridge_details: dict[str, Any] | None = None
    weak_label: str = ""
    weak_label_reason: str = ""
    supervision_tier: str = ""
    allowed_predicates: list[str] | None = None
    candidate_predicates: list[str] | None = None
    positive_predicates: list[str] | None = None
    hard_negative_predicates: list[str] | None = None
    unknown_predicates: list[str] | None = None
    review_predicates: list[str] | None = None
    weak_labels_by_predicate: dict[str, str] | None = None
    sentence_trigger_hits: dict[str, list[str]] | None = None
    local_trigger_hits: dict[str, list[str]] | None = None
    exact_claim_matches: dict[str, list[str]] | None = None
    candidate_strength_by_predicate: dict[str, str] | None = None


@dataclass(frozen=True)
class RelationBag:
    """同一实体对的 bag，支持多标签监督。"""

    bag_id: str
    doc_id: str
    subject_id: str
    object_id: str
    subject_type: str
    object_type: str
    allowed_predicates: list[str]
    sentence_evidences: list[RelationSentenceEvidence]
    label_names: list[str]

    @property
    def is_na(self) -> bool:
        return self.label_names == [NA_RELATION_LABEL]


@dataclass(frozen=True)
class Vocabulary:
    token_to_index: dict[str, int]
    index_to_token: list[str]
    lowercase_tokens: bool


@dataclass(frozen=True)
class PreparedRelationDataset:
    train_bags: list[RelationBag]
    dev_bags: list[RelationBag]
    test_bags: list[RelationBag]
    label_to_index: dict[str, int]
    index_to_label: list[str]
    vocabulary: Vocabulary
    class_weights: dict[str, float]
    dataset_report: dict[str, Any]


class BagFeatureDataset:
    """轻量级 dataset，故意不直接依赖 torch，便于在缺依赖时仍可做静态检查。"""

    def __init__(
        self,
        bags: list[RelationBag],
        vocabulary: Vocabulary,
        label_to_index: dict[str, int],
        *,
        max_sentence_length: int,
        position_clip: int,
    ) -> None:
        self.bags = bags
        self.vocabulary = vocabulary
        self.label_to_index = label_to_index
        self.max_sentence_length = max_sentence_length
        self.position_clip = position_clip

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bag = self.bags[index]
        return vectorize_bag(
            bag=bag,
            vocabulary=self.vocabulary,
            label_to_index=self.label_to_index,
            max_sentence_length=self.max_sentence_length,
            position_clip=self.position_clip,
        )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{path.as_posix()} 第 {line_number} 行不是对象，无法作为 JSONL 记录读取。")
        records.append(payload)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(serialized) + ("\n" if serialized else ""), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_token(token: str, lowercase: bool) -> str:
    return token.lower() if lowercase else token


def clip_relative_position(relative_position: int, clip_value: int) -> int:
    return max(-clip_value, min(clip_value, relative_position))


def position_to_index(relative_position: int, clip_value: int, *, is_padding: bool = False) -> int:
    if is_padding:
        return 0
    return clip_relative_position(relative_position, clip_value) + clip_value + 1


def build_relation_constraints(ontology_path: Path) -> dict[str, dict[str, set[str]]]:
    ontology = read_json(ontology_path)
    relation_constraints: dict[str, dict[str, set[str]]] = {}
    for item in ontology.get("relations", []):
        relation_name = str(item["name"])
        domain_value = item.get("domain", "Entity")
        range_value = item.get("range", "Entity")
        domain_types = {domain_value} if isinstance(domain_value, str) else {str(value) for value in domain_value}
        range_types = {range_value} if isinstance(range_value, str) else {str(value) for value in range_value}
        relation_constraints[relation_name] = {
            "domain": domain_types,
            "range": range_types,
        }
    return relation_constraints


def load_claim_relation_map(
    claims_path: Path,
    target_relations: set[str] | None = None,
) -> tuple[dict[tuple[str, str], set[str]], set[str]]:
    relation_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    discovered_relations: set[str] = set()
    for row in read_csv_rows(claims_path):
        subject_id = str(row["subject_id"]).strip()
        object_id = str(row["object_id"]).strip()
        predicate = str(row["predicate"]).strip()
        if not subject_id or not object_id or not predicate:
            continue
        discovered_relations.add(predicate)
        if target_relations is not None and predicate not in target_relations:
            continue
        relation_map[(subject_id, object_id)].add(predicate)
    return relation_map, discovered_relations


def load_entity_type_map(entities_path: Path) -> dict[str, str]:
    return {
        str(row["entity_id"]).strip(): str(row.get("entity_type") or "Entity").strip() or "Entity"
        for row in read_csv_rows(entities_path)
    }


def infer_target_relations(config: RelationExtractionConfig) -> list[str]:
    relation_constraints = build_relation_constraints(config.data.ontology_path)
    if config.target_relations:
        return [relation for relation in config.target_relations if relation in relation_constraints]
    _, discovered_relations = load_claim_relation_map(config.data.claims_path)
    return sorted(discovered_relations & set(relation_constraints))


def _relation_is_type_compatible(
    subject_type: str,
    object_type: str,
    constraint: dict[str, set[str]],
) -> bool:
    return subject_type in constraint["domain"] and object_type in constraint["range"]


def _pair_supports_any_target_relation(
    subject_type: str,
    object_type: str,
    relation_constraints: dict[str, dict[str, set[str]]],
    target_relations: Iterable[str],
) -> bool:
    for relation_name in target_relations:
        constraint = relation_constraints.get(relation_name)
        if constraint is None:
            continue
        if _relation_is_type_compatible(subject_type, object_type, constraint):
            return True
    return False


def _build_sentence_map(sentences_path: Path) -> dict[str, dict[str, Any]]:
    sentence_map: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(sentences_path):
        tokenized_record = record if "tokens" in record and "token_spans" in record else build_tokenized_record(record)
        sentence_map[str(tokenized_record["sentence_id"])] = tokenized_record
    return sentence_map


RESOLVED_LINK_DECISIONS = {"LINKED", "LINKED_BY_COREF"}


def _group_resolved_mentions(resolved_mentions_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped_mentions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in read_jsonl(resolved_mentions_path):
        if str(record.get("decision", "")).upper() not in RESOLVED_LINK_DECISIONS:
            continue
        entity_id = str(record.get("entity_id") or "").strip()
        sentence_id = str(record.get("sentence_id") or "").strip()
        if not entity_id or not sentence_id:
            continue
        grouped_mentions[sentence_id].append(record)
    for sentence_id, mentions in grouped_mentions.items():
        mentions.sort(
            key=lambda item: (
                int(item.get("token_start", 0)),
                int(item.get("token_end", 0)),
                str(item.get("mention_id", "")),
            )
        )
    return grouped_mentions


def _candidate_to_sentence_evidence(candidate: dict[str, Any]) -> RelationSentenceEvidence:
    subject_span = (
        int(candidate.get("subject_token_start", candidate.get("subject_token_span", [0, 0])[0])),
        int(candidate.get("subject_token_end", candidate.get("subject_token_span", [0, 0])[1])),
    )
    object_span = (
        int(candidate.get("object_token_start", candidate.get("object_token_span", [0, 0])[0])),
        int(candidate.get("object_token_end", candidate.get("object_token_span", [0, 0])[1])),
    )
    sentence_index = candidate.get("sentence_index_in_doc")
    return RelationSentenceEvidence(
        sentence_id=str(candidate.get("sentence_id", "")),
        doc_id=str(candidate.get("doc_id", "")),
        source_id=str(candidate.get("source_id", "")),
        sentence_index_in_doc=int(sentence_index) if sentence_index not in (None, "") else None,
        text=str(candidate.get("text", "")),
        tokens=list(candidate.get("tokens", [])),
        token_spans=[list(span) for span in candidate.get("token_spans", [])],
        subject_span=subject_span,
        object_span=object_span,
        subject_mention_id=str(candidate.get("subject_mention_id", "")),
        object_mention_id=str(candidate.get("object_mention_id", "")),
        candidate_id=str(candidate.get("candidate_id", "")),
        predicate=str(candidate.get("predicate", "")),
        subject_text=str(candidate.get("subject_text", "")),
        object_text=str(candidate.get("object_text", "")),
        pair_source=str(candidate.get("pair_source", "")),
        exact_claim_match=bool(candidate.get("exact_claim_match", False)),
        matched_claim_ids=[str(item) for item in candidate.get("matched_claim_ids", [])],
        bridge_predicates=[str(item) for item in candidate.get("bridge_predicates", [])],
        bridge_details=dict(candidate.get("bridge_details", {})),
        weak_label=str(candidate.get("weak_label", "")),
        weak_label_reason=str(candidate.get("weak_label_reason", "")),
        supervision_tier=str(candidate.get("supervision_tier", "")),
        allowed_predicates=[str(item) for item in candidate.get("allowed_predicates", [])],
        candidate_predicates=[str(item) for item in candidate.get("candidate_predicates", [])],
        positive_predicates=[str(item) for item in candidate.get("positive_predicates", [])],
        hard_negative_predicates=[str(item) for item in candidate.get("hard_negative_predicates", [])],
        unknown_predicates=[str(item) for item in candidate.get("unknown_predicates", [])],
        review_predicates=[str(item) for item in candidate.get("review_predicates", [])],
        weak_labels_by_predicate=dict(candidate.get("weak_labels_by_predicate", {})),
        sentence_trigger_hits={str(key): list(value) for key, value in dict(candidate.get("sentence_trigger_hits", {})).items()},
        local_trigger_hits={str(key): list(value) for key, value in dict(candidate.get("local_trigger_hits", {})).items()},
        exact_claim_matches={str(key): list(value) for key, value in dict(candidate.get("exact_claim_matches", {})).items()},
        candidate_strength_by_predicate=dict(candidate.get("candidate_strength_by_predicate", {})),
    )


def _merge_unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _merge_dict_list_values(first: dict[str, list[str]] | None, second: dict[str, list[str]] | None) -> dict[str, list[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for payload in (first or {}, second or {}):
        for key, values in payload.items():
            normalized_key = str(key)
            for value in values:
                if str(value):
                    merged[normalized_key].add(str(value))
    return {key: sorted(values) for key, values in sorted(merged.items())}


def _merge_sentence_evidence(
    existing: RelationSentenceEvidence,
    incoming: RelationSentenceEvidence,
) -> RelationSentenceEvidence:
    return RelationSentenceEvidence(
        sentence_id=existing.sentence_id,
        doc_id=existing.doc_id,
        text=existing.text,
        tokens=existing.tokens,
        token_spans=existing.token_spans,
        subject_span=existing.subject_span,
        object_span=existing.object_span,
        subject_mention_id=existing.subject_mention_id,
        object_mention_id=existing.object_mention_id,
        source_id=existing.source_id,
        sentence_index_in_doc=existing.sentence_index_in_doc,
        candidate_id=existing.candidate_id,
        predicate=existing.predicate,
        subject_text=existing.subject_text,
        object_text=existing.object_text,
        pair_source=existing.pair_source,
        exact_claim_match=existing.exact_claim_match or incoming.exact_claim_match,
        matched_claim_ids=_merge_unique(list(existing.matched_claim_ids or []) + list(incoming.matched_claim_ids or [])),
        bridge_predicates=_merge_unique(list(existing.bridge_predicates or []) + list(incoming.bridge_predicates or [])),
        bridge_details=existing.bridge_details or incoming.bridge_details,
        weak_label=existing.weak_label or incoming.weak_label,
        weak_label_reason=existing.weak_label_reason or incoming.weak_label_reason,
        supervision_tier=existing.supervision_tier or incoming.supervision_tier,
        allowed_predicates=_merge_unique(list(existing.allowed_predicates or []) + list(incoming.allowed_predicates or [])),
        candidate_predicates=_merge_unique(list(existing.candidate_predicates or []) + list(incoming.candidate_predicates or [])),
        positive_predicates=_merge_unique(list(existing.positive_predicates or []) + list(incoming.positive_predicates or [])),
        hard_negative_predicates=_merge_unique(
            list(existing.hard_negative_predicates or []) + list(incoming.hard_negative_predicates or [])
        ),
        unknown_predicates=_merge_unique(list(existing.unknown_predicates or []) + list(incoming.unknown_predicates or [])),
        review_predicates=_merge_unique(list(existing.review_predicates or []) + list(incoming.review_predicates or [])),
        weak_labels_by_predicate={
            **dict(existing.weak_labels_by_predicate or {}),
            **dict(incoming.weak_labels_by_predicate or {}),
        },
        sentence_trigger_hits=_merge_dict_list_values(existing.sentence_trigger_hits, incoming.sentence_trigger_hits),
        local_trigger_hits=_merge_dict_list_values(existing.local_trigger_hits, incoming.local_trigger_hits),
        exact_claim_matches=_merge_dict_list_values(existing.exact_claim_matches, incoming.exact_claim_matches),
        candidate_strength_by_predicate={
            **dict(existing.candidate_strength_by_predicate or {}),
            **dict(incoming.candidate_strength_by_predicate or {}),
        },
    )


def _build_relation_bags_from_candidate_records(
    config: RelationExtractionConfig,
    *,
    candidate_records: list[dict[str, Any]],
    include_gold_labels: bool,
) -> tuple[list[RelationBag], list[str]]:
    target_relations = infer_target_relations(config)
    target_relation_set = set(target_relations)
    bag_payloads: dict[str, dict[str, Any]] = {}
    for candidate in candidate_records:
        hard_negative_predicates = [
            str(item).strip().upper()
            for item in candidate.get("hard_negative_predicates", [])
            if str(item).strip().upper() in (target_relation_set | {NA_RELATION_LABEL})
        ]
        candidate_predicates = [
            str(item).strip().upper()
            for item in candidate.get("candidate_predicates", [candidate.get("predicate", "")])
            if str(item).strip().upper() in target_relation_set
        ]
        if not candidate_predicates and not hard_negative_predicates:
            continue
        positive_predicates = [
            str(item).strip().upper()
            for item in candidate.get("positive_predicates", [])
            if str(item).strip().upper() in target_relation_set
        ]
        if include_gold_labels and not positive_predicates and not hard_negative_predicates:
            continue
        subject_id = str(candidate.get("subject_entity_id", "")).strip()
        object_id = str(candidate.get("object_entity_id", "")).strip()
        sentence_id = str(candidate.get("sentence_id", "")).strip()
        doc_id = str(candidate.get("doc_id", "")).strip()
        if not subject_id or not object_id or not sentence_id or not doc_id:
            continue
        evidence = _candidate_to_sentence_evidence(candidate)
        bag_id = f"{doc_id}__{subject_id}__{object_id}"
        bag_payload = bag_payloads.setdefault(
            bag_id,
            {
                "doc_id": doc_id,
                "subject_id": subject_id,
                "object_id": object_id,
                "subject_type": str(candidate.get("subject_entity_type", "Entity")),
                "object_type": str(candidate.get("object_entity_type", "Entity")),
                "allowed_predicates": set(),
                "sentence_evidences": {},
                "label_names": set(),
                "hard_negative_predicates": set(),
            },
        )
        bag_payload["allowed_predicates"].update(
            str(item).strip().upper()
            for item in candidate.get("allowed_predicates", candidate_predicates)
            if str(item).strip().upper() in target_relation_set
        )
        evidence_key = (
            evidence.sentence_id,
            evidence.subject_mention_id,
            evidence.object_mention_id,
        )
        existing_evidence = bag_payload["sentence_evidences"].get(evidence_key)
        if existing_evidence is None:
            bag_payload["sentence_evidences"][evidence_key] = evidence
        else:
            bag_payload["sentence_evidences"][evidence_key] = _merge_sentence_evidence(existing_evidence, evidence)
        if include_gold_labels:
            bag_payload["label_names"].update(positive_predicates)
            bag_payload["hard_negative_predicates"].update(hard_negative_predicates)
    bags: list[RelationBag] = []
    for bag_id, payload in sorted(bag_payloads.items()):
        if include_gold_labels and not payload["label_names"] and not payload["hard_negative_predicates"]:
            continue
        label_names = sorted(payload["label_names"]) if include_gold_labels and payload["label_names"] else [NA_RELATION_LABEL]
        bags.append(
            RelationBag(
                bag_id=bag_id,
                doc_id=str(payload["doc_id"]),
                subject_id=str(payload["subject_id"]),
                object_id=str(payload["object_id"]),
                subject_type=str(payload["subject_type"]),
                object_type=str(payload["object_type"]),
                allowed_predicates=sorted(payload["allowed_predicates"]),
                sentence_evidences=sorted(
                    payload["sentence_evidences"].values(),
                    key=lambda item: (item.doc_id, item.sentence_id, item.subject_span[0], item.object_span[0]),
                )[: config.model.max_sentences_per_bag],
                label_names=label_names,
            )
        )
    return bags, target_relations


def _build_relation_bags_from_distant_labels(
    config: RelationExtractionConfig,
    *,
    include_gold_labels: bool,
) -> tuple[list[RelationBag], list[str]]:
    return _build_relation_bags_from_candidate_records(
        config,
        candidate_records=read_jsonl(config.data.distant_labeled_path),
        include_gold_labels=include_gold_labels,
    )


def _build_relation_bags_from_pair_candidates(config: RelationExtractionConfig) -> tuple[list[RelationBag], list[str]]:
    return _build_relation_bags_from_candidate_records(
        config,
        candidate_records=read_jsonl(config.data.pair_candidates_path),
        include_gold_labels=False,
    )


def build_relation_bags(
    config: RelationExtractionConfig,
    *,
    include_gold_labels: bool,
) -> tuple[list[RelationBag], list[str]]:
    if include_gold_labels:
        if not config.data.distant_labeled_path.exists():
            raise FileNotFoundError(
                "训练/评估关系 bag 必须读取 distant_labeled.jsonl。"
                f" 当前文件不存在：{config.data.distant_labeled_path.as_posix()}"
            )
        return _build_relation_bags_from_distant_labels(config, include_gold_labels=include_gold_labels)
    if not config.data.pair_candidates_path.exists():
        raise FileNotFoundError(
            "预测关系 bag 必须读取 pair_candidates.jsonl。"
            f" 当前文件不存在：{config.data.pair_candidates_path.as_posix()}"
        )
    return _build_relation_bags_from_pair_candidates(config)


def split_relation_bags(
    bags: list[RelationBag],
    *,
    train_ratio: float,
    dev_ratio: float,
    random_seed: int,
) -> tuple[list[RelationBag], list[RelationBag], list[RelationBag]]:
    bags_by_doc: dict[str, list[RelationBag]] = defaultdict(list)
    for bag in bags:
        bags_by_doc[bag.doc_id].append(bag)
    ordered_doc_ids = sorted(bags_by_doc)
    sampler = random.Random(random_seed)
    sampler.shuffle(ordered_doc_ids)
    total_count = len(ordered_doc_ids)
    train_end = int(total_count * train_ratio)
    dev_end = train_end + int(total_count * dev_ratio)
    train_doc_ids = set(ordered_doc_ids[:train_end])
    dev_doc_ids = set(ordered_doc_ids[train_end:dev_end])
    test_doc_ids = set(ordered_doc_ids[dev_end:])
    train_bags = sorted(
        [bag for bag in bags if bag.doc_id in train_doc_ids],
        key=lambda item: item.bag_id,
    )
    dev_bags = sorted(
        [bag for bag in bags if bag.doc_id in dev_doc_ids],
        key=lambda item: item.bag_id,
    )
    test_bags = sorted(
        [bag for bag in bags if bag.doc_id in test_doc_ids],
        key=lambda item: item.bag_id,
    )
    return train_bags, dev_bags, test_bags


def downsample_na_bags(
    bags: list[RelationBag],
    *,
    na_downsample_ratio: float,
    random_seed: int,
) -> tuple[list[RelationBag], dict[str, int]]:
    positive_bags = [bag for bag in bags if not bag.is_na]
    na_bags = [bag for bag in bags if bag.is_na]
    if na_downsample_ratio <= 0 or not positive_bags or not na_bags:
        return bags, {
            "positive_before": len(positive_bags),
            "na_before": len(na_bags),
            "positive_after": len(positive_bags),
            "na_after": len(na_bags),
        }
    target_na_count = min(len(na_bags), max(1, int(len(positive_bags) * na_downsample_ratio)))
    sampler = random.Random(random_seed)
    sampled_na_bags = sampler.sample(na_bags, target_na_count)
    merged_bags = sorted(positive_bags + sampled_na_bags, key=lambda item: item.bag_id)
    return merged_bags, {
        "positive_before": len(positive_bags),
        "na_before": len(na_bags),
        "positive_after": len(positive_bags),
        "na_after": len(sampled_na_bags),
    }


def build_label_maps(target_relations: list[str]) -> tuple[dict[str, int], list[str]]:
    index_to_label = [NA_RELATION_LABEL] + sorted(target_relations)
    label_to_index = {label: index for index, label in enumerate(index_to_label)}
    return label_to_index, index_to_label


def bag_to_label_vector(bag: RelationBag, label_to_index: dict[str, int]) -> list[float]:
    vector = [0.0] * len(label_to_index)
    for label_name in bag.label_names:
        label_index = label_to_index.get(label_name)
        if label_index is None:
            continue
        vector[label_index] = 1.0
    return vector


def compute_class_weights(bags: list[RelationBag], label_to_index: dict[str, int]) -> dict[str, float]:
    total_count = len(bags)
    if total_count == 0:
        return {label_name: 1.0 for label_name in label_to_index}
    positive_counter = Counter[str]()
    for bag in bags:
        for label_name in bag.label_names:
            positive_counter[label_name] += 1
    class_weights: dict[str, float] = {}
    for label_name in label_to_index:
        positive_count = positive_counter.get(label_name, 0)
        if positive_count <= 0:
            class_weights[label_name] = 1.0
            continue
        negative_count = max(total_count - positive_count, 1)
        class_weights[label_name] = round(negative_count / positive_count, 6)
    return class_weights


def build_vocabulary(
    bags: list[RelationBag],
    *,
    min_token_frequency: int,
    lowercase_tokens: bool,
) -> Vocabulary:
    counter: Counter[str] = Counter()
    for bag in bags:
        for evidence in bag.sentence_evidences:
            counter.update(normalize_token(token, lowercase_tokens) for token in evidence.tokens)
    index_to_token = [PAD_TOKEN, UNK_TOKEN]
    for token, count in sorted(counter.items()):
        if count >= min_token_frequency:
            index_to_token.append(token)
    token_to_index = {token: index for index, token in enumerate(index_to_token)}
    return Vocabulary(
        token_to_index=token_to_index,
        index_to_token=index_to_token,
        lowercase_tokens=lowercase_tokens,
    )


def _crop_sentence_to_pair_window(
    tokens: list[str],
    token_spans: list[list[int]],
    subject_span: tuple[int, int],
    object_span: tuple[int, int],
    *,
    max_sentence_length: int,
) -> tuple[list[str], list[list[int]], tuple[int, int], tuple[int, int]]:
    if len(tokens) <= max_sentence_length:
        return tokens, token_spans, subject_span, object_span
    pair_left = min(subject_span[0], object_span[0])
    pair_right = max(subject_span[1], object_span[1])
    required_width = pair_right - pair_left
    if required_width >= max_sentence_length:
        window_end = min(len(tokens), pair_right)
        window_start = max(0, window_end - max_sentence_length)
    else:
        extra_context = max_sentence_length - required_width
        window_start = max(0, pair_left - extra_context // 2)
        window_end = min(len(tokens), window_start + max_sentence_length)
        window_start = max(0, window_end - max_sentence_length)
    cropped_tokens = tokens[window_start:window_end]
    cropped_spans = token_spans[window_start:window_end]
    cropped_subject_span = (subject_span[0] - window_start, subject_span[1] - window_start)
    cropped_object_span = (object_span[0] - window_start, object_span[1] - window_start)
    return cropped_tokens, cropped_spans, cropped_subject_span, cropped_object_span


def _build_piece_ids(
    sequence_length: int,
    subject_span: tuple[int, int],
    object_span: tuple[int, int],
) -> list[int]:
    first_start = min(subject_span[0], object_span[0])
    second_start = max(subject_span[0], object_span[0])
    piece_ids: list[int] = []
    for token_index in range(sequence_length):
        if token_index <= first_start:
            piece_ids.append(1)
        elif token_index < second_start:
            piece_ids.append(2)
        else:
            piece_ids.append(3)
    return piece_ids


def vectorize_sentence_evidence(
    evidence: RelationSentenceEvidence,
    vocabulary: Vocabulary,
    *,
    max_sentence_length: int,
    position_clip: int,
) -> dict[str, Any]:
    original_subject_span = list(evidence.subject_span)
    original_object_span = list(evidence.object_span)
    tokens, token_spans, subject_span, object_span = _crop_sentence_to_pair_window(
        evidence.tokens,
        evidence.token_spans,
        evidence.subject_span,
        evidence.object_span,
        max_sentence_length=max_sentence_length,
    )
    token_ids: list[int] = []
    position1_ids: list[int] = []
    position2_ids: list[int] = []
    piece_ids = _build_piece_ids(len(tokens), subject_span, object_span)
    for token_index, token in enumerate(tokens):
        normalized_token = normalize_token(token, vocabulary.lowercase_tokens)
        token_ids.append(vocabulary.token_to_index.get(normalized_token, vocabulary.token_to_index[UNK_TOKEN]))
        position1_ids.append(position_to_index(token_index - subject_span[0], position_clip))
        position2_ids.append(position_to_index(token_index - object_span[0], position_clip))
    attention_mask = [1] * len(tokens)
    while len(token_ids) < max_sentence_length:
        token_ids.append(vocabulary.token_to_index[PAD_TOKEN])
        position1_ids.append(position_to_index(0, position_clip, is_padding=True))
        position2_ids.append(position_to_index(0, position_clip, is_padding=True))
        piece_ids.append(0)
        attention_mask.append(0)
    return {
        "sentence_id": evidence.sentence_id,
        "doc_id": evidence.doc_id,
        "source_id": evidence.source_id,
        "sentence_index_in_doc": evidence.sentence_index_in_doc,
        "candidate_id": evidence.candidate_id,
        "predicate": evidence.predicate,
        "token_ids": token_ids,
        "position1_ids": position1_ids,
        "position2_ids": position2_ids,
        "piece_ids": piece_ids,
        "attention_mask": attention_mask,
        "subject_span": list(subject_span),
        "object_span": list(object_span),
        "original_subject_span": original_subject_span,
        "original_object_span": original_object_span,
        "subject_mention_id": evidence.subject_mention_id,
        "object_mention_id": evidence.object_mention_id,
        "subject_text": evidence.subject_text,
        "object_text": evidence.object_text,
        "pair_source": evidence.pair_source,
        "exact_claim_match": evidence.exact_claim_match,
        "matched_claim_ids": list(evidence.matched_claim_ids or []),
        "bridge_predicates": list(evidence.bridge_predicates or []),
        "bridge_details": dict(evidence.bridge_details or {}),
        "weak_label": evidence.weak_label,
        "weak_label_reason": evidence.weak_label_reason,
        "supervision_tier": evidence.supervision_tier,
        "allowed_predicates": list(evidence.allowed_predicates or []),
        "candidate_predicates": list(evidence.candidate_predicates or []),
        "positive_predicates": list(evidence.positive_predicates or []),
        "hard_negative_predicates": list(evidence.hard_negative_predicates or []),
        "unknown_predicates": list(evidence.unknown_predicates or []),
        "review_predicates": list(evidence.review_predicates or []),
        "weak_labels_by_predicate": dict(evidence.weak_labels_by_predicate or {}),
        "sentence_trigger_hits": dict(evidence.sentence_trigger_hits or {}),
        "local_trigger_hits": dict(evidence.local_trigger_hits or {}),
        "exact_claim_matches": dict(evidence.exact_claim_matches or {}),
        "candidate_strength_by_predicate": dict(evidence.candidate_strength_by_predicate or {}),
        "text": evidence.text,
        "token_spans": token_spans,
    }


def vectorize_bag(
    *,
    bag: RelationBag,
    vocabulary: Vocabulary,
    label_to_index: dict[str, int],
    max_sentence_length: int,
    position_clip: int,
) -> dict[str, Any]:
    return {
        "bag_id": bag.bag_id,
        "doc_id": bag.doc_id,
        "subject_id": bag.subject_id,
        "object_id": bag.object_id,
        "subject_type": bag.subject_type,
        "object_type": bag.object_type,
        "allowed_predicates": list(bag.allowed_predicates),
        "sentence_features": [
            vectorize_sentence_evidence(
                evidence,
                vocabulary,
                max_sentence_length=max_sentence_length,
                position_clip=position_clip,
            )
            for evidence in bag.sentence_evidences
        ],
        "label_names": list(bag.label_names),
        "label_vector": bag_to_label_vector(bag, label_to_index),
    }


def collate_relation_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    flat_token_ids: list[list[int]] = []
    flat_position1_ids: list[list[int]] = []
    flat_position2_ids: list[list[int]] = []
    flat_piece_ids: list[list[int]] = []
    flat_attention_mask: list[list[int]] = []
    flat_sentence_ids: list[str] = []
    flat_sentence_texts: list[str] = []
    flat_evidence_metadata: list[dict[str, Any]] = []
    bag_scopes: list[tuple[int, int]] = []
    label_vectors: list[list[float]] = []
    bag_metadata: list[dict[str, Any]] = []

    cursor = 0
    for bag in batch:
        sentence_features = bag["sentence_features"]
        bag_start = cursor
        for feature in sentence_features:
            flat_token_ids.append(feature["token_ids"])
            flat_position1_ids.append(feature["position1_ids"])
            flat_position2_ids.append(feature["position2_ids"])
            flat_piece_ids.append(feature["piece_ids"])
            flat_attention_mask.append(feature["attention_mask"])
            flat_sentence_ids.append(feature["sentence_id"])
            flat_sentence_texts.append(feature["text"])
            flat_evidence_metadata.append(
                {
                    "candidate_id": feature.get("candidate_id", ""),
                    "sentence_id": feature["sentence_id"],
                    "doc_id": feature.get("doc_id", ""),
                    "source_id": feature.get("source_id", ""),
                    "sentence_index_in_doc": feature.get("sentence_index_in_doc"),
                    "text": feature["text"],
                    "predicate": feature.get("predicate", ""),
                    "subject_mention_id": feature.get("subject_mention_id", ""),
                    "object_mention_id": feature.get("object_mention_id", ""),
                    "subject_text": feature.get("subject_text", ""),
                    "object_text": feature.get("object_text", ""),
                    "subject_span": feature.get("subject_span", []),
                    "object_span": feature.get("object_span", []),
                    "original_subject_span": feature.get("original_subject_span", []),
                    "original_object_span": feature.get("original_object_span", []),
                    "pair_source": feature.get("pair_source", ""),
                    "exact_claim_match": feature.get("exact_claim_match", False),
                    "matched_claim_ids": feature.get("matched_claim_ids", []),
                    "bridge_predicates": feature.get("bridge_predicates", []),
                    "bridge_details": feature.get("bridge_details", {}),
                    "weak_label": feature.get("weak_label", ""),
                    "weak_label_reason": feature.get("weak_label_reason", ""),
                    "supervision_tier": feature.get("supervision_tier", ""),
                    "allowed_predicates": feature.get("allowed_predicates", []),
                    "candidate_predicates": feature.get("candidate_predicates", []),
                    "positive_predicates": feature.get("positive_predicates", []),
                    "hard_negative_predicates": feature.get("hard_negative_predicates", []),
                    "unknown_predicates": feature.get("unknown_predicates", []),
                    "review_predicates": feature.get("review_predicates", []),
                    "weak_labels_by_predicate": feature.get("weak_labels_by_predicate", {}),
                    "sentence_trigger_hits": feature.get("sentence_trigger_hits", {}),
                    "local_trigger_hits": feature.get("local_trigger_hits", {}),
                    "exact_claim_matches": feature.get("exact_claim_matches", {}),
                    "candidate_strength_by_predicate": feature.get("candidate_strength_by_predicate", {}),
                }
            )
            cursor += 1
        bag_scopes.append((bag_start, cursor))
        label_vectors.append(list(bag["label_vector"]))
        bag_metadata.append(
            {
                "bag_id": bag["bag_id"],
                "doc_id": bag.get("doc_id", ""),
                "subject_id": bag["subject_id"],
                "object_id": bag["object_id"],
                "subject_type": bag["subject_type"],
                "object_type": bag["object_type"],
                "allowed_predicates": bag.get("allowed_predicates", []),
                "label_names": bag["label_names"],
            }
        )
    return {
        "token_ids": flat_token_ids,
        "position1_ids": flat_position1_ids,
        "position2_ids": flat_position2_ids,
        "piece_ids": flat_piece_ids,
        "attention_mask": flat_attention_mask,
        "sentence_ids": flat_sentence_ids,
        "sentence_texts": flat_sentence_texts,
        "evidence_metadata": flat_evidence_metadata,
        "bag_scopes": bag_scopes,
        "bag_metadata": bag_metadata,
        "label_vectors": label_vectors,
    }


def build_dataset_report(
    train_bags: list[RelationBag],
    dev_bags: list[RelationBag],
    test_bags: list[RelationBag],
    *,
    sampled_train_bags: list[RelationBag],
    label_to_index: dict[str, int],
    downsampling_report: dict[str, int],
) -> dict[str, Any]:
    def summarize(split_bags: list[RelationBag]) -> dict[str, Any]:
        relation_counter: Counter[str] = Counter()
        sentence_counter = 0
        hard_negative_bag_count = 0
        for bag in split_bags:
            sentence_counter += len(bag.sentence_evidences)
            relation_counter.update(bag.label_names)
            if bag.is_na and any(
                NA_RELATION_LABEL in set(evidence.hard_negative_predicates or [])
                for evidence in bag.sentence_evidences
            ):
                hard_negative_bag_count += 1
        return {
            "bag_count": len(split_bags),
            "sentence_count": sentence_counter,
            "relation_counts": dict(sorted(relation_counter.items())),
            "na_bag_count": sum(1 for bag in split_bags if bag.is_na),
            "hard_negative_bag_count": hard_negative_bag_count,
            "positive_bag_count": sum(1 for bag in split_bags if not bag.is_na),
        }

    def positive_counts(split_bags: list[RelationBag]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for bag in split_bags:
            counter.update(label for label in bag.label_names if label != NA_RELATION_LABEL)
        return counter

    train_counts = positive_counts(train_bags)
    dev_counts = positive_counts(dev_bags)
    test_counts = positive_counts(test_bags)
    all_counts = train_counts + dev_counts + test_counts
    target_relations = [label for label in label_to_index if label != NA_RELATION_LABEL]
    warnings: list[str] = []
    for relation_name in target_relations:
        if train_counts.get(relation_name, 0) == 0:
            warnings.append(f"train split 缺少 {relation_name} 正例，训练该关系会不稳定。")
        if dev_counts.get(relation_name, 0) == 0:
            warnings.append(f"dev split 缺少 {relation_name} 正例，dev macro F1 对该关系不可用。")
        if test_counts.get(relation_name, 0) == 0:
            warnings.append(f"test split 缺少 {relation_name} 正例，test macro F1 对该关系不可用。")
        if all_counts.get(relation_name, 0) == 1:
            warnings.append(f"{relation_name} 全量只有 1 个正例，指标方差会很高。")

    sampled_summary = summarize(sampled_train_bags)
    if sampled_summary["hard_negative_bag_count"] == 0:
        warnings.append("当前训练集没有显式 hard negative / NA bag，模型只学习正例和 implicit negative。")

    return {
        "label_space": list(label_to_index),
        "train_before_downsampling": summarize(train_bags),
        "train_after_downsampling": sampled_summary,
        "dev": summarize(dev_bags),
        "test": summarize(test_bags),
        "na_downsampling": downsampling_report,
        "coverage_warnings": warnings,
    }


def prepare_training_data(config: RelationExtractionConfig) -> PreparedRelationDataset:
    bags, target_relations = build_relation_bags(config, include_gold_labels=True)
    positive_bags = [bag for bag in bags if not bag.is_na]
    if not positive_bags:
        raise RuntimeError(
            "当前远程监督样本里没有任何 ds_strict 正例 bag，无法启动关系模型训练。"
            "请先检查 trigger 规则、linking 覆盖率与 claims 对齐情况。"
        )
    label_to_index, index_to_label = build_label_maps(target_relations)
    train_bags, dev_bags, test_bags = split_relation_bags(
        bags,
        train_ratio=config.training.train_ratio,
        dev_ratio=config.training.dev_ratio,
        random_seed=config.training.random_seed,
    )
    sampled_train_bags, downsampling_report = downsample_na_bags(
        train_bags,
        na_downsample_ratio=config.training.na_downsample_ratio,
        random_seed=config.training.random_seed,
    )
    vocabulary = build_vocabulary(
        sampled_train_bags,
        min_token_frequency=config.embeddings.min_token_frequency,
        lowercase_tokens=config.embeddings.lowercase_tokens,
    )
    class_weights = compute_class_weights(sampled_train_bags, label_to_index)
    dataset_report = build_dataset_report(
        train_bags,
        dev_bags,
        test_bags,
        sampled_train_bags=sampled_train_bags,
        label_to_index=label_to_index,
        downsampling_report=downsampling_report,
    )
    return PreparedRelationDataset(
        train_bags=sampled_train_bags,
        dev_bags=dev_bags,
        test_bags=test_bags,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        vocabulary=vocabulary,
        class_weights=class_weights,
        dataset_report=dataset_report,
    )


def select_bags_by_ids(bags: list[RelationBag], bag_ids: set[str]) -> list[RelationBag]:
    return [bag for bag in bags if bag.bag_id in bag_ids]
