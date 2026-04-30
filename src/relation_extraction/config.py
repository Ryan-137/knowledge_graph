from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

POSITION_CLIP_VALUE = 40
POSITION_PADDING_INDEX = 0
NA_RELATION_LABEL = "NA"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
DEFAULT_EXTRACTOR_NAME = "pcnn_mil_attention"
DEFAULT_TARGET_RELATIONS = (
    "BORN_IN",
    "DIED_IN",
    "STUDIED_AT",
    "WORKED_AT",
    "AUTHORED",
    "PROPOSED",
    "DESIGNED",
    "AWARDED",
    "LOCATED_IN",
)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _serialize_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


@dataclass(frozen=True)
class RelationDataPaths:
    """关系抽取训练/推理所依赖的数据输入与工件输出路径。"""

    sentences_path: Path = Path("knowledge_graph/data/processed/unstructured/sentences.jsonl")
    tokenized_sentences_path: Path = Path("knowledge_graph/data/processed/mentions/tokenized_sentences.jsonl")
    resolved_mentions_path: Path = Path("knowledge_graph/data/processed/coreference/resolved_mentions.jsonl")
    claims_path: Path = Path("knowledge_graph/data/processed/structured/csv/claims.csv")
    entities_path: Path = Path("knowledge_graph/data/processed/structured/csv/entities.csv")
    aliases_path: Path = Path("knowledge_graph/data/processed/structured/csv/aliases.csv")
    ontology_path: Path = Path("knowledge_graph/knowledge/ontology.json")
    pair_candidates_path: Path = Path("knowledge_graph/data/processed/relations/pair_candidates.jsonl")
    distant_labeled_path: Path = Path("knowledge_graph/data/processed/relations/distant_labeled.jsonl")
    relation_gold_path: Path = Path("knowledge_graph/data/processed/relations/relation_gold.jsonl")
    output_dir: Path = Path("knowledge_graph/data/processed/relations/model")


@dataclass(frozen=True)
class EmbeddingConfig:
    """词向量与词表配置。"""

    pretrained_txt_path: Path | None = None
    embedding_dim: int = 100
    position_embedding_dim: int = 16
    min_token_frequency: int = 1
    lowercase_tokens: bool = True
    initializer_range: float = 0.05


@dataclass(frozen=True)
class RelationModelConfig:
    """PCNN + MIL-Attention 模型超参数。"""

    max_sentence_length: int = 128
    max_sentences_per_bag: int = 16
    convolution_channels: int = 128
    convolution_kernel_sizes: list[int] = field(default_factory=lambda: [3, 5])
    dropout: float = 0.3
    relation_threshold: float = 0.5
    threshold_by_relation: dict[str, float] = field(default_factory=dict)
    top_k_support_sentences: int = 3
    prediction_mask_mode: str = "candidate"


@dataclass(frozen=True)
class RelationTrainingConfig:
    """训练与采样配置。"""

    random_seed: int = 2026
    batch_size: int = 16
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    train_ratio: float = 0.7
    dev_ratio: float = 0.15
    na_downsample_ratio: float = 3.0
    log_every_steps: int = 10
    device: str = "auto"
    early_stop_patience: int = 5
    use_class_weights: bool = True


@dataclass(frozen=True)
class RelationExtractionConfig:
    """
    关系抽取统一配置。

    target_relations 为空时，会自动从 `claims.csv` 与 `ontology.json` 的交集中推断训练标签集合。
    """

    data: RelationDataPaths = field(default_factory=RelationDataPaths)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: RelationModelConfig = field(default_factory=RelationModelConfig)
    training: RelationTrainingConfig = field(default_factory=RelationTrainingConfig)
    target_relations: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_RELATIONS))

    @property
    def position_clip(self) -> int:
        return POSITION_CLIP_VALUE

    @property
    def position_vocab_size(self) -> int:
        # 0 专门保留给 padding，1..81 对应 [-40, 40]。
        return POSITION_CLIP_VALUE * 2 + 2

    def to_dict(self) -> dict[str, Any]:
        return _serialize_value(asdict(self))


def _read_config_payload(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"关系抽取配置文件不存在：{config_path.as_posix()}")
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"关系抽取配置文件不是 JSON 对象：{config_path.as_posix()}")
    return payload


def _ontology_relation_names(ontology: dict[str, Any]) -> set[str]:
    return {
        str(relation.get("name", "")).strip().upper()
        for relation in ontology.get("relations", [])
        if str(relation.get("name", "")).strip()
    }


def _read_ontology_relation_names(ontology_path: Path) -> set[str]:
    if not ontology_path.exists():
        raise FileNotFoundError(f"关系抽取 ontology 文件不存在：{ontology_path.as_posix()}")
    ontology = json.loads(ontology_path.read_text(encoding="utf-8-sig"))
    if not isinstance(ontology, dict):
        raise ValueError(f"关系抽取 ontology 文件不是 JSON 对象：{ontology_path.as_posix()}")
    return _ontology_relation_names(ontology)


def resolve_target_relations(
    *,
    ontology_path: Path,
    configured_target_relations: Sequence[str] | None,
    trigger_relation_names: Sequence[str],
) -> list[str]:
    """统一解析关系抽取链路的目标关系集合。"""

    requested_relations = [
        str(relation_name).strip().upper()
        for relation_name in (configured_target_relations or DEFAULT_TARGET_RELATIONS)
        if str(relation_name).strip()
    ]
    if not requested_relations:
        raise ValueError("关系抽取 target_relations 为空，无法确定训练/预测关系集合。")

    ontology_relations = _read_ontology_relation_names(ontology_path)
    trigger_relations = {str(relation_name).strip().upper() for relation_name in trigger_relation_names}
    missing_in_ontology = sorted(set(requested_relations) - ontology_relations)
    if missing_in_ontology:
        raise ValueError(f"关系抽取配置包含 ontology 未定义关系：{missing_in_ontology}")
    missing_triggers = sorted(set(requested_relations) - trigger_relations)
    if missing_triggers:
        raise ValueError(f"关系抽取配置包含缺少 trigger 规则的关系：{missing_triggers}")
    return sorted(set(requested_relations) & ontology_relations)


def _resolve_path(raw_value: str | Path | None) -> Path | None:
    if raw_value in (None, ""):
        return None
    return Path(raw_value)


def load_relation_extraction_config(
    *,
    config_path: Path | None,
    output_dir: Path,
    sentences_path: Path,
    tokenized_sentences_path: Path,
    resolved_mentions_path: Path,
    entities_path: Path,
    aliases_path: Path,
    claims_path: Path,
    ontology_path: Path,
    pair_candidates_path: Path,
    distant_labeled_path: Path,
    target_relations: Sequence[str] | None = None,
    relation_threshold: float | None = None,
    random_seed: int | None = None,
    dev_ratio: float | None = None,
) -> RelationExtractionConfig:
    payload = _read_config_payload(config_path)
    prediction_mask_mode = str(payload.get("prediction_mask_mode", "candidate")).strip() or "candidate"
    if prediction_mask_mode not in {"ontology", "candidate", "ontology-and-candidate"}:
        raise ValueError(f"未知 prediction_mask_mode：{prediction_mask_mode}")
    configured_embedding_path = _resolve_path(payload.get("embedding_path"))
    embeddings = EmbeddingConfig(
        pretrained_txt_path=configured_embedding_path,
        embedding_dim=int(payload.get("embedding_dim", 100)),
        position_embedding_dim=int(payload.get("position_embedding_dim", 16)),
        min_token_frequency=int(payload.get("min_token_frequency", 1)),
        initializer_range=float(payload.get("initializer_range", 0.05)),
    )
    model = RelationModelConfig(
        max_sentence_length=int(payload.get("max_seq_len", 128)),
        max_sentences_per_bag=int(payload.get("max_sentences_per_bag", 16)),
        convolution_channels=int(payload.get("cnn_hidden_size", 128)),
        convolution_kernel_sizes=[int(item) for item in payload.get("cnn_kernel_sizes", [3, 5])],
        dropout=float(payload.get("dropout", 0.3)),
        relation_threshold=float(relation_threshold if relation_threshold is not None else payload.get("relation_threshold", 0.5)),
        threshold_by_relation={
            str(relation_name).strip().upper(): float(threshold)
            for relation_name, threshold in dict(payload.get("threshold_by_relation", {})).items()
            if str(relation_name).strip()
        },
        top_k_support_sentences=int(payload.get("top_k_support_sentences", 3)),
        prediction_mask_mode=prediction_mask_mode,
    )
    effective_dev_ratio = float(dev_ratio if dev_ratio is not None else payload.get("dev_ratio", 0.15))
    train_ratio = float(payload.get("train_ratio", max(0.1, 1.0 - effective_dev_ratio - 0.15)))
    training = RelationTrainingConfig(
        random_seed=int(random_seed if random_seed is not None else payload.get("seed", payload.get("random_seed", 2026))),
        batch_size=int(payload.get("batch_size", 16)),
        num_epochs=int(payload.get("max_epochs", 25)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-5)),
        gradient_clip_norm=float(payload.get("gradient_clip_norm", 5.0)),
        train_ratio=train_ratio,
        dev_ratio=effective_dev_ratio,
        na_downsample_ratio=float(payload.get("na_downsample_ratio", 3.0)),
        log_every_steps=int(payload.get("log_every_steps", 10)),
        early_stop_patience=int(payload.get("early_stop_patience", 5)),
        use_class_weights=bool(payload.get("use_class_weights", True)),
        device=str(payload.get("device", "auto")),
    )
    from relation_extraction.rules import SUPPORTED_RELATION_NAMES

    normalized_target_relations = resolve_target_relations(
        ontology_path=ontology_path,
        configured_target_relations=target_relations or payload.get("target_relations"),
        trigger_relation_names=SUPPORTED_RELATION_NAMES,
    )
    return RelationExtractionConfig(
        data=RelationDataPaths(
            sentences_path=sentences_path,
            tokenized_sentences_path=tokenized_sentences_path,
            resolved_mentions_path=resolved_mentions_path,
            claims_path=claims_path,
            entities_path=entities_path,
            aliases_path=aliases_path,
            ontology_path=ontology_path,
            pair_candidates_path=pair_candidates_path,
            distant_labeled_path=distant_labeled_path,
            output_dir=output_dir,
        ),
        embeddings=embeddings,
        model=model,
        training=training,
        target_relations=normalized_target_relations,
    )
