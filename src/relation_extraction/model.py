from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PAD_TOKEN, RelationExtractionConfig
from .dataset import Vocabulary

_TORCH_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - 当前环境缺 torch，主要走清晰报错分支
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


def require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise RuntimeError(
            "当前 knowgraph 环境缺少 torch，无法执行关系抽取训练/推理。"
            "请先安装与本机 CUDA/CPU 匹配的 PyTorch，再重新运行。"
        ) from _TORCH_IMPORT_ERROR
    return torch, nn


def _read_pretrained_txt_vectors(
    embedding_path: Path,
    *,
    lowercase_tokens: bool,
) -> tuple[dict[str, list[float]], int]:
    vectors: dict[str, list[float]] = {}
    inferred_dimension: int | None = None
    with embedding_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                # 兼容 word2vec 文本首行 header: `vocab_size embedding_dim`
                continue
            if len(parts) < 2:
                continue
            token = parts[0].lower() if lowercase_tokens else parts[0]
            try:
                vector = [float(item) for item in parts[1:]]
            except ValueError as exc:  # pragma: no cover - 数据坏行
                raise ValueError(f"{embedding_path.as_posix()} 第 {line_number} 行不是合法词向量。") from exc
            if inferred_dimension is None:
                inferred_dimension = len(vector)
            if len(vector) != inferred_dimension:
                continue
            vectors[token] = vector
    if inferred_dimension is None:
        raise ValueError(f"{embedding_path.as_posix()} 中没有读到任何合法词向量。")
    return vectors, inferred_dimension


if nn is not None:

    class PiecewiseMaxPooling(nn.Module):
        """PCNN 的分段最大池化。"""

        def forward(self, convolution_output: Any, piece_ids: Any) -> Any:
            negative_infinity = -1e4
            pooled_outputs: list[Any] = []
            for piece_value in (1, 2, 3):
                piece_mask = piece_ids.eq(piece_value).unsqueeze(1)
                masked_output = convolution_output.masked_fill(~piece_mask, negative_infinity)
                piece_pooled = masked_output.max(dim=2).values
                # 某段完全不存在时，max 会退化为负无穷，这里显式置零，避免污染梯度。
                piece_pooled = torch.where(
                    piece_pooled <= negative_infinity / 2,
                    torch.zeros_like(piece_pooled),
                    piece_pooled,
                )
                pooled_outputs.append(piece_pooled)
            return torch.cat(pooled_outputs, dim=1)


    class PCNNSentenceEncoder(nn.Module):
        """句子级 PCNN 编码器。"""

        def __init__(
            self,
            config: RelationExtractionConfig,
            vocabulary: Vocabulary,
        ) -> None:
            super().__init__()
            self.config = config
            self.vocabulary = vocabulary
            embedding_matrix, embedding_report = self._build_word_embedding_matrix()
            self.embedding_report = embedding_report

            self.word_embedding = nn.Embedding.from_pretrained(
                embedding_matrix,
                freeze=False,
                padding_idx=vocabulary.token_to_index[PAD_TOKEN],
            )
            self.position1_embedding = nn.Embedding(
                num_embeddings=config.position_vocab_size,
                embedding_dim=config.embeddings.position_embedding_dim,
                padding_idx=0,
            )
            self.position2_embedding = nn.Embedding(
                num_embeddings=config.position_vocab_size,
                embedding_dim=config.embeddings.position_embedding_dim,
                padding_idx=0,
            )
            input_dimension = (
                self.word_embedding.embedding_dim
                + config.embeddings.position_embedding_dim * 2
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=input_dimension,
                        out_channels=config.model.convolution_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                    for kernel_size in config.model.convolution_kernel_sizes
                ]
            )
            self.activation = nn.ReLU()
            self.pooling = PiecewiseMaxPooling()
            self.dropout = nn.Dropout(config.model.dropout)
            self.output_dimension = config.model.convolution_channels * 3 * len(config.model.convolution_kernel_sizes)

        def _build_word_embedding_matrix(self) -> tuple[Any, dict[str, Any]]:
            vocabulary_size = len(self.vocabulary.index_to_token)
            embedding_dimension = self.config.embeddings.embedding_dim
            embedding_matrix = torch.empty(vocabulary_size, embedding_dimension)
            nn.init.uniform_(
                embedding_matrix,
                -self.config.embeddings.initializer_range,
                self.config.embeddings.initializer_range,
            )
            embedding_matrix[self.vocabulary.token_to_index[PAD_TOKEN]].zero_()
            covered_count = 0
            if self.config.embeddings.pretrained_txt_path is not None:
                pretrained_vectors, inferred_dimension = _read_pretrained_txt_vectors(
                    self.config.embeddings.pretrained_txt_path,
                    lowercase_tokens=self.vocabulary.lowercase_tokens,
                )
                if inferred_dimension != embedding_dimension:
                    raise ValueError(
                        f"预训练词向量维度为 {inferred_dimension}，但配置 embedding_dim={embedding_dimension}。"
                    )
                for index, token in enumerate(self.vocabulary.index_to_token):
                    vector = pretrained_vectors.get(token)
                    if vector is None:
                        continue
                    embedding_matrix[index] = torch.tensor(vector, dtype=torch.float32)
                    covered_count += 1
            return embedding_matrix, {
                "vocabulary_size": vocabulary_size,
                "embedding_dim": embedding_dimension,
                "pretrained_txt_path": (
                    self.config.embeddings.pretrained_txt_path.as_posix()
                    if self.config.embeddings.pretrained_txt_path is not None
                    else None
                ),
                "pretrained_coverage": round(covered_count / max(vocabulary_size, 1), 6),
                "covered_token_count": covered_count,
            }

        def forward(
            self,
            token_ids: Any,
            position1_ids: Any,
            position2_ids: Any,
            piece_ids: Any,
        ) -> Any:
            word_features = self.word_embedding(token_ids)
            position1_features = self.position1_embedding(position1_ids)
            position2_features = self.position2_embedding(position2_ids)
            stacked_features = torch.cat([word_features, position1_features, position2_features], dim=-1)
            convolution_input = stacked_features.transpose(1, 2)
            pooled_outputs: list[Any] = []
            for convolution in self.convolutions:
                convolution_output = self.activation(convolution(convolution_input))
                pooled_outputs.append(self.pooling(convolution_output, piece_ids))
            pooled_output = torch.cat(pooled_outputs, dim=1)
            return self.dropout(torch.tanh(pooled_output))


    class PCNNMILRelationExtractor(nn.Module):
        """PCNN 句向量 + relation-aware MIL attention 的 bag-level 多标签关系抽取器。"""

        def __init__(
            self,
            config: RelationExtractionConfig,
            vocabulary: Vocabulary,
            index_to_label: list[str],
        ) -> None:
            super().__init__()
            self.config = config
            self.index_to_label = index_to_label
            self.sentence_encoder = PCNNSentenceEncoder(config, vocabulary)
            representation_dimension = self.sentence_encoder.output_dimension
            self.relation_attention_queries = nn.Embedding(len(index_to_label), representation_dimension)
            self.relation_classifier = nn.Parameter(torch.empty(len(index_to_label), representation_dimension))
            self.relation_bias = nn.Parameter(torch.zeros(len(index_to_label)))
            nn.init.xavier_uniform_(self.relation_attention_queries.weight)
            nn.init.xavier_uniform_(self.relation_classifier)

        @property
        def embedding_report(self) -> dict[str, Any]:
            return self.sentence_encoder.embedding_report

        def forward(
            self,
            token_ids: Any,
            position1_ids: Any,
            position2_ids: Any,
            piece_ids: Any,
            bag_scopes: list[tuple[int, int]],
        ) -> dict[str, Any]:
            sentence_representations = self.sentence_encoder(
                token_ids=token_ids,
                position1_ids=position1_ids,
                position2_ids=position2_ids,
                piece_ids=piece_ids,
            )
            bag_logits: list[Any] = []
            attention_weights: list[Any] = []
            attention_queries = self.relation_attention_queries.weight
            for bag_start, bag_end in bag_scopes:
                bag_sentences = sentence_representations[bag_start:bag_end]
                if bag_sentences.size(0) == 0:
                    raise ValueError("检测到空 bag，这会破坏 MIL attention 逻辑。")
                relation_sentence_scores = torch.matmul(attention_queries, bag_sentences.transpose(0, 1))
                relation_attention = torch.softmax(relation_sentence_scores, dim=1)
                bag_relation_representations = torch.matmul(relation_attention, bag_sentences)
                bag_logit = (bag_relation_representations * self.relation_classifier).sum(dim=1) + self.relation_bias
                bag_logits.append(bag_logit)
                attention_weights.append(relation_attention)
            return {
                "logits": torch.stack(bag_logits, dim=0),
                "attention_weights": attention_weights,
            }


else:

    class PCNNMILRelationExtractor:  # pragma: no cover - 仅在缺 torch 环境下兜底报错
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            require_torch()
