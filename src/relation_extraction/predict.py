from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.event_mapping import event_type_for_predicate
from kg_core.taxonomy import normalize_entity_type

from .config import (
    DEFAULT_EXTRACTOR_NAME,
    NA_RELATION_LABEL,
    EmbeddingConfig,
    RelationExtractionConfig,
    RelationModelConfig,
    RelationTrainingConfig,
    load_relation_extraction_config,
)
from .dataset import (
    BagFeatureDataset,
    Vocabulary,
    read_json,
    build_relation_bags,
    collate_relation_batch,
    write_json,
    write_jsonl,
)
from .model import PCNNMILRelationExtractor, require_torch


def resolve_device(device_name: str) -> Any:
    torch, _ = require_torch()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def restore_vocabulary(checkpoint_payload: dict[str, Any]) -> Vocabulary:
    vocabulary_payload = checkpoint_payload["vocabulary"]
    return Vocabulary(
        token_to_index={str(key): int(value) for key, value in vocabulary_payload["token_to_index"].items()},
        index_to_token=[str(item) for item in vocabulary_payload["index_to_token"]],
        lowercase_tokens=bool(vocabulary_payload["lowercase_tokens"]),
    )


def load_relation_checkpoint(checkpoint_path: Path, *, device: Any) -> dict[str, Any]:
    torch, _ = require_torch()
    return torch.load(checkpoint_path, map_location=device)


def _restore_config_from_checkpoint(
    checkpoint_payload: dict[str, Any],
    runtime_config: RelationExtractionConfig,
) -> RelationExtractionConfig:
    checkpoint_config = checkpoint_payload.get("config")
    if not isinstance(checkpoint_config, dict):
        return runtime_config
    embedding_payload = dict(checkpoint_config.get("embeddings", {}))
    model_payload = dict(checkpoint_config.get("model", {}))
    training_payload = dict(checkpoint_config.get("training", {}))
    restored_config = RelationExtractionConfig(
        data=runtime_config.data,
        embeddings=EmbeddingConfig(
            pretrained_txt_path=runtime_config.embeddings.pretrained_txt_path,
            embedding_dim=int(embedding_payload.get("embedding_dim", runtime_config.embeddings.embedding_dim)),
            position_embedding_dim=int(
                embedding_payload.get("position_embedding_dim", runtime_config.embeddings.position_embedding_dim)
            ),
            min_token_frequency=int(embedding_payload.get("min_token_frequency", runtime_config.embeddings.min_token_frequency)),
            lowercase_tokens=bool(embedding_payload.get("lowercase_tokens", runtime_config.embeddings.lowercase_tokens)),
            initializer_range=float(embedding_payload.get("initializer_range", runtime_config.embeddings.initializer_range)),
        ),
        model=RelationModelConfig(
            max_sentence_length=int(model_payload.get("max_sentence_length", runtime_config.model.max_sentence_length)),
            max_sentences_per_bag=int(model_payload.get("max_sentences_per_bag", runtime_config.model.max_sentences_per_bag)),
            convolution_channels=int(model_payload.get("convolution_channels", runtime_config.model.convolution_channels)),
            convolution_kernel_sizes=[
                int(item)
                for item in model_payload.get("convolution_kernel_sizes", runtime_config.model.convolution_kernel_sizes)
            ],
            dropout=float(model_payload.get("dropout", runtime_config.model.dropout)),
            relation_threshold=runtime_config.model.relation_threshold,
            threshold_by_relation=runtime_config.model.threshold_by_relation,
            top_k_support_sentences=runtime_config.model.top_k_support_sentences,
            prediction_mask_mode=runtime_config.model.prediction_mask_mode,
        ),
        training=RelationTrainingConfig(
            random_seed=runtime_config.training.random_seed,
            batch_size=runtime_config.training.batch_size,
            num_epochs=int(training_payload.get("num_epochs", runtime_config.training.num_epochs)),
            learning_rate=float(training_payload.get("learning_rate", runtime_config.training.learning_rate)),
            weight_decay=float(training_payload.get("weight_decay", runtime_config.training.weight_decay)),
            gradient_clip_norm=float(training_payload.get("gradient_clip_norm", runtime_config.training.gradient_clip_norm)),
            train_ratio=float(training_payload.get("train_ratio", runtime_config.training.train_ratio)),
            dev_ratio=float(training_payload.get("dev_ratio", runtime_config.training.dev_ratio)),
            na_downsample_ratio=float(training_payload.get("na_downsample_ratio", runtime_config.training.na_downsample_ratio)),
            log_every_steps=int(training_payload.get("log_every_steps", runtime_config.training.log_every_steps)),
            device=runtime_config.training.device,
            early_stop_patience=int(training_payload.get("early_stop_patience", runtime_config.training.early_stop_patience)),
            use_class_weights=bool(training_payload.get("use_class_weights", runtime_config.training.use_class_weights)),
        ),
        target_relations=list(runtime_config.target_relations),
    )
    return restored_config


def load_relation_model(
    checkpoint_path: Path,
    *,
    config: RelationExtractionConfig,
    device: Any,
) -> tuple[PCNNMILRelationExtractor, dict[str, Any], Vocabulary, list[str], dict[str, int], RelationExtractionConfig]:
    checkpoint_payload = load_relation_checkpoint(checkpoint_path, device=device)
    restored_config = _restore_config_from_checkpoint(checkpoint_payload, config)
    vocabulary = restore_vocabulary(checkpoint_payload)
    index_to_label = [str(item) for item in checkpoint_payload["index_to_label"]]
    label_to_index = {str(key): int(value) for key, value in checkpoint_payload["label_to_index"].items()}
    model = PCNNMILRelationExtractor(restored_config, vocabulary, index_to_label)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint_payload, vocabulary, index_to_label, label_to_index, restored_config


def batch_to_tensors(batch: dict[str, Any], *, device: Any) -> dict[str, Any]:
    torch, _ = require_torch()
    return {
        "token_ids": torch.tensor(batch["token_ids"], dtype=torch.long, device=device),
        "position1_ids": torch.tensor(batch["position1_ids"], dtype=torch.long, device=device),
        "position2_ids": torch.tensor(batch["position2_ids"], dtype=torch.long, device=device),
        "piece_ids": torch.tensor(batch["piece_ids"], dtype=torch.long, device=device),
        "label_vectors": torch.tensor(batch["label_vectors"], dtype=torch.float32, device=device),
        "bag_scopes": list(batch["bag_scopes"]),
    }


def _resolve_threshold(config: RelationExtractionConfig, relation_name: str) -> float:
    return float(config.model.threshold_by_relation.get(relation_name, config.model.relation_threshold))


def _allowed_predicates_from_ontology(
    *,
    config: RelationExtractionConfig,
    subject_type: str,
    object_type: str,
    index_to_label: list[str],
) -> set[str]:
    ontology = read_json(config.data.ontology_path)
    normalized_subject = normalize_entity_type(subject_type)
    normalized_object = normalize_entity_type(object_type)
    label_set = {label for label in index_to_label if label != NA_RELATION_LABEL}
    allowed: set[str] = set()
    for relation in ontology.get("relations", []):
        relation_name = str(relation.get("name", "")).strip().upper()
        if relation_name not in label_set:
            continue
        raw_domains = relation.get("domain", "Entity")
        raw_ranges = relation.get("range", "Entity")
        domains = raw_domains if isinstance(raw_domains, list) else [raw_domains]
        ranges = raw_ranges if isinstance(raw_ranges, list) else [raw_ranges]
        domain_set = {normalize_entity_type(item) for item in domains}
        range_set = {normalize_entity_type(item) for item in ranges}
        if normalized_subject in domain_set and normalized_object in range_set:
            allowed.add(relation_name)
    return allowed


def _candidate_predicates_from_evidence(evidence_items: list[dict[str, Any]]) -> set[str]:
    return {
        str(predicate).strip().upper()
        for evidence in evidence_items
        for predicate in evidence.get("candidate_predicates", [])
        if str(predicate).strip()
    }


def _resolve_output_predicates(
    *,
    config: RelationExtractionConfig,
    metadata: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    index_to_label: list[str],
) -> tuple[set[str], set[str], set[str], str]:
    allowed_predicates = {
        str(item).strip().upper()
        for item in metadata.get("allowed_predicates", [])
        if str(item).strip()
    }
    if not allowed_predicates:
        allowed_predicates = _allowed_predicates_from_ontology(
            config=config,
            subject_type=str(metadata.get("subject_type", "")),
            object_type=str(metadata.get("object_type", "")),
            index_to_label=index_to_label,
        )
    if not allowed_predicates:
        raise ValueError(
            "预测阶段无法确定 allowed_predicates，不能开放全部关系输出。"
            f" bag_id={metadata.get('bag_id')} subject_type={metadata.get('subject_type')} object_type={metadata.get('object_type')}"
        )
    candidate_predicates = _candidate_predicates_from_evidence(evidence_items)
    mask_mode = config.model.prediction_mask_mode
    if mask_mode == "ontology":
        output_predicates = set(allowed_predicates)
    elif mask_mode in {"candidate", "ontology-and-candidate"}:
        output_predicates = allowed_predicates & candidate_predicates
    else:
        raise ValueError(f"未知 prediction_mask_mode: {mask_mode}")
    return output_predicates, allowed_predicates, candidate_predicates, mask_mode


def predict_bags_with_model(
    model: PCNNMILRelationExtractor,
    bags: list[Any],
    *,
    vocabulary: Vocabulary,
    index_to_label: list[str],
    label_to_index: dict[str, int],
    config: RelationExtractionConfig,
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    torch, _ = require_torch()
    data_loader = torch.utils.data.DataLoader(
        BagFeatureDataset(
            bags,
            vocabulary,
            label_to_index,
            max_sentence_length=config.model.max_sentence_length,
            position_clip=config.position_clip,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_relation_batch,
    )
    prediction_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in data_loader:
            tensor_batch = batch_to_tensors(batch, device=device)
            model_outputs = model(
                token_ids=tensor_batch["token_ids"],
                position1_ids=tensor_batch["position1_ids"],
                position2_ids=tensor_batch["position2_ids"],
                piece_ids=tensor_batch["piece_ids"],
                bag_scopes=tensor_batch["bag_scopes"],
            )
            probabilities = torch.sigmoid(model_outputs["logits"]).detach().cpu().tolist()
            attention_weights = [item.detach().cpu().tolist() for item in model_outputs["attention_weights"]]
            for batch_index, metadata in enumerate(batch["bag_metadata"]):
                bag_start, bag_end = batch["bag_scopes"][batch_index]
                bag_sentence_ids = batch["sentence_ids"][bag_start:bag_end]
                bag_sentence_texts = batch["sentence_texts"][bag_start:bag_end]
                bag_evidence_metadata = batch["evidence_metadata"][bag_start:bag_end]
                bag_probabilities = probabilities[batch_index]
                bag_attention = attention_weights[batch_index]
                probability_map = {
                    relation_name: round(float(bag_probabilities[label_index]), 6)
                    for label_index, relation_name in enumerate(index_to_label)
                }
                predicted_relations: list[dict[str, Any]] = []
                output_predicates, allowed_predicates, candidate_predicates, mask_mode = _resolve_output_predicates(
                    config=config,
                    metadata=metadata,
                    evidence_items=bag_evidence_metadata,
                    index_to_label=index_to_label,
                )
                masked_relations: list[str] = []
                for label_index, relation_name in enumerate(index_to_label):
                    if relation_name == NA_RELATION_LABEL:
                        continue
                    if relation_name not in output_predicates:
                        masked_relations.append(relation_name)
                        continue
                    threshold = _resolve_threshold(config, relation_name)
                    probability = probability_map[relation_name]
                    if probability < threshold:
                        continue
                    relation_attention = bag_attention[label_index]
                    ranked_sentence_indices = sorted(
                        range(len(relation_attention)),
                        key=lambda sentence_index: relation_attention[sentence_index],
                        reverse=True,
                    )[: config.model.top_k_support_sentences]
                    supporting_sentence_ids = [bag_sentence_ids[index] for index in ranked_sentence_indices]
                    supporting_texts = [bag_sentence_texts[index] for index in ranked_sentence_indices]
                    supporting_evidence = [bag_evidence_metadata[index] for index in ranked_sentence_indices]
                    predicted_relations.append(
                        {
                            "predicate": relation_name,
                            "probability": probability,
                            "threshold_used": round(threshold, 6),
                            "supporting_sentence_ids": supporting_sentence_ids,
                            "supporting_texts": supporting_texts,
                            "supporting_evidence": supporting_evidence,
                        }
                    )
                prediction_records.append(
                    {
                        "bag_id": metadata["bag_id"],
                        "doc_id": metadata["doc_id"],
                        "subject_id": metadata["subject_id"],
                        "object_id": metadata["object_id"],
                        "subject_type": metadata["subject_type"],
                        "object_type": metadata["object_type"],
                        "mask_mode": mask_mode,
                        "allowed_predicates": sorted(allowed_predicates),
                        "candidate_predicates": sorted(candidate_predicates),
                        "masked_relations": sorted(masked_relations),
                        "sentence_ids": bag_sentence_ids,
                        "probabilities": probability_map,
                        "predicted_relations": predicted_relations,
                        "extractor": DEFAULT_EXTRACTOR_NAME,
                    }
                )
    return prediction_records


def _predict_relations_from_checkpoint(
    checkpoint_path: Path,
    *,
    config: RelationExtractionConfig,
    output_path: Path | None = None,
    split_name: str | None = None,
) -> list[dict[str, Any]]:
    device = resolve_device(config.training.device)
    model, checkpoint_payload, vocabulary, index_to_label, label_to_index, inference_config = load_relation_model(
        checkpoint_path,
        config=config,
        device=device,
    )
    inference_bags, _ = build_relation_bags(inference_config, include_gold_labels=False)
    if split_name is not None:
        split_bag_ids_payload = checkpoint_payload.get("split_bag_ids", {})
        requested_bag_ids = {str(item) for item in split_bag_ids_payload.get(split_name, [])}
        inference_bags = [bag for bag in inference_bags if bag.bag_id in requested_bag_ids]
    prediction_records = predict_bags_with_model(
        model,
        inference_bags,
        vocabulary=vocabulary,
        index_to_label=index_to_label,
        label_to_index=label_to_index,
        config=inference_config,
        device=device,
        batch_size=inference_config.training.batch_size,
    )
    if output_path is not None:
        write_json(output_path, {"predictions": prediction_records})
    return prediction_records


def _support_matches_predicate(predicate: str, evidence: dict[str, Any]) -> bool:
    relation_name = str(predicate).strip().upper()
    return (
        relation_name in {str(item).strip().upper() for item in evidence.get("candidate_predicates", [])}
        or relation_name in {str(item).strip().upper() for item in dict(evidence.get("local_trigger_hits", {}))}
        or relation_name in {str(item).strip().upper() for item in dict(evidence.get("exact_claim_matches", {}))}
    )


def _build_extracted_claim_records(prediction_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    extracted_claims: list[dict[str, Any]] = []
    running_index = 0
    for record in prediction_records:
        for predicted_relation in record.get("predicted_relations", []):
            supporting_sentence_ids = list(predicted_relation.get("supporting_sentence_ids", []))
            supporting_texts = list(predicted_relation.get("supporting_texts", []))
            supporting_evidence = list(predicted_relation.get("supporting_evidence", []))
            if not supporting_sentence_ids:
                continue
            primary_evidence = supporting_evidence[0] if supporting_evidence else {}
            predicate = str(predicted_relation.get("predicate") or "").strip().upper()
            support_matches_predicate = _support_matches_predicate(predicate, primary_evidence)
            running_index += 1
            extracted_claims.append(
                {
                    "claim_candidate_id": f"relclaim_{running_index:06d}",
                    "bag_id": record.get("bag_id"),
                    "doc_id": record.get("doc_id"),
                    "source_id": primary_evidence.get("source_id", ""),
                    "sentence_index_in_doc": primary_evidence.get("sentence_index_in_doc"),
                    "subject_id": record.get("subject_id"),
                    "object_id": record.get("object_id"),
                    "subject_mention_id": primary_evidence.get("subject_mention_id", ""),
                    "object_mention_id": primary_evidence.get("object_mention_id", ""),
                    "subject_text": primary_evidence.get("subject_text", ""),
                    "object_text": primary_evidence.get("object_text", ""),
                    "subject_span": primary_evidence.get("original_subject_span", primary_evidence.get("subject_span", [])),
                    "object_span": primary_evidence.get("original_object_span", primary_evidence.get("object_span", [])),
                    "predicate": predicate,
                    "probability": predicted_relation.get("probability"),
                    "confidence": predicted_relation.get("probability"),
                    "threshold_used": predicted_relation.get("threshold_used"),
                    "supporting_candidate_ids": [
                        evidence.get("candidate_id", "") for evidence in supporting_evidence if evidence.get("candidate_id")
                    ],
                    "supporting_sentence_ids": supporting_sentence_ids,
                    "evidence_sentence_id": supporting_sentence_ids[0],
                    "evidence_text": supporting_texts[0] if supporting_texts else "",
                    "evidence_candidate_id": primary_evidence.get("candidate_id", ""),
                    "supporting_texts": supporting_texts,
                    "supporting_evidence": supporting_evidence,
                    "pair_source": primary_evidence.get("pair_source", ""),
                    "is_from_bridge": primary_evidence.get("pair_source") == "bridge_augmented",
                    "exact_claim_match": primary_evidence.get("exact_claim_match", False),
                    "matched_claim_ids": primary_evidence.get("matched_claim_ids", []),
                    "bridge_predicates": primary_evidence.get("bridge_predicates", []),
                    "bridge_details": primary_evidence.get("bridge_details", {}),
                    "supervision_tier": primary_evidence.get("supervision_tier", ""),
                    "support_matches_predicate": support_matches_predicate,
                    "support_evidence_mismatch": not support_matches_predicate,
                    "extractor": DEFAULT_EXTRACTOR_NAME,
                    "event_type_hint": event_type_for_predicate(predicate) or "",
                }
            )
    return extracted_claims


def predict_relations(
    checkpoint_path: Path | None = None,
    *,
    config: RelationExtractionConfig | None = None,
    output_path: Path | None = None,
    split_name: str | None = None,
    model_dir: Path | None = None,
    config_path: Path | None = None,
    sentences_path: Path | None = None,
    tokenized_sentences_path: Path | None = None,
    resolved_mentions_path: Path | None = None,
    entities_csv_path: Path | None = None,
    aliases_csv_path: Path | None = None,
    claims_csv_path: Path | None = None,
    ontology_path: Path | None = None,
    pair_candidates_path: Path | None = None,
    distant_labeled_path: Path | None = None,
    target_relations: list[str] | None = None,
    threshold: float | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    if config is not None and checkpoint_path is not None:
        return _predict_relations_from_checkpoint(
            checkpoint_path,
            config=config,
            output_path=output_path,
            split_name=split_name,
        )
    if model_dir is None or config_path is None:
        raise ValueError("关系预测缺少 model_dir/config_path，无法加载训练好的关系模型。")
    loaded_config = load_relation_extraction_config(
        config_path=config_path,
        output_dir=model_dir,
        sentences_path=sentences_path or Path(""),
        tokenized_sentences_path=tokenized_sentences_path or Path(""),
        resolved_mentions_path=resolved_mentions_path or Path(""),
        entities_path=entities_csv_path or Path(""),
        aliases_path=aliases_csv_path or Path(""),
        claims_path=claims_csv_path or Path(""),
        ontology_path=ontology_path or Path(""),
        pair_candidates_path=pair_candidates_path or Path(""),
        distant_labeled_path=distant_labeled_path or Path(""),
        target_relations=target_relations,
        relation_threshold=threshold,
    )
    resolved_checkpoint_path = model_dir / "relation_model.pt"
    prediction_records = _predict_relations_from_checkpoint(
        resolved_checkpoint_path,
        config=loaded_config,
        output_path=None,
        split_name=split_name,
    )
    extracted_claims = _build_extracted_claim_records(prediction_records)
    if output_path is not None:
        write_jsonl(output_path, extracted_claims)
        write_json(
            output_path.with_suffix(".report.json"),
            {"predictions": prediction_records},
        )
    return {
        "prediction_count": len(prediction_records),
        "extracted_claim_count": len(extracted_claims),
        "output_path": output_path.as_posix() if output_path is not None else None,
        "checkpoint_path": resolved_checkpoint_path.as_posix(),
    }
