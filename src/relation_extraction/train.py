from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RelationExtractionConfig, load_relation_extraction_config
from .dataset import (
    BagFeatureDataset,
    PreparedRelationDataset,
    collate_relation_batch,
    prepare_training_data,
    write_json,
)
from .evaluate import evaluate_prediction_records
from .model import PCNNMILRelationExtractor, require_torch
from .predict import batch_to_tensors, predict_bags_with_model, resolve_device
from .reporting import build_prediction_report, build_training_report


def _build_class_weight_tensor(
    prepared_dataset: PreparedRelationDataset,
    *,
    device: Any,
) -> Any:
    torch, _ = require_torch()
    return torch.tensor(
        [prepared_dataset.class_weights[label_name] for label_name in prepared_dataset.index_to_label],
        dtype=torch.float32,
        device=device,
    )


def _save_checkpoint(
    checkpoint_path: Path,
    *,
    model: PCNNMILRelationExtractor,
    config: RelationExtractionConfig,
    prepared_dataset: PreparedRelationDataset,
    embedding_report: dict[str, Any],
    best_epoch: int,
    best_dev_metrics: dict[str, Any],
) -> None:
    torch, _ = require_torch()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "label_to_index": prepared_dataset.label_to_index,
            "index_to_label": prepared_dataset.index_to_label,
            "vocabulary": {
                "token_to_index": prepared_dataset.vocabulary.token_to_index,
                "index_to_token": prepared_dataset.vocabulary.index_to_token,
                "lowercase_tokens": prepared_dataset.vocabulary.lowercase_tokens,
            },
            "class_weights": prepared_dataset.class_weights,
            "dataset_report": prepared_dataset.dataset_report,
            "split_bag_ids": {
                "train": [bag.bag_id for bag in prepared_dataset.train_bags],
                "dev": [bag.bag_id for bag in prepared_dataset.dev_bags],
                "test": [bag.bag_id for bag in prepared_dataset.test_bags],
            },
            "embedding_report": embedding_report,
            "best_epoch": best_epoch,
            "best_dev_metrics": best_dev_metrics,
        },
        checkpoint_path,
    )


def _write_training_side_artifacts(
    output_dir: Path,
    *,
    config: RelationExtractionConfig,
    prepared_dataset: PreparedRelationDataset,
    dev_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    write_json(output_dir / "training_config.json", config.to_dict())
    write_json(
        output_dir / "label_vocab.json",
        {
            "label_to_index": prepared_dataset.label_to_index,
            "index_to_label": prepared_dataset.index_to_label,
        },
    )
    write_json(output_dir / "eval_dev.json", dev_metrics)
    write_json(output_dir / "eval_test_gold.json", test_metrics)


def _evaluate_split(
    model: PCNNMILRelationExtractor,
    bags: list[Any],
    *,
    prepared_dataset: PreparedRelationDataset,
    config: RelationExtractionConfig,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_records = predict_bags_with_model(
        model,
        bags,
        vocabulary=prepared_dataset.vocabulary,
        index_to_label=prepared_dataset.index_to_label,
        label_to_index=prepared_dataset.label_to_index,
        config=config,
        device=device,
        batch_size=config.training.batch_size,
    )
    metrics = evaluate_prediction_records(
        prediction_records,
        bags,
        labels=prepared_dataset.index_to_label,
    )
    return prediction_records, metrics


def train_relation_extractor(config: RelationExtractionConfig) -> dict[str, Any]:
    torch, nn = require_torch()
    device = resolve_device(config.training.device)
    prepared_dataset = prepare_training_data(config)
    if config.embeddings.pretrained_txt_path is None or not config.embeddings.pretrained_txt_path.exists():
        raise FileNotFoundError(
            "关系抽取训练要求静态预训练词向量，但当前 embedding_path 不存在。"
            f" 请先准备词向量文件：{config.embeddings.pretrained_txt_path!s}"
        )
    train_loader = torch.utils.data.DataLoader(
        BagFeatureDataset(
            prepared_dataset.train_bags,
            prepared_dataset.vocabulary,
            prepared_dataset.label_to_index,
            max_sentence_length=config.model.max_sentence_length,
            position_clip=config.position_clip,
        ),
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_relation_batch,
    )
    model = PCNNMILRelationExtractor(
        config,
        prepared_dataset.vocabulary,
        prepared_dataset.index_to_label,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    class_weight_tensor = _build_class_weight_tensor(prepared_dataset, device=device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=class_weight_tensor if config.training.use_class_weights else None
    )

    output_dir = config.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "relation_model.pt"
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_dev_metrics: dict[str, Any] = {"macro": {"f1": -1.0}}
    epochs_without_improvement = 0

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        for batch in train_loader:
            tensor_batch = batch_to_tensors(batch, device=device)
            optimizer.zero_grad()
            model_outputs = model(
                token_ids=tensor_batch["token_ids"],
                position1_ids=tensor_batch["position1_ids"],
                position2_ids=tensor_batch["position2_ids"],
                piece_ids=tensor_batch["piece_ids"],
                bag_scopes=tensor_batch["bag_scopes"],
            )
            loss = criterion(model_outputs["logits"], tensor_batch["label_vectors"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.item())
            step_count += 1

        _, dev_metrics = _evaluate_split(
            model,
            prepared_dataset.dev_bags,
            prepared_dataset=prepared_dataset,
            config=config,
            device=device,
        )
        average_loss = epoch_loss / max(step_count, 1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(average_loss, 6),
            "dev_micro_f1": float(dev_metrics["micro"]["f1"]),
            "dev_macro_f1": float(dev_metrics["macro"]["f1"]),
        }
        history.append(epoch_record)
        if float(dev_metrics["macro"]["f1"]) >= float(best_dev_metrics["macro"]["f1"]):
            best_epoch = epoch
            best_dev_metrics = dev_metrics
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                model=model,
                config=config,
                prepared_dataset=prepared_dataset,
                embedding_report=model.embedding_report,
                best_epoch=best_epoch,
                best_dev_metrics=best_dev_metrics,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stop_patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    test_prediction_records, test_metrics = _evaluate_split(
        model,
        prepared_dataset.test_bags,
        prepared_dataset=prepared_dataset,
        config=config,
        device=device,
    )
    write_json(output_dir / "test_predictions.json", {"predictions": test_prediction_records})
    test_prediction_report = build_prediction_report(test_prediction_records)
    training_report = build_training_report(
        config_payload=config.to_dict(),
        dataset_report=prepared_dataset.dataset_report,
        class_weights=prepared_dataset.class_weights,
        embedding_report=model.embedding_report,
        history=history,
        best_epoch=best_epoch,
        best_dev_metrics=best_dev_metrics,
        checkpoint_path=checkpoint_path.as_posix(),
    )
    training_report["test_metrics"] = test_metrics
    training_report["test_prediction_summary"] = test_prediction_report
    write_json(output_dir / "training_report.json", training_report)
    _write_training_side_artifacts(
        output_dir,
        config=config,
        prepared_dataset=prepared_dataset,
        dev_metrics=best_dev_metrics,
        test_metrics=test_metrics,
    )
    return training_report


def train_relation_model(
    *,
    config_path: Path,
    output_dir: Path,
    sentences_path: Path,
    tokenized_sentences_path: Path,
    resolved_mentions_path: Path,
    entities_csv_path: Path,
    aliases_csv_path: Path,
    claims_csv_path: Path,
    ontology_path: Path,
    pair_candidates_path: Path,
    distant_labeled_path: Path,
    target_relations: list[str] | None = None,
    seed: int | None = None,
    dev_ratio: float | None = None,
) -> dict[str, Any]:
    config = load_relation_extraction_config(
        config_path=config_path,
        output_dir=output_dir,
        sentences_path=sentences_path,
        tokenized_sentences_path=tokenized_sentences_path,
        resolved_mentions_path=resolved_mentions_path,
        entities_path=entities_csv_path,
        aliases_path=aliases_csv_path,
        claims_path=claims_csv_path,
        ontology_path=ontology_path,
        pair_candidates_path=pair_candidates_path,
        distant_labeled_path=distant_labeled_path,
        target_relations=target_relations,
        random_seed=seed,
        dev_ratio=dev_ratio,
    )
    return train_relation_extractor(config)
