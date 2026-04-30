from __future__ import annotations

from importlib import import_module

from kg_core.schemas import MentionRecord, TokenSpan, TokenizedSentence
from kg_core.taxonomy import BIO_LABELS, ENTITY_TYPES, VALID_LABEL_SET

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "build_labeled_record": (".data", "build_labeled_record"),
    "build_tokenized_record": (".data", "build_tokenized_record"),
    "extract_gold_seed": (".data", "extract_gold_seed"),
    "split_weak_and_gold_datasets": (".data", "split_weak_and_gold_datasets"),
    "summarize_label_distribution": (".data", "summarize_label_distribution"),
    "tokenize_sentence_text": (".data", "tokenize_sentence_text"),
    "tokenize_sentences_file": (".data", "tokenize_sentences_file"),
    "decode_mentions_from_labels": (".decode", "decode_mentions_from_labels"),
    "labels_to_spans": (".decode", "labels_to_spans"),
    "legalize_bio_labels": (".decode", "legalize_bio_labels"),
    "DictionaryResources": (".dictionary", "DictionaryResources"),
    "MaxForwardDictionaryMatcher": (".dictionary", "MaxForwardDictionaryMatcher"),
    "load_dictionary_resources": (".dictionary", "load_dictionary_resources"),
    "FeatureConfig": (".features", "FeatureConfig"),
    "build_sentence_features": (".features", "build_sentence_features"),
    "load_feature_config": (".predict", "load_feature_config"),
    "load_model": (".predict", "load_model"),
    "predict_mentions": (".predict", "predict_mentions"),
    "CrfTrainingConfig": (".train", "CrfTrainingConfig"),
    "build_dataset_features": (".train", "build_dataset_features"),
    "evaluate_predictions": (".train", "evaluate_predictions"),
    "require_sklearn_crfsuite": (".train", "require_sklearn_crfsuite"),
    "train_crf_model": (".train", "train_crf_model"),
    "train_from_paths": (".train", "train_from_paths"),
    "WeakLabelApiConfig": (".weak_label", "WeakLabelApiConfig"),
    "auto_check_labels": (".weak_label", "auto_check_labels"),
    "resolve_weak_label_api_config": (".weak_label", "resolve_weak_label_api_config"),
    "weak_label_records": (".weak_label", "weak_label_records"),
}

__all__ = [
    "BIO_LABELS",
    "ENTITY_TYPES",
    "VALID_LABEL_SET",
    "MentionRecord",
    "TokenSpan",
    "TokenizedSentence",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str) -> object:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
