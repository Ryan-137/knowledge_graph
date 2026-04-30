from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


SCRIPT_FILE = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_FILE.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for bootstrap_path in (PROJECT_ROOT, SRC_ROOT):
    bootstrap_text = str(bootstrap_path)
    if bootstrap_text not in sys.path:
        sys.path.insert(0, bootstrap_text)

from coreference import resolve_coreferences_from_paths
from entity_linking import (
    LinkingConfig,
    evaluate_linking_from_paths,
    link_mentions_from_paths,
    mine_linking_gaps_from_path,
)
from kg_core import ProjectPaths
from kg_core.io import write_json
from kg_core.paths import ensure_src_on_path
from mention_crf import (
    CrfTrainingConfig,
    FeatureConfig,
    MaxForwardDictionaryMatcher,
    build_dataset_features,
    evaluate_predictions,
    extract_gold_seed,
    load_dictionary_resources,
    load_model,
    predict_mentions,
    require_sklearn_crfsuite,
    split_weak_and_gold_datasets,
    summarize_label_distribution,
    tokenize_sentences_file,
    train_from_paths,
    weak_label_records,
)
from mention_crf.data import sample_weak_label_candidates, write_jsonl
from structured_seed import FetchConfig, StructuredFetchPipeline
from unstructured_preprocess import run_document_preprocess, run_sentence_preprocess


PATHS = ProjectPaths.from_script(__file__)
ensure_src_on_path(PATHS)

DEFAULT_STRUCTURED_CONFIG = PATHS.configs_root / "structured_fetch_config.json"
DEFAULT_SOURCE_CONFIG = PATHS.configs_root / "unstructured_sources.yaml"
DEFAULT_ONTOLOGY_PATH = PATHS.project_root / "knowledge" / "ontology.json"
DEFAULT_RELATION_TRAINING_CONFIG = PATHS.configs_root / "relation_training_config.json"
DEFAULT_RELATION_PATTERNS = PATHS.configs_root / "relation_patterns.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="图灵知识图谱统一工程化流水线")
    subparsers = parser.add_subparsers(dest="module", required=True)

    structured_parser = subparsers.add_parser("structured", help="结构化种子库构建")
    structured_parser.add_argument("--config", default=str(DEFAULT_STRUCTURED_CONFIG))
    structured_subparsers = structured_parser.add_subparsers(dest="command", required=True)
    structured_subparsers.add_parser("init-db", help="初始化 SQLite 表结构")
    structured_subparsers.add_parser("build", help="执行结构化抓取与种子库构建")
    structured_subparsers.add_parser("validate", help="执行结构化库一致性校验")
    structured_subparsers.add_parser("export-csv", help="导出结构化 CSV")

    unstructured_parser = subparsers.add_parser("unstructured", help="非结构化数据预处理")
    unstructured_subparsers = unstructured_parser.add_subparsers(dest="command", required=True)
    preprocess_parser = unstructured_subparsers.add_parser("preprocess", help="顺序执行文档级与句子级预处理")
    preprocess_parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG))
    preprocess_parser.add_argument("--documents-output", default=str(PATHS.documents_jsonl))
    preprocess_parser.add_argument("--documents-report", default=str(PATHS.unstructured_dir / "documents.report.json"))
    preprocess_parser.add_argument("--sentences-output", default=str(PATHS.sentences_jsonl))
    preprocess_parser.add_argument("--sentences-report", default=str(PATHS.unstructured_dir / "sentences.report.json"))
    preprocess_parser.add_argument("--strict", action="store_true")

    mentions_parser = subparsers.add_parser("mentions", help="CRF mention 识别链路")
    mentions_subparsers = mentions_parser.add_subparsers(dest="command", required=True)

    prepare_parser = mentions_subparsers.add_parser("prepare", help="生成 tokenized 句子和人工金标模板")
    prepare_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    prepare_parser.add_argument("--tokenized-output", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    prepare_parser.add_argument("--gold-template-output", default=str(PATHS.mentions_dir / "test_gold_template.jsonl"))
    prepare_parser.add_argument("--sample-size", type=int, default=120)
    prepare_parser.add_argument("--seed", type=int, default=42)

    weak_label_parser = mentions_subparsers.add_parser("weak-label", help="调用弱监督接口生成 CRF 训练语料")
    weak_label_parser.add_argument("--tokenized", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    weak_label_parser.add_argument("--output", default=str(PATHS.mentions_dir / "weak_labeled.jsonl"))
    weak_label_parser.add_argument("--reject-report", default=str(PATHS.mentions_dir / "weak_label_rejects.json"))
    weak_label_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    weak_label_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    weak_label_parser.add_argument("--base-sample-per-doc", type=int, default=1)
    weak_label_parser.add_argument("--sample-budget", type=int, default=240)
    weak_label_parser.add_argument("--targeted-topup-per-keyword", type=int, default=2)
    weak_label_parser.add_argument("--seed", type=int, default=42)
    weak_label_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    weak_label_parser.add_argument("--timeout-seconds", type=int, default=60)
    weak_label_parser.add_argument("--progress-every", type=int, default=1)
    weak_label_parser.add_argument("--api-key", default=None)
    weak_label_parser.add_argument("--base-url", default=None)
    weak_label_parser.add_argument("--model-name", default=None)

    train_parser = mentions_subparsers.add_parser("train", help="切分数据集并训练 CRF 模型")
    train_parser.add_argument("--weak-labeled", default=str(PATHS.mentions_dir / "weak_labeled.jsonl"))
    train_parser.add_argument("--train-output", default=str(PATHS.mentions_dir / "train.jsonl"))
    train_parser.add_argument("--dev-output", default=str(PATHS.mentions_dir / "dev.jsonl"))
    train_parser.add_argument("--test-gold", default=str(PATHS.mentions_dir / "test_gold.jsonl"))
    train_parser.add_argument("--summary-output", default=str(PATHS.mentions_dir / "dataset_summary.json"))
    train_parser.add_argument("--model-dir", default=str(PATHS.mentions_dir / "model"))
    train_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    train_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    train_parser.add_argument("--dev-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--disable-pos", action="store_true")
    train_parser.add_argument("--disable-dict", action="store_true")
    train_parser.add_argument("--disable-time-hint", action="store_true")
    train_parser.add_argument("--c1", type=float, default=0.1)
    train_parser.add_argument("--c2", type=float, default=0.1)
    train_parser.add_argument("--max-iterations", type=int, default=200)

    predict_parser = mentions_subparsers.add_parser("predict", help="使用已训练 CRF 模型抽取 mention")
    predict_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    predict_parser.add_argument("--output", default=str(PATHS.mentions_jsonl))
    predict_parser.add_argument("--model", default=str(PATHS.mentions_dir / "model" / "crf_model.pkl"))
    predict_parser.add_argument("--feature-config", default=str(PATHS.mentions_dir / "model" / "feature_config.json"))
    predict_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    predict_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))

    linking_parser = subparsers.add_parser("linking", help="候选排序实体链接")
    linking_subparsers = linking_parser.add_subparsers(dest="command", required=True)
    link_parser = linking_subparsers.add_parser("link", help="执行 mention 到实体的链接")
    link_parser.add_argument("--mentions", default=str(PATHS.mentions_jsonl))
    link_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    link_parser.add_argument("--documents", default=str(PATHS.documents_jsonl))
    link_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    link_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    link_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    link_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    link_parser.add_argument("--output", default=str(PATHS.linked_mentions_jsonl))
    link_parser.add_argument("--report", default=str(PATHS.linking_dir / "linking_report.json"))
    link_parser.add_argument("--top-k", type=int, default=5)
    link_parser.add_argument("--link-threshold", type=float, default=0.85)
    link_parser.add_argument("--multi-token-link-threshold", type=float, default=0.82)
    link_parser.add_argument("--single-token-link-threshold", type=float, default=0.88)
    link_parser.add_argument("--review-threshold", type=float, default=0.65)
    link_parser.add_argument("--margin-threshold", type=float, default=0.05)
    link_parser.add_argument("--fuzzy-threshold", type=float, default=0.78)
    link_parser.add_argument("--tfidf-recall-limit", type=int, default=8)
    link_parser.add_argument("--anchor-threshold", type=float, default=0.88)
    link_parser.add_argument("--limit", type=int, default=None)
    gap_parser = linking_subparsers.add_parser("mine-gaps", help="统计高频断链 mention，生成补库清单")
    gap_parser.add_argument("--linked-mentions", default=str(PATHS.linked_mentions_jsonl))
    gap_parser.add_argument("--output", default=str(PATHS.linking_dir / "linking_gap_report.json"))
    gap_parser.add_argument("--top-n", type=int, default=50)
    evaluate_parser = linking_subparsers.add_parser("evaluate", help="基于 dev gold 评测 linking 效果")
    evaluate_parser.add_argument("--predictions", default=str(PATHS.linked_mentions_jsonl))
    evaluate_parser.add_argument("--gold", default=str(PATHS.linking_dir / "linking_dev_gold.jsonl"))
    evaluate_parser.add_argument("--output", default=str(PATHS.linking_dir / "linking_eval.json"))

    coreference_parser = subparsers.add_parser("coreference", help="规则共指与 anchor 传播")
    coreference_subparsers = coreference_parser.add_subparsers(dest="command", required=True)
    resolve_parser = coreference_subparsers.add_parser("resolve", help="将 skipped mention 解析为 resolved mentions")
    resolve_parser.add_argument("--linked-mentions", default=str(PATHS.linked_mentions_jsonl))
    resolve_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    resolve_parser.add_argument("--output", default=str(PATHS.resolved_mentions_jsonl))
    resolve_parser.add_argument("--report", default=str(PATHS.coreference_report_json))
    resolve_parser.add_argument("--unresolved-output", default=str(PATHS.coreference_dir / "coreference_unresolved.jsonl"))
    resolve_parser.add_argument("--max-sentence-distance", type=int, default=3)

    relations_parser = subparsers.add_parser("relations", help="关系抽取链路")
    relations_subparsers = relations_parser.add_subparsers(dest="command", required=True)

    relations_prepare_parser = relations_subparsers.add_parser("prepare", help="基于 resolved mention 生成实体对候选")
    relations_prepare_parser.add_argument("--resolved-mentions", default=str(PATHS.resolved_mentions_jsonl))
    relations_prepare_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    relations_prepare_parser.add_argument("--tokenized-sentences", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    relations_prepare_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    relations_prepare_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    relations_prepare_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    relations_prepare_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    relations_prepare_parser.add_argument("--output", default=str(PATHS.pair_candidates_jsonl))

    relations_weak_label_parser = relations_subparsers.add_parser("weak-label", help="基于结构化 claims 生成远程监督关系样本")
    relations_weak_label_parser.add_argument("--pair-candidates", default=str(PATHS.pair_candidates_jsonl))
    relations_weak_label_parser.add_argument("--resolved-mentions", default=str(PATHS.resolved_mentions_jsonl))
    relations_weak_label_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    relations_weak_label_parser.add_argument("--tokenized-sentences", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    relations_weak_label_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    relations_weak_label_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    relations_weak_label_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    relations_weak_label_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    relations_weak_label_parser.add_argument("--output", default=str(PATHS.distant_labeled_jsonl))

    relations_train_parser = relations_subparsers.add_parser("train", help="训练关系分类模型")
    relations_train_parser.add_argument("--config", default=str(DEFAULT_RELATION_TRAINING_CONFIG))
    relations_train_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    relations_train_parser.add_argument("--tokenized-sentences", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    relations_train_parser.add_argument("--resolved-mentions", default=str(PATHS.resolved_mentions_jsonl))
    relations_train_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    relations_train_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    relations_train_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    relations_train_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    relations_train_parser.add_argument("--pair-candidates", default=str(PATHS.pair_candidates_jsonl))
    relations_train_parser.add_argument("--distant-labeled", default=str(PATHS.distant_labeled_jsonl))
    relations_train_parser.add_argument("--model-dir", default=str(PATHS.relation_model_dir))
    relations_train_parser.add_argument("--dev-ratio", type=float, default=0.2)
    relations_train_parser.add_argument("--seed", type=int, default=42)

    relations_predict_parser = relations_subparsers.add_parser("predict", help="对实体对候选执行关系预测")
    relations_predict_parser.add_argument("--config", default=str(DEFAULT_RELATION_TRAINING_CONFIG))
    relations_predict_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    relations_predict_parser.add_argument("--tokenized-sentences", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    relations_predict_parser.add_argument("--resolved-mentions", default=str(PATHS.resolved_mentions_jsonl))
    relations_predict_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    relations_predict_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    relations_predict_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    relations_predict_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    relations_predict_parser.add_argument("--pair-candidates", default=str(PATHS.pair_candidates_jsonl))
    relations_predict_parser.add_argument("--distant-labeled", default=str(PATHS.distant_labeled_jsonl))
    relations_predict_parser.add_argument("--model-dir", default=str(PATHS.relation_model_dir))
    relations_predict_parser.add_argument("--output", default=str(PATHS.relations_dir / "extracted_claims.jsonl"))
    relations_predict_parser.add_argument("--threshold", type=float, default=0.5)

    relations_evaluate_parser = relations_subparsers.add_parser("evaluate", help="评估关系预测结果")
    relations_evaluate_parser.add_argument("--config", default=str(DEFAULT_RELATION_TRAINING_CONFIG))
    relations_evaluate_parser.add_argument("--sentences", default=str(PATHS.sentences_jsonl))
    relations_evaluate_parser.add_argument("--tokenized-sentences", default=str(PATHS.mentions_dir / "tokenized_sentences.jsonl"))
    relations_evaluate_parser.add_argument("--resolved-mentions", default=str(PATHS.resolved_mentions_jsonl))
    relations_evaluate_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    relations_evaluate_parser.add_argument("--aliases-csv", default=str(PATHS.structured_csv_dir / "aliases.csv"))
    relations_evaluate_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    relations_evaluate_parser.add_argument("--ontology", default=str(DEFAULT_ONTOLOGY_PATH))
    relations_evaluate_parser.add_argument("--pair-candidates", default=str(PATHS.pair_candidates_jsonl))
    relations_evaluate_parser.add_argument("--distant-labeled", default=str(PATHS.distant_labeled_jsonl))
    relations_evaluate_parser.add_argument("--model-dir", default=str(PATHS.relation_model_dir))
    relations_evaluate_parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    relations_evaluate_parser.add_argument("--report", default=str(PATHS.relations_dir / "evaluation.json"))

    facts_parser = subparsers.add_parser("facts", help="V1 事实抽取链路")
    facts_subparsers = facts_parser.add_subparsers(dest="command", required=True)

    facts_common_defaults = {
        "resolved_mentions": str(PATHS.resolved_mentions_jsonl),
        "pair_candidates": str(PATHS.pair_candidates_jsonl),
        "sentences": str(PATHS.sentences_jsonl),
        "claims_csv": str(PATHS.structured_csv_dir / "claims.csv"),
        "ontology": str(DEFAULT_ONTOLOGY_PATH),
        "patterns": str(DEFAULT_RELATION_PATTERNS),
        "fact_candidates": str(PATHS.fact_candidates_jsonl),
        "fact_verified": str(PATHS.fact_verified_jsonl),
        "facts_final": str(PATHS.facts_final_jsonl),
        "fact_conflicts": str(PATHS.fact_conflicts_jsonl),
    }

    def add_fact_common_args(parser_obj: argparse.ArgumentParser) -> None:
        parser_obj.add_argument("--resolved-mentions", default=facts_common_defaults["resolved_mentions"])
        parser_obj.add_argument("--pair-candidates", default=facts_common_defaults["pair_candidates"])
        parser_obj.add_argument("--sentences", default=facts_common_defaults["sentences"])
        parser_obj.add_argument("--claims-csv", default=facts_common_defaults["claims_csv"])
        parser_obj.add_argument("--ontology", default=facts_common_defaults["ontology"])
        parser_obj.add_argument("--patterns", default=facts_common_defaults["patterns"])
        parser_obj.add_argument("--fact-candidates", default=facts_common_defaults["fact_candidates"])
        parser_obj.add_argument("--fact-verified", default=facts_common_defaults["fact_verified"])
        parser_obj.add_argument("--facts-final", default=facts_common_defaults["facts_final"])
        parser_obj.add_argument("--fact-conflicts", default=facts_common_defaults["fact_conflicts"])

    facts_generate_parser = facts_subparsers.add_parser("generate-candidates", help="从 pair candidates 生成事实候选")
    add_fact_common_args(facts_generate_parser)

    facts_score_parser = facts_subparsers.add_parser("score", help="加入结构化 claims 远程监督信号")
    add_fact_common_args(facts_score_parser)

    facts_verify_parser = facts_subparsers.add_parser("verify-llm", help="候选事实 LLM/离线校验")
    add_fact_common_args(facts_verify_parser)
    facts_verify_parser.add_argument("--api-key", default=None)
    facts_verify_parser.add_argument("--base-url", default=None)
    facts_verify_parser.add_argument("--model-name", default=None)
    facts_verify_parser.add_argument("--timeout-seconds", type=int, default=60)

    facts_aggregate_parser = facts_subparsers.add_parser("aggregate", help="聚合证据并输出最终事实")
    add_fact_common_args(facts_aggregate_parser)

    facts_run_parser = facts_subparsers.add_parser("run", help="顺序执行 V1 事实抽取链路")
    add_fact_common_args(facts_run_parser)
    facts_run_parser.add_argument("--api-key", default=None)
    facts_run_parser.add_argument("--base-url", default=None)
    facts_run_parser.add_argument("--model-name", default=None)
    facts_run_parser.add_argument("--timeout-seconds", type=int, default=60)

    visualization_parser = subparsers.add_parser("visualization", help="最终图谱可视化导出层")
    visualization_subparsers = visualization_parser.add_subparsers(dest="command", required=True)
    visualization_export_parser = visualization_subparsers.add_parser("export", help="导出图谱 JSON、Gephi CSV、Neo4j CSV 和离线 HTML")
    visualization_export_parser.add_argument("--entities-csv", default=str(PATHS.structured_csv_dir / "entities.csv"))
    visualization_export_parser.add_argument("--claims-csv", default=str(PATHS.structured_csv_dir / "claims.csv"))
    visualization_export_parser.add_argument("--events-csv", default=str(PATHS.structured_csv_dir / "event_candidates.csv"))
    visualization_export_parser.add_argument("--text-facts", default=str(PATHS.facts_final_jsonl))
    visualization_export_parser.add_argument("--output-dir", default=str(PATHS.visualization_dir))
    visualization_export_parser.add_argument("--html-max-nodes", type=int, default=260)
    return parser


def handle_structured(args: argparse.Namespace) -> int:
    config = FetchConfig.load(args.config)
    pipeline = StructuredFetchPipeline(config)
    try:
        if args.command == "init-db":
            pipeline.init_db()
            print("SQLite 表结构初始化完成")
            return 0
        if args.command == "build":
            pipeline.run()
            print("结构化种子库构建完成")
            return 0
        if args.command == "validate":
            issues = pipeline.validate()
            print(json.dumps(issues, ensure_ascii=False, indent=2))
            return 1 if any(item["level"] == "error" for item in issues) else 0
        paths = pipeline.export_csv()
        print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))
        return 0
    finally:
        pipeline.close()


def handle_unstructured(args: argparse.Namespace) -> int:
    repo_root = PATHS.project_root.parent
    document_count, document_error_count = run_document_preprocess(
        repo_root=repo_root,
        config_path=Path(args.source_config),
        output_path=Path(args.documents_output),
        report_path=Path(args.documents_report),
        strict=args.strict,
    )
    sentence_count, sentence_error_count = run_sentence_preprocess(
        documents_path=Path(args.documents_output),
        output_path=Path(args.sentences_output),
        report_path=Path(args.sentences_report),
        strict=args.strict,
    )
    print(
        json.dumps(
            {
                "document_count": document_count,
                "document_error_count": document_error_count,
                "sentence_count": sentence_count,
                "sentence_error_count": sentence_error_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_dictionary_matcher(entities_csv: str, aliases_csv: str) -> MaxForwardDictionaryMatcher:
    resources = load_dictionary_resources(Path(entities_csv), Path(aliases_csv))
    return MaxForwardDictionaryMatcher(resources)


def handle_mentions(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        tokenized_count = tokenize_sentences_file(Path(args.sentences), Path(args.tokenized_output))
        template_count = extract_gold_seed(
            tokenized_path=Path(args.tokenized_output),
            output_path=Path(args.gold_template_output),
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "tokenized_sentence_count": tokenized_count,
                    "gold_template_count": template_count,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "weak-label":
        matcher = build_dictionary_matcher(args.entities_csv, args.aliases_csv)
        sampled_records, sampling_summary = sample_weak_label_candidates(
            tokenized_path=Path(args.tokenized),
            base_sample_per_doc=args.base_sample_per_doc,
            sample_budget=args.sample_budget,
            targeted_topup_per_keyword=args.targeted_topup_per_keyword,
            seed=args.seed,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary_dir:
            sampled_input_path = Path(temporary_dir) / "weak_label_candidates.jsonl"
            write_jsonl(sampled_input_path, sampled_records)
            accepted_count, rejected_count = weak_label_records(
                tokenized_path=sampled_input_path,
                output_path=output_path,
                reject_report_path=Path(args.reject_report),
                matcher=matcher,
                sleep_seconds=args.sleep_seconds,
                timeout_seconds=args.timeout_seconds,
                progress_every=args.progress_every,
                api_key=args.api_key,
                base_url=args.base_url,
                model_name=args.model_name,
            )
        print(
            json.dumps(
                {
                    "sampling_summary": sampling_summary,
                    "accepted_count": accepted_count,
                    "rejected_count": rejected_count,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "train":
        require_sklearn_crfsuite()
        feature_config = FeatureConfig(
            use_pos=not args.disable_pos,
            use_dict=not args.disable_dict,
            use_time_hint=not args.disable_time_hint,
            window_size=2,
        )
        split_summary = split_weak_and_gold_datasets(
            weak_labeled_path=Path(args.weak_labeled),
            train_output_path=Path(args.train_output),
            dev_output_path=Path(args.dev_output),
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        summary_output_path = Path(args.summary_output)
        summary = {
            "train": summarize_label_distribution(Path(args.train_output), summary_output_path.with_name("train_summary.json")),
            "dev": summarize_label_distribution(Path(args.dev_output), summary_output_path.with_name("dev_summary.json")),
            "test_gold_exists": Path(args.test_gold).exists(),
            "weak_record_count": split_summary["weak_record_count"],
            "accepted_count": split_summary["accepted_count"],
            "train_count": split_summary["train_count"],
            "dev_count": split_summary["dev_count"],
            "dev_ratio": args.dev_ratio,
            "weak_doc_count": split_summary["weak_doc_count"],
            "accepted_doc_count": split_summary["accepted_doc_count"],
            "accepted_doc_coverage": split_summary["accepted_doc_coverage"],
            "train_doc_coverage": split_summary["train_doc_coverage"],
            "dev_doc_coverage": split_summary["dev_doc_coverage"],
            "accepted_review_status_distribution": split_summary["accepted_review_status_distribution"],
            "accepted_weak_label_confidence_distribution": split_summary["accepted_weak_label_confidence_distribution"],
        }
        write_json(summary_output_path, summary)

        matcher = None if args.disable_dict else build_dictionary_matcher(args.entities_csv, args.aliases_csv)
        evaluation = train_from_paths(
            train_path=Path(args.train_output),
            dev_path=Path(args.dev_output),
            output_dir=Path(args.model_dir),
            feature_config=feature_config,
            training_config=CrfTrainingConfig(c1=args.c1, c2=args.c2, max_iterations=args.max_iterations),
            matcher=matcher,
        )
        test_metrics = None
        test_gold_path = Path(args.test_gold)
        if test_gold_path.exists():
            from kg_core.io import read_jsonl

            test_records = read_jsonl(test_gold_path)
            x_test, y_test = build_dataset_features(test_records, feature_config, matcher)
            if x_test:
                model = load_model(Path(args.model_dir) / "crf_model.pkl")
                y_pred = model.predict(x_test)
                test_evaluation = evaluate_predictions(test_records, y_test, y_pred)
                write_json(Path(args.model_dir) / "eval_test_gold.json", test_evaluation)
                test_metrics = test_evaluation["metrics"]
        print(
            json.dumps(
                {
                    "dataset_summary": summary,
                    "dev_metrics": evaluation["metrics"],
                    "test_metrics": test_metrics,
                },
                ensure_ascii=False,
            )
        )
        return 0

    matcher = build_dictionary_matcher(args.entities_csv, args.aliases_csv)
    sentence_count, mention_count = predict_mentions(
        sentences_path=Path(args.sentences),
        output_path=Path(args.output),
        model_path=Path(args.model),
        feature_config_path=Path(args.feature_config),
        matcher=matcher,
    )
    print(json.dumps({"sentence_count": sentence_count, "mention_count": mention_count}, ensure_ascii=False))
    return 0


def handle_linking(args: argparse.Namespace) -> int:
    if args.command == "link":
        config = LinkingConfig(
            top_k=args.top_k,
            link_threshold=args.link_threshold,
            multi_token_link_threshold=args.multi_token_link_threshold,
            single_token_link_threshold=args.single_token_link_threshold,
            review_threshold=args.review_threshold,
            ambiguity_margin_threshold=args.margin_threshold,
            fuzzy_threshold=args.fuzzy_threshold,
            tfidf_recall_limit=args.tfidf_recall_limit,
            anchor_threshold=args.anchor_threshold,
            enable_document_disambiguation=True,
        )
        linked_mentions = link_mentions_from_paths(
            mentions_path=args.mentions,
            sentences_path=args.sentences,
            documents_path=args.documents,
            entities_csv_path=args.entities_csv,
            aliases_csv_path=args.aliases_csv,
            claims_csv_path=args.claims_csv,
            ontology_path=args.ontology,
            output_path=args.output,
            report_path=args.report,
            config=config,
            limit=args.limit,
        )
        linked_count = sum(1 for item in linked_mentions if item.get("decision") == "LINKED")
        review_count = sum(1 for item in linked_mentions if item.get("decision") == "REVIEW")
        nil_count = sum(1 for item in linked_mentions if item.get("decision") == "NIL")
        skipped_count = sum(
            1 for item in linked_mentions if str(item.get("decision") or "").upper().startswith("SKIPPED_")
        )
        print(
            json.dumps(
                {
                    "total": len(linked_mentions),
                    "linked_count": linked_count,
                    "review_count": review_count,
                    "nil_count": nil_count,
                    "skipped_count": skipped_count,
                    "output": args.output,
                    "report": args.report,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "mine-gaps":
        gap_records = mine_linking_gaps_from_path(
            linked_mentions_path=args.linked_mentions,
            output_path=args.output,
            top_n=args.top_n,
        )
        print(
            json.dumps(
                {
                    "gap_count": len(gap_records),
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
        return 0

    evaluation_summary = evaluate_linking_from_paths(
        predictions_path=args.predictions,
        gold_path=args.gold,
        output_path=args.output,
    )
    print(json.dumps(evaluation_summary, ensure_ascii=False))
    return 0


def handle_coreference(args: argparse.Namespace) -> int:
    resolved_mentions = resolve_coreferences_from_paths(
        linked_mentions_path=args.linked_mentions,
        sentences_path=args.sentences,
        output_path=args.output,
        report_path=args.report,
        unresolved_output_path=args.unresolved_output,
        max_sentence_distance=args.max_sentence_distance,
    )
    print(
        json.dumps(
            {
                "total": len(resolved_mentions),
                "linked_by_coref_count": sum(1 for item in resolved_mentions if item.get("decision") == "LINKED_BY_COREF"),
                "coref_unresolved_count": sum(1 for item in resolved_mentions if item.get("decision") == "COREF_UNRESOLVED"),
                "output": args.output,
                "report": args.report,
            },
            ensure_ascii=False,
        )
    )
    return 0


def emit_cli_result(result: object, fallback_payload: dict[str, object]) -> None:
    """统一输出 CLI 结果，方便后续模块按 dict 或对象返回。"""

    if result is None:
        print(json.dumps(fallback_payload, ensure_ascii=False))
        return
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if isinstance(result, Path):
        print(str(result))
        return
    if isinstance(result, (dict, list)):
        print(json.dumps(result, ensure_ascii=False))
        return
    print(result)


def handle_relations(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        from relation_extraction.prepare import prepare_relation_pairs

        result = prepare_relation_pairs(
            resolved_mentions_path=Path(args.resolved_mentions),
            sentences_path=Path(args.sentences),
            tokenized_sentences_path=Path(args.tokenized_sentences),
            entities_csv_path=Path(args.entities_csv),
            aliases_csv_path=Path(args.aliases_csv),
            claims_csv_path=Path(args.claims_csv),
            ontology_path=Path(args.ontology),
            output_path=Path(args.output),
        )
        emit_cli_result(
            result,
            {
                "command": "prepare",
                "resolved_mentions": args.resolved_mentions,
                "sentences": args.sentences,
                "output": args.output,
            },
        )
        return 0

    if args.command == "weak-label":
        from relation_extraction import weak_label_relations

        result = weak_label_relations(
            pair_candidates_path=Path(args.pair_candidates),
            entities_csv_path=Path(args.entities_csv),
            claims_csv_path=Path(args.claims_csv),
            ontology_path=Path(args.ontology),
            output_path=Path(args.output),
        )
        emit_cli_result(result, {"command": "weak-label", "pair_candidates": args.pair_candidates, "output": args.output})
        return 0

    if args.command == "train":
        from relation_extraction import train_relation_model

        result = train_relation_model(
            config_path=Path(args.config),
            output_dir=Path(args.model_dir),
            sentences_path=Path(args.sentences),
            tokenized_sentences_path=Path(args.tokenized_sentences),
            resolved_mentions_path=Path(args.resolved_mentions),
            entities_csv_path=Path(args.entities_csv),
            aliases_csv_path=Path(args.aliases_csv),
            claims_csv_path=Path(args.claims_csv),
            ontology_path=Path(args.ontology),
            pair_candidates_path=Path(args.pair_candidates),
            distant_labeled_path=Path(args.distant_labeled),
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        )
        emit_cli_result(result, {"command": "train", "model_dir": args.model_dir})
        return 0

    if args.command == "predict":
        from relation_extraction import predict_relations

        result = predict_relations(
            model_dir=Path(args.model_dir),
            config_path=Path(args.config),
            sentences_path=Path(args.sentences),
            tokenized_sentences_path=Path(args.tokenized_sentences),
            resolved_mentions_path=Path(args.resolved_mentions),
            entities_csv_path=Path(args.entities_csv),
            aliases_csv_path=Path(args.aliases_csv),
            claims_csv_path=Path(args.claims_csv),
            ontology_path=Path(args.ontology),
            pair_candidates_path=Path(args.pair_candidates),
            distant_labeled_path=Path(args.distant_labeled),
            output_path=Path(args.output),
            threshold=args.threshold,
        )
        emit_cli_result(result, {"command": "predict", "output": args.output})
        return 0

    from relation_extraction import evaluate_relation_predictions

    result = evaluate_relation_predictions(
        model_dir=Path(args.model_dir),
        config_path=Path(args.config),
        output_path=Path(args.report),
        sentences_path=Path(args.sentences),
        tokenized_sentences_path=Path(args.tokenized_sentences),
        resolved_mentions_path=Path(args.resolved_mentions),
        entities_csv_path=Path(args.entities_csv),
        aliases_csv_path=Path(args.aliases_csv),
        claims_csv_path=Path(args.claims_csv),
        ontology_path=Path(args.ontology),
        pair_candidates_path=Path(args.pair_candidates),
        distant_labeled_path=Path(args.distant_labeled),
        split_name=args.split,
    )
    emit_cli_result(result, {"command": "evaluate", "report": args.report})
    return 0


def handle_facts(args: argparse.Namespace) -> int:
    _ = args.resolved_mentions
    if args.command == "generate-candidates":
        from fact_extraction import generate_fact_candidates_from_paths

        result = generate_fact_candidates_from_paths(
            pair_candidates_path=Path(args.pair_candidates),
            sentences_path=Path(args.sentences),
            ontology_path=Path(args.ontology),
            relation_patterns_path=Path(args.patterns),
            output_path=Path(args.fact_candidates),
        )
        emit_cli_result(result, {"command": "generate-candidates", "output": args.fact_candidates})
        return 0

    if args.command == "score":
        from fact_extraction import score_fact_candidates_from_paths

        result = score_fact_candidates_from_paths(
            candidates_path=Path(args.fact_candidates),
            claims_csv_path=Path(args.claims_csv),
            output_path=Path(args.fact_verified),
        )
        emit_cli_result(result, {"command": "score", "output": args.fact_verified})
        return 0

    if args.command == "verify-llm":
        from fact_extraction import verify_fact_candidates_from_paths

        result = verify_fact_candidates_from_paths(
            candidates_path=Path(args.fact_verified),
            output_path=Path(args.fact_verified),
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model_name,
            timeout_seconds=args.timeout_seconds,
        )
        emit_cli_result(result, {"command": "verify-llm", "output": args.fact_verified})
        return 0

    if args.command == "aggregate":
        from fact_extraction import aggregate_fact_candidates_from_paths

        result = aggregate_fact_candidates_from_paths(
            verified_candidates_path=Path(args.fact_verified),
            verified_facts_output_path=Path(args.fact_verified),
            final_facts_output_path=Path(args.facts_final),
            conflicts_output_path=Path(args.fact_conflicts),
        )
        emit_cli_result(result, {"command": "aggregate", "output": args.facts_final})
        return 0

    from fact_extraction import run_fact_extraction

    result = run_fact_extraction(
        pair_candidates_path=Path(args.pair_candidates),
        sentences_path=Path(args.sentences),
        claims_csv_path=Path(args.claims_csv),
        ontology_path=Path(args.ontology),
        relation_patterns_path=Path(args.patterns),
        fact_candidates_output_path=Path(args.fact_candidates),
        verified_facts_output_path=Path(args.fact_verified),
        final_facts_output_path=Path(args.facts_final),
        conflicts_output_path=Path(args.fact_conflicts),
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        timeout_seconds=args.timeout_seconds,
    )
    emit_cli_result(result, {"command": "run", "output": args.facts_final})
    return 0


def handle_visualization(args: argparse.Namespace) -> int:
    from visualization_export import export_visualization_graph

    result = export_visualization_graph(
        entities_csv_path=Path(args.entities_csv),
        claims_csv_path=Path(args.claims_csv),
        event_candidates_csv_path=Path(args.events_csv),
        text_facts_path=Path(args.text_facts),
        output_dir=Path(args.output_dir),
        html_max_nodes=args.html_max_nodes,
    )
    emit_cli_result(result, {"command": "export", "output_dir": args.output_dir})
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.module == "structured":
        return handle_structured(args)
    if args.module == "unstructured":
        return handle_unstructured(args)
    if args.module == "mentions":
        return handle_mentions(args)
    if args.module == "linking":
        return handle_linking(args)
    if args.module == "coreference":
        return handle_coreference(args)
    if args.module == "relations":
        return handle_relations(args)
    if args.module == "facts":
        return handle_facts(args)
    return handle_visualization(args)


if __name__ == "__main__":
    raise SystemExit(main())
