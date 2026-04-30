from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_csv_records, read_json, read_jsonl, write_json

from .aggregator import aggregate_fact_candidates
from .candidate_generator import generate_fact_candidates, generate_fact_candidates_from_extracted_claims
from .distant_supervision import add_distant_supervision_signals
from .llm_verifier import verify_fact_candidates
from .pattern_rules import load_relation_patterns
from .writer import write_fact_jsonl


def run_fact_extraction(
    *,
    pair_candidates_path: str | Path,
    sentences_path: str | Path,
    claims_csv_path: str | Path,
    ontology_path: str | Path,
    relation_patterns_path: str | Path | None,
    fact_candidates_output_path: str | Path,
    verified_facts_output_path: str | Path,
    final_facts_output_path: str | Path,
    conflicts_output_path: str | Path,
    extracted_claims_path: str | Path | None = None,
    candidate_source: str = "pair_candidates",
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """顺序执行 V1 事实抽取链路。"""

    sentences = read_jsonl(sentences_path)
    ontology = read_json(ontology_path)
    relation_patterns = load_relation_patterns(relation_patterns_path)
    normalized_candidate_source = candidate_source.replace("-", "_")
    if normalized_candidate_source == "extracted_claims":
        if extracted_claims_path is None:
            raise ValueError("candidate_source=extracted_claims 时必须提供 extracted_claims_path。")
        candidates, candidate_summary = generate_fact_candidates_from_extracted_claims(
            read_jsonl(extracted_claims_path),
            sentences,
            ontology,
            relation_patterns=relation_patterns,
        )
    elif normalized_candidate_source == "pair_candidates":
        candidates, candidate_summary = generate_fact_candidates(
            read_jsonl(pair_candidates_path),
            sentences,
            ontology,
            relation_patterns=relation_patterns,
        )
    else:
        raise ValueError(f"未知 facts candidate_source：{candidate_source}")
    write_fact_jsonl(fact_candidates_output_path, candidates)

    scored_candidates, score_summary = add_distant_supervision_signals(
        candidates,
        read_csv_records(claims_csv_path),
    )
    verified_candidates, verify_summary = verify_fact_candidates(
        scored_candidates,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )
    write_fact_jsonl(verified_facts_output_path, verified_candidates)
    verified_facts, final_facts, conflicts, aggregate_summary = aggregate_fact_candidates(verified_candidates)
    write_fact_jsonl(final_facts_output_path, final_facts)
    write_fact_jsonl(conflicts_output_path, conflicts)

    summary = {
        "candidate_source": normalized_candidate_source,
        "generate_candidates": candidate_summary,
        "score": score_summary,
        "verify_llm": verify_summary,
        "aggregate": aggregate_summary,
    }
    write_json(Path(final_facts_output_path).with_suffix(".summary.json"), summary)
    return {
        "fact_candidates_output_path": Path(fact_candidates_output_path).as_posix(),
        "verified_facts_output_path": Path(verified_facts_output_path).as_posix(),
        "final_facts_output_path": Path(final_facts_output_path).as_posix(),
        "conflicts_output_path": Path(conflicts_output_path).as_posix(),
        "summary_path": Path(final_facts_output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }
