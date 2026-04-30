from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_core.io import read_json, read_jsonl, write_json, write_jsonl

from .candidate_generator import build_event_arguments, generate_text_event_candidates
from .verifier import verify_text_events


def run_text_event_extraction(
    *,
    sentences_path: str | Path,
    resolved_mentions_path: str | Path,
    relation_candidates_path: str | Path,
    ontology_path: str | Path,
    relation_patterns_path: str | Path | None,
    event_candidates_output_path: str | Path,
    verified_events_output_path: str | Path,
    event_arguments_output_path: str | Path,
    summary_output_path: str | Path,
) -> dict[str, Any]:
    """执行第一版文本事件抽取，不做跨源融合。"""

    _ = relation_patterns_path
    event_candidates, _candidate_arguments, generate_summary = generate_text_event_candidates(
        read_jsonl(sentences_path),
        read_jsonl(resolved_mentions_path),
        read_jsonl(relation_candidates_path),
    )
    verified_events, verify_summary = verify_text_events(event_candidates, read_json(ontology_path))
    event_arguments = build_event_arguments(verified_events)
    write_jsonl(event_candidates_output_path, event_candidates)
    write_jsonl(verified_events_output_path, verified_events)
    write_jsonl(event_arguments_output_path, event_arguments)
    summary = {
        "generate": generate_summary,
        "verify": verify_summary,
    }
    write_json(summary_output_path, summary)
    return {
        "event_candidates_output_path": Path(event_candidates_output_path).as_posix(),
        "verified_events_output_path": Path(verified_events_output_path).as_posix(),
        "event_arguments_output_path": Path(event_arguments_output_path).as_posix(),
        "summary_path": Path(summary_output_path).as_posix(),
        "summary": summary,
    }
