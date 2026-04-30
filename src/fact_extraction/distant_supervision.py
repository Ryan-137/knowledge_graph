from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_core.io import read_csv_records, read_jsonl, write_json

from .pattern_rules import canonicalize_predicate
from .schema import FactSignal
from .writer import write_fact_jsonl


def _claim_index(claims: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for claim in claims:
        key = (
            str(claim.get("subject_id") or "").strip(),
            canonicalize_predicate(str(claim.get("predicate") or "")),
            str(claim.get("object_id") or "").strip(),
        )
        if not all(key):
            continue
        index.setdefault(key, []).append(claim)
    return index


def _load_claim_qualifiers(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    qualifiers_by_claim_id: dict[str, dict[str, Any]] = {}
    for claim in claims:
        claim_id = str(claim.get("claim_id") or "")
        raw_qualifiers = str(claim.get("qualifiers_json") or "{}")
        try:
            qualifiers = json.loads(raw_qualifiers)
        except json.JSONDecodeError:
            qualifiers = {}
        qualifiers_by_claim_id[claim_id] = qualifiers if isinstance(qualifiers, dict) else {}
    return qualifiers_by_claim_id


def add_distant_supervision_signals(
    fact_candidates: list[dict[str, Any]],
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _claim_index(claims)
    qualifiers_by_claim_id = _load_claim_qualifiers(claims)
    aligned_count = 0
    scored: list[dict[str, Any]] = []

    for candidate in fact_candidates:
        record = dict(candidate)
        key = (
            str(record.get("subject_id") or ""),
            canonicalize_predicate(str(record.get("predicate") or "")),
            str(record.get("object_id") or ""),
        )
        matched_claims = index.get(key, [])
        signals = list(record.get("signals") or [])
        if matched_claims:
            aligned_count += 1
            matched_claim_ids = [str(claim.get("claim_id") or "") for claim in matched_claims]
            claim_qualifiers = {
                claim_id: qualifiers_by_claim_id.get(claim_id, {})
                for claim_id in matched_claim_ids
                if qualifiers_by_claim_id.get(claim_id)
            }
            qualifiers = dict(record.get("qualifiers") or {})
            if claim_qualifiers:
                qualifiers["claim_qualifiers"] = claim_qualifiers
            record["qualifiers"] = qualifiers
            signals.append(
                FactSignal(
                    "wikidata_alignment",
                    0.25,
                    "MATCHED",
                    {
                        "matched_claim_ids": matched_claim_ids,
                        "matched_claim_count": len(matched_claims),
                    },
                ).to_dict()
            )
        record["signals"] = signals
        record["confidence"] = round(sum(float(signal.get("score", 0.0) or 0.0) for signal in signals), 6)
        record["status"] = "SCORED"
        scored.append(record)

    summary = {
        "candidate_count": len(scored),
        "wikidata_alignment_count": aligned_count,
    }
    return scored, summary


def score_fact_candidates_from_paths(
    *,
    candidates_path: str | Path,
    claims_csv_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    scored, summary = add_distant_supervision_signals(
        read_jsonl(candidates_path),
        read_csv_records(claims_csv_path),
    )
    write_fact_jsonl(output_path, scored)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {
        "output_path": Path(output_path).as_posix(),
        "summary_path": Path(output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }
