from __future__ import annotations

from .aggregator import aggregate_fact_candidates, aggregate_fact_candidates_from_paths
from .candidate_generator import generate_fact_candidates, generate_fact_candidates_from_paths
from .distant_supervision import add_distant_supervision_signals, score_fact_candidates_from_paths
from .llm_verifier import verify_fact_candidates, verify_fact_candidates_from_paths
from .pipeline import run_fact_extraction
from .schema import Evidence, FactCandidate, FactSignal, VerifiedFact

__all__ = [
    "Evidence",
    "FactCandidate",
    "FactSignal",
    "VerifiedFact",
    "add_distant_supervision_signals",
    "aggregate_fact_candidates",
    "aggregate_fact_candidates_from_paths",
    "generate_fact_candidates",
    "generate_fact_candidates_from_paths",
    "run_fact_extraction",
    "score_fact_candidates_from_paths",
    "verify_fact_candidates",
    "verify_fact_candidates_from_paths",
]
