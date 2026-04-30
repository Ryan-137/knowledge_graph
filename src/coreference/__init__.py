from __future__ import annotations

from .evaluation import build_coreference_report
from .propagation import resolve_coreferences_from_paths, resolve_coreferences

__all__ = [
    "build_coreference_report",
    "resolve_coreferences",
    "resolve_coreferences_from_paths",
]
