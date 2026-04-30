from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl


@dataclass
class FusionReport:
    """记录融合层对最终图视图做过的改动，方便排查来源。"""

    facts_edges: int = 0
    evidence_edges: int = 0
    event_edges: int = 0
    added_edges: int = 0
    rejected_edges: int = 0
    replaced_links: int = 0
    deduped_edges: int = 0
    placeholder_nodes: int = 0
    skipped_duplicate_added_edges: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def load_corrections(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


def fuse_graph_view(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    corrections: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    report = FusionReport()
    _apply_replace_link(nodes, edges, corrections, report)
    normalized_edges = _normalize_edges(edges.values(), report)
    _apply_reject_edge(normalized_edges, corrections, report)
    _apply_add_edge(nodes, normalized_edges, corrections, report)
    normalized_edges = _normalize_edges(normalized_edges.values(), report, count_existing_duplicates=False)
    _count_final_layers(normalized_edges, report)
    return nodes, normalized_edges, report.to_dict()


def normalize_edge_identity(edge: dict[str, Any]) -> dict[str, Any]:
    edge = dict(edge)
    semantic_key = _semantic_key(edge)
    edge["semantic_key"] = semantic_key
    edge["id"] = _edge_id(semantic_key)
    edge.setdefault("fusion_status", "original")
    edge.setdefault("fusion_reason", "")
    edge["display_label"] = _display_label(edge)
    return edge


def _normalize_edges(
    rows: Any,
    report: FusionReport,
    *,
    count_existing_duplicates: bool = True,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        edge = normalize_edge_identity(row)
        edge_id = str(edge["id"])
        if edge_id in normalized:
            if count_existing_duplicates:
                report.deduped_edges += 1
            continue
        normalized[edge_id] = edge
    return normalized


def _semantic_key(edge: dict[str, Any]) -> str:
    source_layer = str(edge.get("source_layer") or "")
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    label = str(edge.get("label") or "")
    if source_layer == "facts":
        return f"FACT::{source}::{label}::{target}::{_qualifier_hash(edge)}"
    if source_layer == "event":
        return f"EVENT_ARG::{source}::{edge.get('role') or label}::{target}"
    return f"EVIDENCE::{source}::{label}::{target}::{_evidence_hash(edge)}"


def _edge_id(semantic_key: str) -> str:
    prefix = semantic_key.split("::", 1)[0]
    digest = hashlib.sha1(semantic_key.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}::{digest}"


def _qualifier_hash(edge: dict[str, Any]) -> str:
    qualifiers = edge.get("qualifiers")
    if qualifiers in ("", None):
        qualifiers = {}
    if isinstance(qualifiers, str):
        raw_text = qualifiers.strip() or "{}"
    else:
        raw_text = json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]


def _evidence_hash(edge: dict[str, Any]) -> str:
    raw_text = "||".join(
        str(edge.get(key) or "")
        for key in ("source_id", "sentence_id", "event_id", "evidence_text")
    )
    return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]


def _display_label(edge: dict[str, Any]) -> str:
    label = str(edge.get("label") or "")
    layer = str(edge.get("source_layer") or "")
    return f"{label} [{layer}]" if layer else label


def _apply_replace_link(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    corrections: list[dict[str, Any]],
    report: FusionReport,
) -> None:
    for correction in corrections:
        if str(correction.get("action") or "") != "REPLACE_LINK":
            continue
        payload = dict(correction.get("payload") or {})
        old_id = str(payload.get("old_entity_id") or correction.get("target_key") or "")
        new_id = str(payload.get("new_entity_id") or "")
        if not old_id or not new_id:
            continue
        _ensure_placeholder_node(nodes, new_id, correction, report)
        correction_changed = False
        for edge in edges.values():
            edge_changed = False
            if edge.get("source") == old_id:
                edge["source"] = new_id
                edge_changed = True
            if edge.get("target") == old_id:
                edge["target"] = new_id
                edge_changed = True
            if edge_changed:
                edge["fusion_status"] = "link_replaced"
                edge["fusion_reason"] = str(correction.get("reason") or "")
                correction_changed = True
        if correction_changed:
            report.replaced_links += 1


def _apply_reject_edge(
    edges: dict[str, dict[str, Any]],
    corrections: list[dict[str, Any]],
    report: FusionReport,
) -> None:
    for correction in corrections:
        if str(correction.get("action") or "") != "REJECT_EDGE":
            continue
        matched_ids = _match_edge_ids(edges, correction)
        for edge_id in matched_ids:
            if edge_id in edges:
                del edges[edge_id]
                report.rejected_edges += 1


def _apply_add_edge(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
    corrections: list[dict[str, Any]],
    report: FusionReport,
) -> None:
    for correction in corrections:
        if str(correction.get("action") or "") != "ADD_EDGE":
            continue
        payload = dict(correction.get("payload") or {})
        edge = {
            "source": str(payload.get("source") or payload.get("subject_id") or ""),
            "target": str(payload.get("target") or payload.get("object_id") or ""),
            "label": str(payload.get("predicate") or payload.get("label") or ""),
            "type": "Directed",
            "source_layer": str(payload.get("source_layer") or "correction"),
            "source_name": str(correction.get("source_module") or payload.get("source_name") or ""),
            "confidence": payload.get("confidence", correction.get("confidence", "")),
            "evidence_text": str(payload.get("evidence_text") or ""),
            "fusion_status": "added",
            "fusion_reason": str(correction.get("reason") or ""),
        }
        _ensure_placeholder_node(nodes, edge["source"], correction, report)
        _ensure_placeholder_node(nodes, edge["target"], correction, report)
        edge = normalize_edge_identity(edge)
        if edge["id"] in edges or _has_same_correction_triple(edges, edge):
            report.skipped_duplicate_added_edges += 1
            continue
        edges[str(edge["id"])] = edge
        report.added_edges += 1


def _ensure_placeholder_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    correction: dict[str, Any],
    report: FusionReport,
) -> None:
    if not node_id or node_id in nodes:
        return
    nodes[node_id] = {
        "id": node_id,
        "label": node_id,
        "type": "placeholder_entity",
        "labels": ["Entity", "placeholder_entity"],
        "description": "",
        "source_layer": "correction_placeholder",
        "source_name": str(correction.get("source_module") or ""),
        "confidence": correction.get("confidence", ""),
        "fusion_status": "placeholder_added",
        "fusion_reason": "auto-created by fusion correction",
    }
    report.placeholder_nodes += 1


def _match_edge_ids(edges: dict[str, dict[str, Any]], correction: dict[str, Any]) -> list[str]:
    target_key = str(correction.get("target_key") or "")
    if target_key in edges:
        return [target_key]
    if target_key:
        semantic_matches = [edge_id for edge_id, edge in edges.items() if edge.get("semantic_key") == target_key]
        if semantic_matches:
            return semantic_matches
    payload = dict(correction.get("payload") or {})
    source = str(payload.get("source") or payload.get("subject_id") or "")
    target = str(payload.get("target") or payload.get("object_id") or "")
    predicate = str(payload.get("predicate") or payload.get("label") or "")
    source_layer = str(payload.get("source_layer") or "")
    if not source or not target or not predicate:
        return []
    return [
        edge_id
        for edge_id, edge in edges.items()
        if edge.get("source") == source
        and edge.get("target") == target
        and edge.get("label") == predicate
        and (not source_layer or edge.get("source_layer") == source_layer)
    ]


def _has_same_correction_triple(edges: dict[str, dict[str, Any]], candidate: dict[str, Any]) -> bool:
    if candidate.get("source_layer") != "correction":
        return False
    return any(
        edge.get("source") == candidate.get("source")
        and edge.get("target") == candidate.get("target")
        and edge.get("label") == candidate.get("label")
        for edge in edges.values()
    )


def _count_final_layers(edges: dict[str, dict[str, Any]], report: FusionReport) -> None:
    for edge in edges.values():
        source_layer = str(edge.get("source_layer") or "")
        if source_layer == "facts":
            report.facts_edges += 1
        elif source_layer == "event":
            report.event_edges += 1
        else:
            report.evidence_edges += 1
