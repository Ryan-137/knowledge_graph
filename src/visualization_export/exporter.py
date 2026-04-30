from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kg_core.io import read_csv_records, read_jsonl, write_csv_records, write_json
from .fusion import fuse_graph_view, load_corrections


GEPHI_NODE_FIELDS = ["Id", "Label", "type", "labels", "description", "source_layer", "source_name", "confidence", "degree"]
GEPHI_EDGE_FIELDS = [
    "Id",
    "Source",
    "Target",
    "Type",
    "Label",
    "display_label",
    "source_layer",
    "source_name",
    "confidence",
    "evidence_text",
    "semantic_key",
    "fusion_status",
    "fusion_reason",
    "role",
]
NEO4J_NODE_FIELDS = [":ID", ":LABEL", "name", "entity_type", "description", "source_layer", "source_name", "confidence:float"]
NEO4J_REL_FIELDS = [
    ":START_ID",
    ":END_ID",
    ":TYPE",
    "edge_id",
    "source_layer",
    "source_name",
    "confidence:float",
    "evidence_text",
    "semantic_key",
    "fusion_status",
    "fusion_reason",
    "role",
    "display_label",
]


def _stable_id(*parts: object) -> str:
    raw_text = "||".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:20]


def _clean_rel_type(value: object) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", str(value or "RELATED_TO")).strip("_").upper()
    return cleaned or "RELATED_TO"


def _clean_label(value: object) -> str:
    label = str(value or "").strip()
    if re.fullmatch(r"[A-Za-z][0-9A-Za-z_]*", label):
        return label
    return "".join(part.title() for part in _clean_rel_type(value).split("_"))


def _neo4j_label(value: object) -> str:
    return _clean_label(value)


def _node_label(row: dict[str, str]) -> str:
    return row.get("label_en") or row.get("canonical_name") or row.get("label_zh") or row["entity_id"]


def _node_description(row: dict[str, str]) -> str:
    return row.get("description_en") or row.get("description_zh") or row.get("wikipedia_summary_en") or ""


def _add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    node_id = str(node["id"])
    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = node
        return
    for key, value in node.items():
        if value not in ("", None, [], {}) and existing.get(key) in ("", None, [], {}):
            existing[key] = value


def _add_literal_node(nodes: dict[str, dict[str, Any]], raw_value: object, literal_type: str) -> str:
    literal_text = str(raw_value or "").strip()
    literal_id = f"literal:{literal_type}:{_stable_id(literal_text)}"
    _add_node(
        nodes,
        {
            "id": literal_id,
            "label": literal_text,
            "type": "Literal",
            "labels": ["Literal", literal_type],
            "description": literal_text,
            "source_layer": "literal",
            "source_name": "",
            "confidence": "",
        },
    )
    return literal_id


def _ensure_entity_node(nodes: dict[str, dict[str, Any]], entity_id: str, source_layer: str) -> None:
    if entity_id and entity_id not in nodes:
        _add_node(
            nodes,
            {
                "id": entity_id,
                "label": entity_id,
                "type": "Entity",
                "labels": ["Entity"],
                "description": "",
                "source_layer": source_layer,
                "source_name": "",
                "confidence": "",
            },
        )


def _add_edge(edges: dict[str, dict[str, Any]], edge: dict[str, Any]) -> None:
    source = str(edge.get("source") or "")
    target = str(edge.get("target") or "")
    if not source or not target:
        return
    edge_id = str(edge.get("id") or _stable_id(source, target, edge.get("label"), edge.get("source_layer")))
    edge["id"] = edge_id
    edges[edge_id] = edge


def _load_entities(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    nodes: dict[str, dict[str, Any]] = {}
    entity_rows: dict[str, dict[str, str]] = {}
    for row in read_csv_records(path):
        entity_id = row["entity_id"]
        entity_rows[entity_id] = row
        entity_type = row.get("entity_type") or "Entity"
        _add_node(
            nodes,
            {
                "id": entity_id,
                "label": _node_label(row),
                "type": entity_type,
                "labels": ["Entity", entity_type],
                "description": _node_description(row),
                "source_layer": "entity",
                "source_name": row.get("source_name", ""),
                "confidence": row.get("confidence", ""),
                "canonical_name": row.get("canonical_name", ""),
                "label_zh": row.get("label_zh", ""),
                "wikipedia_title_en": row.get("wikipedia_title_en", ""),
            },
        )
    return nodes, entity_rows


def _load_fact_edges(path: Path, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
    for row in read_csv_records(path):
        subject_id = row.get("subject_id", "")
        object_id = row.get("object_id", "")
        if not object_id and row.get("object_text"):
            object_id = _add_literal_node(nodes, row["object_text"], "ObjectText")
        _ensure_entity_node(nodes, subject_id, "facts")
        _ensure_entity_node(nodes, object_id, "facts")
        _add_edge(
            edges,
            {
                "id": row.get("claim_id") or _stable_id("fact", subject_id, row.get("predicate"), object_id),
                "source": subject_id,
                "target": object_id,
                "label": row.get("predicate", ""),
                "type": "Directed",
                "source_layer": "facts",
                "source_name": row.get("source_name", ""),
                "confidence": row.get("confidence", ""),
                "evidence_text": row.get("object_text", ""),
                "statement_id": row.get("statement_id", ""),
                "qualifiers": row.get("qualifiers_json", ""),
            },
        )


def _evidence_text_from_fact(row: dict[str, Any]) -> str:
    evidence = row.get("evidence") or []
    if evidence and isinstance(evidence[0], dict):
        return str(evidence[0].get("text") or "")
    return ""


def _primary_evidence_from_fact(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence") or []
    if evidence and isinstance(evidence[0], dict):
        return dict(evidence[0])
    return {}


def _load_text_fact_edges(path: Path, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
    if not path.exists():
        return
    for row in read_jsonl(path):
        evidence = _primary_evidence_from_fact(row)
        subject_id = str(row.get("subject_id") or "")
        object_id = str(row.get("object_id") or "")
        if not object_id and row.get("object_text"):
            object_id = _add_literal_node(nodes, row["object_text"], "ObjectText")
        _ensure_entity_node(nodes, subject_id, "evidence")
        _ensure_entity_node(nodes, object_id, "evidence")
        _add_edge(
            edges,
            {
                "id": row.get("fact_id") or row.get("claim_candidate_id") or _stable_id("evidence", subject_id, row.get("predicate"), object_id),
                "source": subject_id,
                "target": object_id,
                "label": row.get("predicate", ""),
                "type": "Directed",
                "source_layer": "evidence",
                "source_name": row.get("extractor", "text_re"),
                "confidence": row.get("confidence", ""),
                "evidence_text": row.get("evidence_text") or _evidence_text_from_fact(row),
                "status": row.get("status", ""),
                "qualifiers": row.get("qualifiers", {}),
                "sentence_id": evidence.get("sentence_id", ""),
                "source_id": evidence.get("source_id", ""),
                "doc_id": evidence.get("doc_id", ""),
            },
        )


def _event_label(row: dict[str, str], entity_rows: dict[str, dict[str, str]]) -> str:
    subject_name = _node_label(entity_rows[row["subject_id"]]) if row.get("subject_id") in entity_rows else row.get("subject_id", "")
    object_name = _node_label(entity_rows[row["object_id"]]) if row.get("object_id") in entity_rows else row.get("object_id", "")
    event_type = row.get("event_type") or "Event"
    if object_name:
        return f"{event_type}: {subject_name} -> {object_name}"
    return f"{event_type}: {subject_name}"


def _event_roles(row: dict[str, str]) -> list[dict[str, str]]:
    roles_text = row.get("roles_json") or ""
    if roles_text:
        roles = json.loads(roles_text)
        if isinstance(roles, list):
            return [role for role in roles if isinstance(role, dict) and role.get("entity_id") and role.get("role")]
    roles = [
        {"role": "subject", "entity_id": row.get("subject_id", "")},
        {"role": "object", "entity_id": row.get("object_id", "")},
    ]
    if row.get("location_id"):
        roles.append({"role": "location", "entity_id": row["location_id"]})
    return [role for role in roles if role.get("entity_id")]


def _load_event_graph(path: Path, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]], entity_rows: dict[str, dict[str, str]]) -> None:
    if not path.exists():
        return
    for row in read_csv_records(path):
        event_id = row.get("event_id") or row["event_candidate_id"]
        event_type = row.get("event_type") or "Event"
        _add_node(
            nodes,
            {
                "id": event_id,
                "label": _event_label(row, entity_rows),
                "type": event_type,
                "labels": ["Event", event_type],
                "description": row.get("predicate", ""),
                "source_layer": "event",
                "source_name": row.get("source_name", ""),
                "confidence": row.get("confidence", ""),
                "event_type": event_type,
                "node_type": "event",
                "event_status": row.get("status", "VERIFIED"),
                "start_time": row.get("start_time_norm", ""),
                "end_time": row.get("end_time_norm", ""),
            },
        )
        for role in _event_roles(row):
            role_name = str(role.get("role", "")).strip()
            target_id = str(role.get("entity_id", "")).strip()
            if target_id:
                _ensure_entity_node(nodes, target_id, "event")
                _add_edge(
                    edges,
                    {
                        "id": _stable_id(event_id, role_name, target_id),
                        "source": event_id,
                        "target": target_id,
                        "label": "EXT_EVENT_ARG",
                        "role": role_name,
                        "type": "Directed",
                        "source_layer": "event",
                        "source_name": row.get("source_name", ""),
                        "confidence": row.get("confidence", ""),
                        "evidence_text": row.get("time_text", ""),
                        "event_id": event_id,
                    },
                )
        time_value = row.get("start_time_norm") or row.get("time_text")
        if time_value:
            time_node_id = _add_literal_node(nodes, time_value, "Time")
            _add_edge(
                edges,
                {
                    "id": _stable_id(event_id, "EVENT_TIME", time_node_id),
                    "source": event_id,
                    "target": time_node_id,
                    "label": "EXT_EVENT_ARG",
                    "role": "time",
                    "type": "Directed",
                    "source_layer": "event",
                    "source_name": row.get("source_name", ""),
                    "confidence": "",
                    "evidence_text": row.get("time_text", ""),
                    "event_id": event_id,
                },
            )


def _attach_degrees(nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> None:
    degree_counter: Counter[str] = Counter()
    for edge in edges.values():
        degree_counter[str(edge["source"])] += 1
        degree_counter[str(edge["target"])] += 1
    for node_id, node in nodes.items():
        node["degree"] = degree_counter[node_id]


def _build_stats(nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(Counter(str(node.get("type", "")) for node in nodes.values())),
        "edge_layer_counts": dict(Counter(str(edge.get("source_layer", "")) for edge in edges.values())),
        "edge_type_counts": dict(Counter(str(edge.get("label", "")) for edge in edges.values())),
    }


def _write_gephi_csv(output_dir: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[Path, Path]:
    nodes_path = output_dir / "nodes.csv"
    edges_path = output_dir / "edges.csv"
    write_csv_records(
        nodes_path,
        (
            {
                "Id": node["id"],
                "Label": node["label"],
                "type": node.get("type", ""),
                "labels": ";".join(node.get("labels", [])),
                "description": node.get("description", ""),
                "source_layer": node.get("source_layer", ""),
                "source_name": node.get("source_name", ""),
                "confidence": node.get("confidence", ""),
                "degree": node.get("degree", 0),
            }
            for node in nodes
        ),
        GEPHI_NODE_FIELDS,
    )
    write_csv_records(
        edges_path,
        (
            {
                "Id": edge["id"],
                "Source": edge["source"],
                "Target": edge["target"],
                "Type": edge.get("type", "Directed"),
                "Label": edge.get("display_label") or edge.get("label", ""),
                "display_label": edge.get("display_label", ""),
                "source_layer": edge.get("source_layer", ""),
                "source_name": edge.get("source_name", ""),
                "confidence": edge.get("confidence", ""),
                "evidence_text": edge.get("evidence_text", ""),
                "semantic_key": edge.get("semantic_key", ""),
                "fusion_status": edge.get("fusion_status", ""),
                "fusion_reason": edge.get("fusion_reason", ""),
                "role": edge.get("role", ""),
            }
            for edge in edges
        ),
        GEPHI_EDGE_FIELDS,
    )
    return nodes_path, edges_path


def _write_neo4j_csv(output_dir: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> tuple[Path, Path]:
    neo4j_dir = output_dir / "neo4j"
    nodes_path = neo4j_dir / "nodes.csv"
    rels_path = neo4j_dir / "relationships.csv"
    write_csv_records(
        nodes_path,
        (
            {
                ":ID": node["id"],
                ":LABEL": ";".join(_neo4j_label(label) for label in node.get("labels", [])),
                "name": node["label"],
                "entity_type": node.get("type", ""),
                "description": node.get("description", ""),
                "source_layer": node.get("source_layer", ""),
                "source_name": node.get("source_name", ""),
                "confidence:float": node.get("confidence", ""),
            }
            for node in nodes
        ),
        NEO4J_NODE_FIELDS,
    )
    write_csv_records(
        rels_path,
        (
            {
                ":START_ID": edge["source"],
                ":END_ID": edge["target"],
                ":TYPE": _clean_rel_type(edge.get("label", "")),
                "edge_id": edge["id"],
                "source_layer": edge.get("source_layer", ""),
                "source_name": edge.get("source_name", ""),
                "confidence:float": edge.get("confidence", ""),
                "evidence_text": edge.get("evidence_text", ""),
                "semantic_key": edge.get("semantic_key", ""),
                "fusion_status": edge.get("fusion_status", ""),
                "fusion_reason": edge.get("fusion_reason", ""),
                "role": edge.get("role", ""),
                "display_label": edge.get("display_label", ""),
            }
            for edge in edges
        ),
        NEO4J_REL_FIELDS,
    )
    return nodes_path, rels_path


def _cypher_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _write_neo4j_load_script(output_dir: Path, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> Path:
    neo4j_dir = output_dir / "neo4j"
    script_path = neo4j_dir / "load_csv.cypher"
    node_labels = sorted({_clean_label(label) for node in nodes for label in node.get("labels", []) if label})
    rel_types = sorted({_clean_rel_type(edge.get("label", "")) for edge in edges})
    lines = [
        "// 将 nodes.csv 和 relationships.csv 放到 Neo4j import 目录后，在 Neo4j Browser 或 cypher-shell 中执行本脚本。",
        "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KgNode) REQUIRE n.id IS UNIQUE;",
        "",
        "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row",
        "MERGE (n:KgNode {id: row.`:ID`})",
        "SET n.name = row.name,",
        "    n.neo4j_labels = row.`:LABEL`,",
        "    n.entity_type = row.entity_type,",
        "    n.description = row.description,",
        "    n.source_layer = row.source_layer,",
        "    n.source_name = row.source_name,",
        "    n.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END;",
        "",
    ]
    for label in node_labels:
        lines.extend(
            [
                "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row",
                f"WITH row WHERE row.`:LABEL` CONTAINS {_cypher_string(label)}",
                "MATCH (n:KgNode {id: row.`:ID`})",
                f"SET n:`{label}`;",
                "",
            ]
        )
    for rel_type in rel_types:
        lines.extend(
            [
                "LOAD CSV WITH HEADERS FROM 'file:///relationships.csv' AS row",
                f"WITH row WHERE row.`:TYPE` = {_cypher_string(rel_type)}",
                "MATCH (source:KgNode {id: row.`:START_ID`})",
                "MATCH (target:KgNode {id: row.`:END_ID`})",
                f"MERGE (source)-[r:`{rel_type}` {{edge_id: row.edge_id}}]->(target)",
                "SET r.source_layer = row.source_layer,",
                "    r.source_name = row.source_name,",
                "    r.confidence = CASE WHEN row.`confidence:float` = '' THEN null ELSE toFloat(row.`confidence:float`) END,",
                "    r.evidence_text = row.evidence_text,",
                "    r.semantic_key = row.semantic_key,",
                "    r.fusion_status = row.fusion_status,",
                "    r.fusion_reason = row.fusion_reason,",
                "    r.role = row.role,",
                "    r.display_label = row.display_label;",
                "",
            ]
        )
    lines.extend(
        [
            "MATCH (n:KgNode) RETURN labels(n) AS labels, count(*) AS count ORDER BY count DESC;",
            "MATCH ()-[r]->() RETURN type(r) AS relation, r.source_layer AS layer, count(*) AS count ORDER BY count DESC;",
        ]
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _write_neo4j_browser_queries(output_dir: Path) -> Path:
    neo4j_dir = output_dir / "neo4j"
    queries_path = neo4j_dir / "browser_queries.cypher"
    queries = """// Neo4j Browser 可视化查询：每次复制一个查询到 Browser 执行，切换到 Graph 视图。

// 1. 全量图：显示所有节点和所有关系
MATCH (n:KgNode)
OPTIONAL MATCH p = (n)-[r]->(m:KgNode)
RETURN n, p;

// 2. Alan Turing 两跳核心网络
MATCH p = (turing:KgNode {id: 'Q7251'})-[*1..2]-(neighbor:KgNode)
RETURN p
LIMIT 180;

// 3. facts 层：Wikidata 结构化事实骨架
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'facts'
RETURN p
LIMIT 120;

// 4. evidence 层：文本抽取证据关系
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'evidence'
RETURN p
LIMIT 120;

// 5. wikipedia_direct 层：从 Wikipedia infobox 直连抽取的结构化事实
MATCH p = (source:KgNode)-[r]->(target:KgNode)
WHERE r.source_layer = 'wikipedia_direct'
RETURN p
LIMIT 80;

// 6. 事件节点网络：事件作为独立节点连接人物、作品、地点和时间
MATCH p = (event:Event)-[r]->(target:KgNode)
RETURN p
LIMIT 160;

// 7. 高连接节点总览
MATCH (n:KgNode)
WITH n, size((n)--()) AS degree
ORDER BY degree DESC
LIMIT 30
MATCH p = (n)--(m)
RETURN p
LIMIT 220;
"""
    queries_path.write_text(queries, encoding="utf-8")
    return queries_path


def _write_neo4j_browser_style(output_dir: Path) -> Path:
    neo4j_dir = output_dir / "neo4j"
    style_path = neo4j_dir / "browser_style.grass"
    style = """node {
  diameter: 34px;
  color: #6c757d;
  border-color: #424242;
  border-width: 1.5px;
  text-color-internal: #ffffff;
  caption: '{name}';
  font-size: 10px;
}

node.Person { color: #d1495b; diameter: 52px; }
node.Organization { color: #00798c; }
node.Place { color: #edae49; }
node.Concept { color: #30638e; }
node.Machine { color: #4d908e; }
node.Work { color: #7b2cbf; }
node.Award { color: #f77f00; }
node.Event { color: #2a9d8f; }
node.WikipediaPage { color: #3a86ff; diameter: 30px; }
node.Literal { color: #6c757d; diameter: 26px; }

relationship {
  color: #8d99ae;
  shaft-width: 2px;
  caption: '<type>';
  font-size: 9px;
}

relationship.PERSON,
relationship.BIRTH_PLACE,
relationship.DEATH_PLACE,
relationship.STUDENT,
relationship.INSTITUTION,
relationship.EMPLOYEE,
relationship.EMPLOYER,
relationship.AUTHOR,
relationship.WORK,
relationship.PROPOSER,
relationship.CONCEPT,
relationship.DESIGNER,
relationship.MACHINE,
relationship.RECIPIENT,
relationship.AWARD,
relationship.EVENT_TIME,
relationship.EXT_EVENT_ARG,
relationship.LOCATION {
  color: #2a9d8f;
}
"""
    style_path.write_text(style, encoding="utf-8")
    return style_path


def _write_neo4j_visualization_assets(
    output_dir: Path,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    load_script_path = _write_neo4j_load_script(output_dir, nodes, edges)
    browser_queries_path = _write_neo4j_browser_queries(output_dir)
    browser_style_path = _write_neo4j_browser_style(output_dir)
    return load_script_path, browser_queries_path, browser_style_path


def _html_payload(graph_payload: dict[str, Any], max_nodes: int) -> str:
    nodes = sorted(graph_payload["nodes"], key=lambda item: int(item.get("degree", 0)), reverse=True)[:max_nodes]
    kept_ids = {node["id"] for node in nodes}
    edges = [edge for edge in graph_payload["edges"] if edge["source"] in kept_ids and edge["target"] in kept_ids]
    payload = {"nodes": nodes, "edges": edges, "stats": graph_payload["stats"]}
    graph_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>图灵知识图谱可视化</title>
<style>
html, body {{ margin: 0; height: 100%; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f7f7f2; color: #202124; }}
#toolbar {{ position: fixed; top: 0; left: 0; right: 0; height: 54px; display: flex; align-items: center; gap: 16px; padding: 0 18px; background: #1f2933; color: white; z-index: 2; box-sizing: border-box; }}
#toolbar strong {{ font-size: 16px; }}
#toolbar span {{ font-size: 13px; color: #d8dee9; }}
#graph {{ position: fixed; inset: 54px 320px 0 0; }}
#side {{ position: fixed; top: 54px; right: 0; bottom: 0; width: 320px; overflow: auto; border-left: 1px solid #d9d9d2; background: #ffffff; padding: 16px; box-sizing: border-box; }}
.row {{ margin: 10px 0; font-size: 13px; line-height: 1.5; overflow-wrap: anywhere; }}
canvas {{ width: 100%; height: 100%; display: block; }}
</style>
</head>
<body>
<div id="toolbar"><strong>图灵知识图谱</strong><span id="summary"></span></div>
<div id="graph"><canvas id="canvas"></canvas></div>
<aside id="side"><div id="details">点击节点查看详情</div></aside>
<script id="graph-data" type="application/json">{graph_json}</script>
<script>
const graph = JSON.parse(document.getElementById("graph-data").textContent);
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const box = document.getElementById("graph");
const details = document.getElementById("details");
const colors = {{Person:"#d1495b", Organization:"#00798c", Place:"#edae49", Concept:"#30638e", Machine:"#4d908e", Work:"#7b2cbf", Award:"#f77f00", WikipediaPage:"#3a86ff", Literal:"#6c757d", Event:"#2a9d8f", PublicationEvent:"#2a9d8f", EducationEvent:"#43aa8b", EmploymentEvent:"#577590", HonorEvent:"#f8961e"}};
const nodes = graph.nodes.map((node, index) => ({{...node, x: 80 + (index % 24) * 32, y: 80 + Math.floor(index / 24) * 32, vx: 0, vy: 0}}));
const nodeById = new Map(nodes.map(node => [node.id, node]));
const edges = graph.edges.map(edge => ({{...edge, sourceNode: nodeById.get(edge.source), targetNode: nodeById.get(edge.target)}})).filter(edge => edge.sourceNode && edge.targetNode);
document.getElementById("summary").textContent = `${{graph.stats.node_count}} 个节点 / ${{graph.stats.edge_count}} 条边，当前预览 ${{nodes.length}} 个高连接节点`;
function resize() {{
  canvas.width = box.clientWidth * devicePixelRatio;
  canvas.height = box.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
}}
function tick() {{
  const width = box.clientWidth;
  const height = box.clientHeight;
  for (const node of nodes) {{
    node.vx += (width / 2 - node.x) * 0.0008;
    node.vy += (height / 2 - node.y) * 0.0008;
  }}
  for (let i = 0; i < nodes.length; i++) {{
    for (let j = i + 1; j < nodes.length; j++) {{
      const a = nodes[i], b = nodes[j];
      const dx = a.x - b.x, dy = a.y - b.y;
      const distance2 = Math.max(dx * dx + dy * dy, 25);
      const force = 900 / distance2;
      a.vx += dx * force * 0.01; a.vy += dy * force * 0.01;
      b.vx -= dx * force * 0.01; b.vy -= dy * force * 0.01;
    }}
  }}
  for (const edge of edges) {{
    const dx = edge.targetNode.x - edge.sourceNode.x;
    const dy = edge.targetNode.y - edge.sourceNode.y;
    const distance = Math.max(Math.hypot(dx, dy), 1);
    const force = (distance - 130) * 0.002;
    edge.sourceNode.vx += dx / distance * force;
    edge.sourceNode.vy += dy / distance * force;
    edge.targetNode.vx -= dx / distance * force;
    edge.targetNode.vy -= dy / distance * force;
  }}
  for (const node of nodes) {{
    node.vx *= 0.84; node.vy *= 0.84;
    node.x = Math.max(18, Math.min(width - 18, node.x + node.vx));
    node.y = Math.max(18, Math.min(height - 18, node.y + node.vy));
  }}
}}
function draw() {{
  ctx.clearRect(0, 0, box.clientWidth, box.clientHeight);
  ctx.lineWidth = 1;
  ctx.font = "12px Arial";
  for (const edge of edges) {{
    ctx.strokeStyle = edge.source_layer === "facts" ? "rgba(20,20,20,.25)" : edge.source_layer === "evidence" ? "rgba(209,73,91,.28)" : edge.source_layer === "wikipedia_direct" ? "rgba(58,134,255,.35)" : "rgba(42,157,143,.25)";
    ctx.beginPath(); ctx.moveTo(edge.sourceNode.x, edge.sourceNode.y); ctx.lineTo(edge.targetNode.x, edge.targetNode.y); ctx.stroke();
  }}
  for (const node of nodes) {{
    const radius = Math.min(22, 5 + Math.sqrt(Number(node.degree || 0)) * 2.5);
    ctx.fillStyle = colors[node.type] || colors.Event;
    ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, Math.PI * 2); ctx.fill();
    if (Number(node.degree || 0) >= 3) {{
      ctx.fillStyle = "#202124";
      ctx.fillText(String(node.label).slice(0, 24), node.x + radius + 4, node.y + 4);
    }}
  }}
}}
function frame() {{ tick(); draw(); requestAnimationFrame(frame); }}
function showItem(item) {{
  details.innerHTML = Object.entries(item).filter(([key, value]) => !["vx","vy","x","y"].includes(key) && value !== "").map(([key, value]) => `<div class="row"><strong>${{key}}</strong><br>${{String(value).replace(/[&<>]/g, s => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[s]))}}</div>`).join("");
}}
canvas.addEventListener("click", event => {{
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let best = null, bestDistance = 16;
  for (const node of nodes) {{
    const distance = Math.hypot(node.x - x, node.y - y);
    if (distance < bestDistance) {{ best = node; bestDistance = distance; }}
  }}
  if (best) showItem(best);
}});
resize();
window.addEventListener("resize", resize);
frame();
</script>
</body>
</html>
"""


def _write_html(output_dir: Path, graph_payload: dict[str, Any], max_nodes: int) -> Path:
    html_path = output_dir / "graph.html"
    html_path.write_text(_html_payload(graph_payload, max_nodes), encoding="utf-8")
    return html_path


def export_visualization_graph(
    *,
    entities_csv_path: Path,
    claims_csv_path: Path,
    event_candidates_csv_path: Path,
    text_facts_path: Path,
    output_dir: Path,
    corrections_path: Path | None = None,
    html_max_nodes: int = 260,
) -> dict[str, Any]:
    """汇总结构化事实、事件节点和文本证据，产出最终可视化图谱文件。"""

    nodes, entity_rows = _load_entities(entities_csv_path)
    edges: dict[str, dict[str, Any]] = {}
    _load_fact_edges(claims_csv_path, nodes, edges)
    _load_event_graph(event_candidates_csv_path, nodes, edges, entity_rows)
    _load_text_fact_edges(text_facts_path, nodes, edges)
    nodes, edges, fusion_report = fuse_graph_view(nodes, edges, load_corrections(corrections_path))
    _attach_degrees(nodes, edges)

    node_rows = sorted(nodes.values(), key=lambda item: str(item["id"]))
    edge_rows = sorted(edges.values(), key=lambda item: str(item["id"]))
    graph_payload = {"nodes": node_rows, "edges": edge_rows, "stats": _build_stats(nodes, edges)}

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_json_path = output_dir / "graph.json"
    stats_path = output_dir / "stats.json"
    fusion_report_path = output_dir / "fusion_report.json"
    write_json(graph_json_path, graph_payload)
    write_json(stats_path, graph_payload["stats"])
    write_json(fusion_report_path, fusion_report)
    gephi_nodes_path, gephi_edges_path = _write_gephi_csv(output_dir, node_rows, edge_rows)
    neo4j_nodes_path, neo4j_rels_path = _write_neo4j_csv(output_dir, node_rows, edge_rows)
    neo4j_load_path, neo4j_queries_path, neo4j_style_path = _write_neo4j_visualization_assets(output_dir, node_rows, edge_rows)
    html_path = _write_html(output_dir, graph_payload, html_max_nodes)
    return {
        "node_count": graph_payload["stats"]["node_count"],
        "edge_count": graph_payload["stats"]["edge_count"],
        "graph_json": graph_json_path.as_posix(),
        "stats_json": stats_path.as_posix(),
        "fusion_report_json": fusion_report_path.as_posix(),
        "gephi_nodes_csv": gephi_nodes_path.as_posix(),
        "gephi_edges_csv": gephi_edges_path.as_posix(),
        "neo4j_nodes_csv": neo4j_nodes_path.as_posix(),
        "neo4j_relationships_csv": neo4j_rels_path.as_posix(),
        "neo4j_load_csv_cypher": neo4j_load_path.as_posix(),
        "neo4j_browser_queries": neo4j_queries_path.as_posix(),
        "neo4j_browser_style": neo4j_style_path.as_posix(),
        "html": html_path.as_posix(),
        "edge_layer_counts": graph_payload["stats"]["edge_layer_counts"],
    }
