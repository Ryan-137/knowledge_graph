from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """统一管理项目路径，避免各脚本重复推断根目录。"""

    project_root: Path

    @classmethod
    def from_script(cls, script_file: str | Path) -> "ProjectPaths":
        return cls(Path(script_file).resolve().parents[1])

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        current = Path.cwd() if start is None else Path(start).resolve()
        if current.is_file():
            current = current.parent
        for candidate in (current, *current.parents):
            if candidate.name == "knowledge_graph" and (candidate / "src").exists():
                return cls(candidate)
            nested = candidate / "knowledge_graph"
            if (nested / "src").exists():
                return cls(nested)
        raise FileNotFoundError("无法从当前路径定位 knowledge_graph 项目根目录")

    @property
    def src_root(self) -> Path:
        return self.project_root / "src"

    @property
    def configs_root(self) -> Path:
        return self.project_root / "configs"

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def structured_dir(self) -> Path:
        return self.processed_root / "structured"

    @property
    def unstructured_dir(self) -> Path:
        return self.processed_root / "unstructured"

    @property
    def mentions_dir(self) -> Path:
        return self.processed_root / "mentions"

    @property
    def linking_dir(self) -> Path:
        return self.processed_root / "linking"

    @property
    def coreference_dir(self) -> Path:
        return self.processed_root / "coreference"

    @property
    def relations_dir(self) -> Path:
        return self.processed_root / "relations"

    @property
    def facts_dir(self) -> Path:
        return self.processed_root / "facts"

    @property
    def events_dir(self) -> Path:
        return self.processed_root / "events"

    @property
    def visualization_dir(self) -> Path:
        return self.processed_root / "visualization"

    @property
    def fusion_dir(self) -> Path:
        return self.processed_root / "fusion"

    @property
    def structured_db(self) -> Path:
        return self.structured_dir / "structured_kg.db"

    @property
    def structured_csv_dir(self) -> Path:
        return self.structured_dir / "csv"

    @property
    def documents_jsonl(self) -> Path:
        return self.unstructured_dir / "documents.jsonl"

    @property
    def sentences_jsonl(self) -> Path:
        return self.unstructured_dir / "sentences.jsonl"

    @property
    def mentions_jsonl(self) -> Path:
        return self.mentions_dir / "mentions.jsonl"

    @property
    def linked_mentions_jsonl(self) -> Path:
        return self.linking_dir / "linked_mentions.jsonl"

    @property
    def resolved_mentions_jsonl(self) -> Path:
        return self.coreference_dir / "resolved_mentions.jsonl"

    @property
    def coreference_report_json(self) -> Path:
        return self.coreference_dir / "coreference_report.json"

    @property
    def relation_model_dir(self) -> Path:
        return self.relations_dir / "model"

    @property
    def pair_candidates_jsonl(self) -> Path:
        return self.relations_dir / "pair_candidates.jsonl"

    @property
    def distant_labeled_jsonl(self) -> Path:
        return self.relations_dir / "distant_labeled.jsonl"

    @property
    def distant_label_review_queue_jsonl(self) -> Path:
        return self.relations_dir / "distant_label_review_queue.jsonl"

    @property
    def relation_gold_jsonl(self) -> Path:
        return self.relations_dir / "relation_gold.jsonl"

    @property
    def relation_predictions_jsonl(self) -> Path:
        return self.relations_dir / "relation_predictions.jsonl"

    @property
    def fact_candidates_jsonl(self) -> Path:
        return self.facts_dir / "fact_candidates.jsonl"

    @property
    def fact_verified_jsonl(self) -> Path:
        return self.facts_dir / "fact_verified.jsonl"

    @property
    def facts_final_jsonl(self) -> Path:
        return self.facts_dir / "facts_final.jsonl"

    @property
    def fact_conflicts_jsonl(self) -> Path:
        return self.facts_dir / "fact_conflicts.jsonl"

    @property
    def event_candidates_text_jsonl(self) -> Path:
        return self.events_dir / "event_candidates_text.jsonl"

    @property
    def verified_events_text_jsonl(self) -> Path:
        return self.events_dir / "verified_events_text.jsonl"

    @property
    def event_arguments_text_jsonl(self) -> Path:
        return self.events_dir / "event_arguments_text.jsonl"

    @property
    def event_to_fact_candidates_jsonl(self) -> Path:
        return self.events_dir / "event_to_fact_candidates.jsonl"

    @property
    def events_text_summary_json(self) -> Path:
        return self.events_dir / "events_text.summary.json"


    @property
    def fusion_corrections_jsonl(self) -> Path:
        return self.fusion_dir / "corrections.jsonl"

    def resolve_project_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        parts = path.parts
        if parts and parts[0] == self.project_root.name:
            return self.project_root.parent.joinpath(path).resolve()
        return (self.project_root / path).resolve()


def ensure_src_on_path(project_paths: ProjectPaths) -> None:
    """脚本入口统一调用，减少每个脚本手写 sys.path 注入。"""

    import sys

    for path in (project_paths.project_root, project_paths.src_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
