from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .claim_transform import build_claim_record, build_event_candidate_from_claim, build_job_key, utc_now_text
from .clients import HttpClient, RequestContext, WikipediaSummaryClient, WikidataClient
from .config import FetchConfig
from .entity_transform import aggregate_alias_rows, aggregate_entity_rows, build_entity_record
from .entity_typing import infer_entity_type
from .queries import build_entity_alias_query, build_entity_base_query
from .relations import RelationTemplate, build_relation_templates
from .repository import StructuredRepository, chunked


def configure_logger(log_path: Path) -> logging.Logger:
    """配置结构化种子库构建日志输出。"""
    logger = logging.getLogger("structured_seed")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class StructuredFetchPipeline:
    """结构化抓取与种子库构建主流程。"""

    def __init__(self, config: FetchConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.logger = configure_logger(config.log_path)
        self.repository = StructuredRepository(config.database_path)
        self.repository.initialize()
        self.http_client = HttpClient(config, self.logger)
        self.wikidata_client = WikidataClient(self.http_client, config)
        self.wikipedia_client = WikipediaSummaryClient(self.http_client, config)
        self.relation_templates = build_relation_templates()

    def close(self) -> None:
        self.repository.close()

    def init_db(self) -> None:
        self.repository.initialize()

    def run(self) -> None:
        self.logger.info("开始执行结构化种子库构建流程")
        seeds = self.config.load_seed_entities()
        self.fetch_seed_entities(seeds)
        self.fetch_claims_for_all_relations()
        self.backfill_missing_entities(job_group="one_hop_entities", hop_label="一跳")
        # 一跳实体落库后再次抓取 claims，才能继续把扩展深度推进到两跳。
        self.fetch_claims_for_all_relations()
        self.backfill_missing_entities(job_group="two_hop_entities", hop_label="两跳")
        self.fetch_wikipedia_summaries()
        self.build_event_candidates()
        issues = self.repository.validate()
        self.logger.info("校验完成，问题数: %s", len(issues))

    def validate(self) -> list[dict[str, str]]:
        issues = self.repository.validate()
        return [
            {
                "check_name": issue.check_name,
                "level": issue.level,
                "message": issue.message,
                "detail": issue.detail,
            }
            for issue in issues
        ]

    def export_csv(self) -> list[Path]:
        return self.repository.export_csv(self.config.csv_export_dir)

    def fetch_seed_entities(self, seeds: list[dict[str, Any]]) -> None:
        qids = [item["qid"] for item in seeds]
        expected_type_map = {item["qid"]: item.get("expected_type") for item in seeds}
        self.fetch_entities_by_qids(qids, expected_type_map, job_group="seed_entities")

    def fetch_entities_by_qids(
        self,
        qids: list[str],
        expected_type_map: dict[str, str | None] | None = None,
        job_group: str = "entities",
    ) -> None:
        if not qids:
            return
        expected_type_map = expected_type_map or {}
        for batch in chunked(sorted(set(qids)), self.config.entity_batch_size):
            base_rows = self._run_paged_query(
                job_name="entity_base",
                job_group=job_group,
                params={"qids": batch},
                page_size=self.config.page_size,
                query_builder=lambda limit, offset: build_entity_base_query(batch, limit, offset),
            )
            alias_rows = self._run_paged_query(
                job_name="entity_aliases",
                job_group=job_group,
                params={"qids": batch},
                page_size=self.config.page_size,
                query_builder=lambda limit, offset: build_entity_alias_query(batch, limit, offset),
            )
            alias_map = aggregate_alias_rows(alias_rows)
            for grouped in aggregate_entity_rows(base_rows).values():
                qid = grouped["entity_id"]
                entity_type = infer_entity_type(
                    grouped["instance_of_ids"],
                    grouped["instance_of_labels"],
                    expected_type_map.get(qid),
                    grouped.get("description_en"),
                    grouped.get("description_zh"),
                    grouped.get("label_en"),
                    grouped.get("label_zh"),
                )
                entity_payload, aliases = build_entity_record(
                    grouped,
                    alias_map.get(qid, {"en": [], "zh": []}),
                    entity_type,
                    self.config.default_confidence,
                )
                self.repository.upsert_entity(entity_payload, aliases)

    def fetch_claims_for_all_relations(self) -> None:
        for template in self.relation_templates:
            if template.name not in self.config.relations:
                continue
            subject_ids = []
            for subject_type in template.subject_types:
                subject_ids.extend(self.repository.get_entity_ids_by_type(subject_type))
            subject_ids = sorted(set(subject_ids))
            if not subject_ids:
                self.logger.info("跳过关系 %s，当前没有匹配的 subject 类型", template.name)
                continue
            rows = self._run_paged_query(
                job_name=template.name,
                job_group="claims",
                params={"subject_ids": subject_ids},
                page_size=self.config.page_size,
                query_builder=lambda limit, offset, template=template, subject_ids=subject_ids: template.build_query(
                    subject_ids,
                    limit,
                    offset,
                ),
            )
            for row in rows:
                claim = build_claim_record(row, template.name, self.config.default_confidence)
                self.repository.upsert_claim(claim)

    def backfill_missing_entities(self, job_group: str, hop_label: str) -> None:
        missing_ids = sorted(self.repository.get_missing_object_entity_ids())
        if not missing_ids:
            self.logger.info("没有需要回填的%s实体", hop_label)
            return
        self.logger.info("开始回填%s实体，数量=%s", hop_label, len(missing_ids))
        self.fetch_entities_by_qids(missing_ids, job_group=job_group)

    def fetch_wikipedia_summaries(self) -> None:
        rows = self.repository.get_entities_missing_summary()
        for row in rows:
            entity_id = row["entity_id"]
            title = row["wikipedia_title_en"]
            if not title:
                continue
            now_text = utc_now_text()
            context = RequestContext(
                job_name="wikipedia_summary",
                job_group="summary",
                request_params_json={"entity_id": entity_id, "title": title},
            )
            job_key = build_job_key(context.job_name, context.request_params_json, None, None)
            self.repository.upsert_fetch_job(
                {
                    "job_key": job_key,
                    "job_name": context.job_name,
                    "job_group": context.job_group,
                    "request_params_json": context.request_params_json,
                    "page_limit": None,
                    "page_offset": None,
                    "status": "running",
                    "retry_count": 0,
                    "last_error": None,
                    "started_at": now_text,
                    "finished_at": None,
                    "updated_at": now_text,
                }
            )
            try:
                payload = self.wikipedia_client.fetch_summary(title, context=context)
                self.repository.update_entity_summary(
                    entity_id=entity_id,
                    summary=payload.get("extract"),
                    retrieved_at=utc_now_text(),
                    source_name="wikipedia_rest",
                    raw_payload_json=payload,
                )
                self.repository.upsert_fetch_job(
                    {
                        "job_key": job_key,
                        "job_name": context.job_name,
                        "job_group": context.job_group,
                        "request_params_json": context.request_params_json,
                        "page_limit": None,
                        "page_offset": None,
                        "status": "success",
                        "retry_count": 0,
                        "last_error": None,
                        "started_at": now_text,
                        "finished_at": utc_now_text(),
                        "updated_at": utc_now_text(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Wikipedia 摘要抓取失败 entity_id=%s title=%s", entity_id, title)
                self.repository.upsert_fetch_job(
                    {
                        "job_key": job_key,
                        "job_name": context.job_name,
                        "job_group": context.job_group,
                        "request_params_json": context.request_params_json,
                        "page_limit": None,
                        "page_offset": None,
                        "status": "failed",
                        "retry_count": self.config.max_retries,
                        "last_error": str(exc),
                        "started_at": now_text,
                        "finished_at": utc_now_text(),
                        "updated_at": utc_now_text(),
                    }
                )

    def build_event_candidates(self) -> None:
        claims = self.repository.list_claims()
        event_candidates: list[dict[str, Any]] = []
        for claim_row in claims:
            event_candidate = build_event_candidate_from_claim(claim_row)
            if event_candidate is not None:
                event_candidates.append(event_candidate)
        self.repository.replace_event_candidates(event_candidates)

    def _run_paged_query(
        self,
        job_name: str,
        job_group: str,
        params: dict[str, Any],
        page_size: int,
        query_builder: Callable[[int, int], str],
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            query = query_builder(page_size, offset)
            context = RequestContext(
                job_name=job_name,
                job_group=job_group,
                request_params_json={**params, "limit": page_size, "offset": offset},
            )
            job_key = build_job_key(job_name, params, page_size, offset)
            started_at = utc_now_text()
            self.repository.upsert_fetch_job(
                {
                    "job_key": job_key,
                    "job_name": job_name,
                    "job_group": job_group,
                    "request_params_json": {**params, "query": query},
                    "page_limit": page_size,
                    "page_offset": offset,
                    "status": "running",
                    "retry_count": 0,
                    "last_error": None,
                    "started_at": started_at,
                    "finished_at": None,
                    "updated_at": started_at,
                }
            )
            try:
                rows = self.wikidata_client.select(query, context=context)
                self.repository.upsert_fetch_job(
                    {
                        "job_key": job_key,
                        "job_name": job_name,
                        "job_group": job_group,
                        "request_params_json": {**params, "query": query},
                        "page_limit": page_size,
                        "page_offset": offset,
                        "status": "success",
                        "retry_count": 0,
                        "last_error": None,
                        "started_at": started_at,
                        "finished_at": utc_now_text(),
                        "updated_at": utc_now_text(),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Wikidata 查询失败 job=%s offset=%s", job_name, offset)
                self.repository.upsert_fetch_job(
                    {
                        "job_key": job_key,
                        "job_name": job_name,
                        "job_group": job_group,
                        "request_params_json": {**params, "query": query},
                        "page_limit": page_size,
                        "page_offset": offset,
                        "status": "failed",
                        "retry_count": self.config.max_retries,
                        "last_error": str(exc),
                        "started_at": started_at,
                        "finished_at": utc_now_text(),
                        "updated_at": utc_now_text(),
                    }
                )
                raise
            all_rows.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        return all_rows


__all__ = ["RelationTemplate", "StructuredFetchPipeline", "configure_logger"]
