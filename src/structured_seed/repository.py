from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class ValidationIssue:
    """单条校验问题。"""

    check_name: str
    level: str
    message: str
    detail: str


class StructuredRepository:
    """负责 SQLite 表结构、增量写入、校验与 CSV 导出。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        cursor = self.connection.cursor()
        cursor.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                label_en TEXT,
                label_zh TEXT,
                description_en TEXT,
                description_zh TEXT,
                wikipedia_title_en TEXT,
                wikipedia_summary_en TEXT,
                entity_type TEXT NOT NULL,
                external_ids TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS aliases (
                entity_id TEXT NOT NULL,
                alias TEXT NOT NULL,
                alias_lang TEXT,
                normalized_alias TEXT NOT NULL,
                PRIMARY KEY (entity_id, normalized_alias),
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
            );

            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT,
                object_text TEXT,
                statement_id TEXT NOT NULL,
                qualifiers_json TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_record_id TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_candidates (
                event_candidate_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                object_id TEXT,
                start_time_raw TEXT,
                end_time_raw TEXT,
                start_time_norm TEXT,
                end_time_norm TEXT,
                location_id TEXT,
                time_text TEXT,
                source_name TEXT NOT NULL,
                statement_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                raw_payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fetch_jobs (
                job_key TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                job_group TEXT NOT NULL,
                request_params_json TEXT NOT NULL,
                page_limit INTEGER,
                page_offset INTEGER,
                status TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def upsert_entity(self, entity: dict[str, Any], aliases: list[dict[str, Any]]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO entities (
                entity_id, canonical_name, label_en, label_zh, description_en, description_zh,
                wikipedia_title_en, wikipedia_summary_en, entity_type, external_ids,
                source_name, source_record_id, retrieved_at, confidence, raw_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                canonical_name=excluded.canonical_name,
                label_en=excluded.label_en,
                label_zh=excluded.label_zh,
                description_en=excluded.description_en,
                description_zh=excluded.description_zh,
                wikipedia_title_en=COALESCE(excluded.wikipedia_title_en, entities.wikipedia_title_en),
                wikipedia_summary_en=COALESCE(excluded.wikipedia_summary_en, entities.wikipedia_summary_en),
                entity_type=excluded.entity_type,
                external_ids=excluded.external_ids,
                source_name=excluded.source_name,
                source_record_id=excluded.source_record_id,
                retrieved_at=excluded.retrieved_at,
                confidence=excluded.confidence,
                raw_payload_json=excluded.raw_payload_json
            """,
            (
                entity["entity_id"],
                entity["canonical_name"],
                entity.get("label_en"),
                entity.get("label_zh"),
                entity.get("description_en"),
                entity.get("description_zh"),
                entity.get("wikipedia_title_en"),
                entity.get("wikipedia_summary_en"),
                entity["entity_type"],
                json.dumps(entity["external_ids"], ensure_ascii=False, sort_keys=True),
                entity["source_name"],
                entity["source_record_id"],
                entity["retrieved_at"],
                entity["confidence"],
                json.dumps(entity["raw_payload_json"], ensure_ascii=False, sort_keys=True),
            ),
        )
        cursor.execute("DELETE FROM aliases WHERE entity_id = ?", (entity["entity_id"],))
        cursor.executemany(
            """
            INSERT OR REPLACE INTO aliases (entity_id, alias, alias_lang, normalized_alias)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    entity["entity_id"],
                    alias["alias"],
                    alias.get("alias_lang"),
                    alias["normalized_alias"],
                )
                for alias in aliases
            ],
        )
        self.connection.commit()

    def update_entity_summary(
        self,
        entity_id: str,
        summary: str | None,
        retrieved_at: str,
        source_name: str,
        raw_payload_json: dict[str, Any],
    ) -> None:
        cursor = self.connection.cursor()
        current = cursor.execute(
            "SELECT external_ids, raw_payload_json, source_record_id FROM entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if current is None:
            return
        cursor.execute(
            """
            UPDATE entities
            SET wikipedia_summary_en = ?,
                source_name = ?,
                retrieved_at = ?,
                raw_payload_json = ?
            WHERE entity_id = ?
            """,
            (
                summary,
                source_name,
                retrieved_at,
                json.dumps(raw_payload_json, ensure_ascii=False, sort_keys=True),
                entity_id,
            ),
        )
        self.connection.commit()

    def upsert_claim(self, claim: dict[str, Any]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO claims (
                claim_id, subject_id, predicate, object_id, object_text, statement_id,
                qualifiers_json, source_name, source_record_id, retrieved_at, confidence, raw_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                subject_id=excluded.subject_id,
                predicate=excluded.predicate,
                object_id=excluded.object_id,
                object_text=excluded.object_text,
                statement_id=excluded.statement_id,
                qualifiers_json=excluded.qualifiers_json,
                source_name=excluded.source_name,
                source_record_id=excluded.source_record_id,
                retrieved_at=excluded.retrieved_at,
                confidence=excluded.confidence,
                raw_payload_json=excluded.raw_payload_json
            """,
            (
                claim["claim_id"],
                claim["subject_id"],
                claim["predicate"],
                claim.get("object_id"),
                claim.get("object_text"),
                claim["statement_id"],
                json.dumps(claim["qualifiers_json"], ensure_ascii=False, sort_keys=True),
                claim["source_name"],
                claim["source_record_id"],
                claim["retrieved_at"],
                claim["confidence"],
                json.dumps(claim["raw_payload_json"], ensure_ascii=False, sort_keys=True),
            ),
        )
        self.connection.commit()

    def replace_event_candidates(self, event_candidates: list[dict[str, Any]]) -> None:
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM event_candidates")
        cursor.executemany(
            """
            INSERT INTO event_candidates (
                event_candidate_id, event_type, subject_id, object_id, start_time_raw, end_time_raw,
                start_time_norm, end_time_norm, location_id, time_text, source_name, statement_id,
                predicate, raw_payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["event_candidate_id"],
                    item["event_type"],
                    item["subject_id"],
                    item.get("object_id"),
                    item.get("start_time_raw"),
                    item.get("end_time_raw"),
                    item.get("start_time_norm"),
                    item.get("end_time_norm"),
                    item.get("location_id"),
                    item.get("time_text"),
                    item["source_name"],
                    item["statement_id"],
                    item["predicate"],
                    json.dumps(item["raw_payload_json"], ensure_ascii=False, sort_keys=True),
                )
                for item in event_candidates
            ],
        )
        self.connection.commit()

    def upsert_fetch_job(self, job: dict[str, Any]) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO fetch_jobs (
                job_key, job_name, job_group, request_params_json, page_limit, page_offset, status,
                retry_count, last_error, started_at, finished_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                job_name=excluded.job_name,
                job_group=excluded.job_group,
                request_params_json=excluded.request_params_json,
                page_limit=excluded.page_limit,
                page_offset=excluded.page_offset,
                status=excluded.status,
                retry_count=excluded.retry_count,
                last_error=excluded.last_error,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                updated_at=excluded.updated_at
            """,
            (
                job["job_key"],
                job["job_name"],
                job["job_group"],
                json.dumps(job["request_params_json"], ensure_ascii=False, sort_keys=True),
                job.get("page_limit"),
                job.get("page_offset"),
                job["status"],
                job.get("retry_count", 0),
                job.get("last_error"),
                job.get("started_at"),
                job.get("finished_at"),
                job["updated_at"],
            ),
        )
        self.connection.commit()

    def list_entities(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM entities ORDER BY entity_id").fetchall()

    def list_claims(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM claims ORDER BY subject_id, predicate, claim_id").fetchall()

    def list_event_candidates(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM event_candidates ORDER BY event_candidate_id").fetchall()

    def get_entity_ids_by_type(self, entity_type: str) -> list[str]:
        rows = self.connection.execute(
            "SELECT entity_id FROM entities WHERE entity_type = ? ORDER BY entity_id",
            (entity_type,),
        ).fetchall()
        return [row["entity_id"] for row in rows]

    def get_all_entity_ids(self) -> set[str]:
        rows = self.connection.execute("SELECT entity_id FROM entities").fetchall()
        return {row["entity_id"] for row in rows}

    def get_entities_missing_summary(self, limit: int | None = None) -> list[sqlite3.Row]:
        sql = """
            SELECT entity_id, wikipedia_title_en
            FROM entities
            WHERE wikipedia_title_en IS NOT NULL
              AND wikipedia_title_en != ''
              AND (wikipedia_summary_en IS NULL OR wikipedia_summary_en = '')
            ORDER BY entity_id
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return self.connection.execute(sql).fetchall()

    def get_missing_object_entity_ids(self) -> set[str]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT c.object_id
            FROM claims c
            LEFT JOIN entities e ON c.object_id = e.entity_id
            WHERE c.object_id IS NOT NULL
              AND c.object_id != ''
              AND e.entity_id IS NULL
            """
        ).fetchall()
        return {row["object_id"] for row in rows}

    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        issues.extend(self._check_canonical_name())
        issues.extend(self._check_orphan_claims())
        issues.extend(self._check_duplicate_aliases())
        issues.extend(self._check_time_formats())
        return issues

    def export_csv(self, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for table_name in ("entities", "aliases", "claims", "event_candidates", "fetch_jobs"):
            path = output_dir / f"{table_name}.csv"
            rows = self.connection.execute(f"SELECT * FROM {table_name}").fetchall()
            columns = [row["name"] for row in self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if columns:
                    writer.writerow(columns)
                if rows:
                    for row in rows:
                        writer.writerow([row[key] for key in columns])
            paths.append(path)
        return paths

    def _check_canonical_name(self) -> list[ValidationIssue]:
        rows = self.connection.execute(
            "SELECT entity_id FROM entities WHERE canonical_name IS NULL OR TRIM(canonical_name) = ''"
        ).fetchall()
        return [
            ValidationIssue(
                check_name="canonical_name_non_empty",
                level="error",
                message="存在 canonical_name 为空的实体",
                detail=row["entity_id"],
            )
            for row in rows
        ]

    def _check_orphan_claims(self) -> list[ValidationIssue]:
        rows = self.connection.execute(
            """
            SELECT c.claim_id, c.object_id
            FROM claims c
            LEFT JOIN entities e ON c.object_id = e.entity_id
            WHERE c.object_id IS NOT NULL
              AND c.object_id != ''
              AND e.entity_id IS NULL
            """
        ).fetchall()
        return [
            ValidationIssue(
                check_name="orphan_claim_object",
                level="error",
                message="claims 表存在悬挂 object_id",
                detail=f'{row["claim_id"]}:{row["object_id"]}',
            )
            for row in rows
        ]

    def _check_duplicate_aliases(self) -> list[ValidationIssue]:
        rows = self.connection.execute(
            """
            SELECT entity_id, normalized_alias, COUNT(*) AS alias_count
            FROM aliases
            GROUP BY entity_id, normalized_alias
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        return [
            ValidationIssue(
                check_name="duplicate_alias",
                level="warning",
                message="aliases 表存在归一化重复别名",
                detail=f'{row["entity_id"]}:{row["normalized_alias"]}:{row["alias_count"]}',
            )
            for row in rows
        ]

    def _check_time_formats(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        rows = self.connection.execute(
            """
            SELECT event_candidate_id, start_time_norm, end_time_norm
            FROM event_candidates
            """
        ).fetchall()
        for row in rows:
            for field_name in ("start_time_norm", "end_time_norm"):
                value = row[field_name]
                if value is None or value == "":
                    continue
                if len(value) not in (4, 10):
                    issues.append(
                        ValidationIssue(
                            check_name="time_normalization",
                            level="error",
                            message="事件时间标准化格式不合法",
                            detail=f'{row["event_candidate_id"]}:{field_name}={value}',
                        )
                    )
        return issues


def chunked(values: Iterable[str], size: int) -> list[list[str]]:
    """把实体 ID 列表拆成固定大小的分块，避免单次 SPARQL 查询过大。"""
    batch: list[str] = []
    chunks: list[list[str]] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            chunks.append(batch)
            batch = []
    if batch:
        chunks.append(batch)
    return chunks
