from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import FetchConfig


class RemoteRequestError(RuntimeError):
    """外部请求在重试耗尽后仍失败。"""


@dataclass(slots=True)
class RequestContext:
    """记录当前请求所属作业，方便日志和 fetch_jobs 对齐。"""

    job_name: str
    job_group: str
    request_params_json: dict[str, Any]


class HttpClient:
    """带重试、退避与节流的基础 HTTP 客户端。"""

    def __init__(self, config: FetchConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def get_json(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        final_headers = {"User-Agent": self.config.user_agent}
        if headers:
            final_headers.update(headers)
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info("发起请求: %s attempt=%s context=%s", url, attempt, context)
                request = urllib.request.Request(url, headers=final_headers, method="GET")
                with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
                    payload = response.read().decode("utf-8")
                time.sleep(self.config.sleep_seconds)
                return json.loads(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                wait_seconds = self.config.backoff_base_seconds ** attempt
                self.logger.warning(
                    "请求失败，将重试: url=%s attempt=%s wait=%s error=%s context=%s",
                    url,
                    attempt,
                    wait_seconds,
                    exc,
                    context,
                )
                time.sleep(wait_seconds)
        raise RemoteRequestError(f"请求失败: {url}") from last_error


class WikidataClient:
    """封装 Wikidata SPARQL 查询。"""

    def __init__(self, http_client: HttpClient, config: FetchConfig) -> None:
        self.http_client = http_client
        self.config = config

    def select(
        self,
        query: str,
        context: RequestContext,
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "format": "json",
        }
        url = f"{self.config.wikidata_endpoint}?{urllib.parse.urlencode(params)}"
        payload = self.http_client.get_json(
            url,
            headers={"Accept": "application/sparql-results+json"},
            context=context,
        )
        return payload.get("results", {}).get("bindings", [])


class WikipediaSummaryClient:
    """获取 Wikipedia REST 摘要。"""

    def __init__(self, http_client: HttpClient, config: FetchConfig) -> None:
        self.http_client = http_client
        self.config = config

    def fetch_summary(self, title: str, context: RequestContext) -> dict[str, Any]:
        safe_title = urllib.parse.quote(title, safe="")
        url = self.config.wikipedia_summary_api.format(title=safe_title)
        return self.http_client.get_json(
            url,
            headers={"Accept": "application/json"},
            context=context,
        )
