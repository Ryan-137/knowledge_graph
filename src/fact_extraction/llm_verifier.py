from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from kg_core.io import read_jsonl, write_json

from .schema import FactSignal
from .writer import write_fact_jsonl


SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
UNCERTAIN = "UNCERTAIN"


def _has_signal(record: dict[str, Any], signal_name: str) -> bool:
    return any(str(signal.get("name")) == signal_name for signal in record.get("signals", []))


def _offline_verify(record: dict[str, Any]) -> dict[str, Any]:
    if _has_signal(record, "pattern_match"):
        return {
            "label": SUPPORTED,
            "confidence": 0.85,
            "evidence_span": record.get("evidence", {}).get("text", ""),
            "reason": "候选已命中高精度关系触发词，离线校验视为支持。",
            "mode": "offline_pattern",
        }
    if _has_signal(record, "wikidata_alignment"):
        return {
            "label": UNCERTAIN,
            "confidence": 0.5,
            "evidence_span": "",
            "reason": "候选与结构化事实对齐，但证据句缺少触发词，等待人工或在线 LLM 校验。",
            "mode": "offline_kb_alignment",
        }
    return {
        "label": UNCERTAIN,
        "confidence": 0.0,
        "evidence_span": "",
        "reason": "缺少可解释支持信号。",
        "mode": "offline_abstain",
    }


def _build_prompt(record: dict[str, Any]) -> str:
    evidence = dict(record.get("evidence") or {})
    return (
        "Given the sentence and linked entities, decide whether the candidate fact is explicitly supported.\n"
        "Return JSON only with label, confidence, evidence_span and reason.\n"
        "Allowed labels: SUPPORTED, NOT_SUPPORTED, UNCERTAIN.\n\n"
        f"Sentence:\n{evidence.get('text', '')}\n\n"
        "Candidate:\n"
        f"subject: {record.get('subject_text')} ({record.get('subject_id')})\n"
        f"predicate: {record.get('predicate')}\n"
        f"object: {record.get('object_text')} ({record.get('object_id')})\n"
    )


def _online_verify(record: dict[str, Any], *, api_key: str, base_url: str, model_name: str, timeout_seconds: int) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You verify relation extraction candidates. Do not infer facts not stated in the sentence."},
            {"role": "user", "content": _build_prompt(record)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw_payload = json.loads(response.read().decode("utf-8"))
    content = raw_payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    label = str(parsed.get("label") or UNCERTAIN).upper()
    if label not in {SUPPORTED, NOT_SUPPORTED, UNCERTAIN}:
        label = UNCERTAIN
    return {
        "label": label,
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "evidence_span": str(parsed.get("evidence_span") or ""),
        "reason": str(parsed.get("reason") or ""),
        "mode": "openai_compatible",
    }


def verify_fact_candidates(
    fact_candidates: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout_seconds: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    resolved_model_name = model_name or os.environ.get("OPENAI_MODEL")
    use_online = bool(resolved_api_key and resolved_base_url and resolved_model_name)
    verified: list[dict[str, Any]] = []
    label_counts: dict[str, int] = {}

    for candidate in fact_candidates:
        record = dict(candidate)
        verification = (
            _online_verify(
                record,
                api_key=str(resolved_api_key),
                base_url=str(resolved_base_url),
                model_name=str(resolved_model_name),
                timeout_seconds=timeout_seconds,
            )
            if use_online
            else _offline_verify(record)
        )
        label = str(verification["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
        signals = list(record.get("signals") or [])
        if label == SUPPORTED:
            signals.append(FactSignal("llm_verify", 0.25, SUPPORTED, verification).to_dict())
        elif label == NOT_SUPPORTED:
            signals.append(FactSignal("llm_verify", -1.0, NOT_SUPPORTED, verification).to_dict())
        else:
            signals.append(FactSignal("llm_verify", 0.0, UNCERTAIN, verification).to_dict())
        record["signals"] = signals
        record["llm_verification"] = verification
        record["confidence"] = round(sum(float(signal.get("score", 0.0) or 0.0) for signal in signals), 6)
        record["status"] = "REJECTED" if label == NOT_SUPPORTED else "VERIFIED"
        verified.append(record)

    summary = {
        "candidate_count": len(verified),
        "verification_mode": "openai_compatible" if use_online else "offline",
        "label_counts": dict(sorted(label_counts.items())),
    }
    return verified, summary


def verify_fact_candidates_from_paths(
    *,
    candidates_path: str | Path,
    output_path: str | Path,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    verified, summary = verify_fact_candidates(
        read_jsonl(candidates_path),
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )
    write_fact_jsonl(output_path, verified)
    write_json(Path(output_path).with_suffix(".summary.json"), summary)
    return {
        "output_path": Path(output_path).as_posix(),
        "summary_path": Path(output_path).with_suffix(".summary.json").as_posix(),
        "summary": summary,
    }
