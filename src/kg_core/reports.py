from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def count_by_key(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        counter[str(record.get(key, ""))] += 1
    return dict(counter)
