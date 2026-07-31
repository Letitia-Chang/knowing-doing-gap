from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    return data["questions"] if isinstance(data, dict) and "questions" in data else data


def apply_limit(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None else items[:limit]
