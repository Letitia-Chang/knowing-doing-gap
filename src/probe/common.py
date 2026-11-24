"""Shared, GPU-free logic for the hidden-state probe extension.

Prompts are imported from the existing behavioural scripts so the probe study
sees exactly the same decision context as the original experiment.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from label_necessity import SYSTEM as NECESSITY_SYSTEM  # noqa: E402
from label_necessity import build_prompt as necessity_user_prompt  # noqa: E402
from run_action import BASELINE_SYSTEM, INTERVENTION_SYSTEM, extract_json  # noqa: E402

ACTION_SYSTEMS = {"baseline": BASELINE_SYSTEM, "intervention": INTERVENTION_SYSTEM}

# Necessity thresholds on no-tool accuracy: tool_needed = accuracy < threshold.
# 1.0 is the paper's definition (needed if ANY of N runs is wrong); the others are sensitivity checks.
PRIMARY_THRESHOLD = 1.0
SENSITIVITY_THRESHOLDS = (1.0, 0.8, 0.5)


def threshold_column(threshold: float) -> str:
    return "tool_needed" if threshold == PRIMARY_THRESHOLD else f"tool_needed_acc{int(round(threshold * 100))}"


def action_user_prompt(q: dict[str, Any]) -> str:
    """Identical to the user message built by run_action.choose_action."""
    return f"Question ID: {q['id']}\nQuestion: {q['question']}"


def necessity_labels(correct_count: int, n: int) -> dict[str, Any]:
    acc = correct_count / n
    row: dict[str, Any] = {"necessity_correct_count": correct_count, "necessity_n": n, "necessity_accuracy": acc}
    for t in SENSITIVITY_THRESHOLDS:
        row[threshold_column(t)] = acc < t
    return row


def parse_tool_decision(action_obj: Any, raw: str) -> dict[str, Any]:
    """Classify a first-turn model output. Mirrors the branching in run_action.main exactly:

    - JSON primitive (e.g. ``63``)             -> direct answer
    - ``{"action":"tool","tool":T,...}`` or ``{"action":"calculator"|"search",...}`` -> tool call
    - ``{"action":"answer","answer":A}``       -> direct answer
    - any other object / unparseable text      -> malformed, counted as NOT calling a tool
    """
    out = {"tool_called": False, "tool": None, "malformed": False, "direct_answer": None}
    if action_obj is not None and not isinstance(action_obj, dict):
        out["direct_answer"] = str(action_obj)
    elif isinstance(action_obj, dict):
        action = action_obj.get("action")
        if action == "tool" or action in {"calculator", "search"}:
            out["tool_called"] = True
            out["tool"] = str(action_obj.get("tool", "")) if action == "tool" else str(action)
        elif action == "answer":
            out["direct_answer"] = str(action_obj.get("answer", ""))
        else:
            out["malformed"] = True
            out["direct_answer"] = raw
    else:
        out["malformed"] = True
        out["direct_answer"] = raw
    return out


def surface_features(q: dict[str, Any]) -> dict[str, float]:
    """Cheap lexical features used by the surface-baseline control (no model involved)."""
    text = q["question"]
    numbers = re.findall(r"\d+", text)
    return {
        "n_chars": float(len(text)),
        "n_words": float(len(text.split())),
        "n_digits": float(sum(c.isdigit() for c in text)),
        "n_numbers": float(len(numbers)),
        "max_number_len": float(max((len(n) for n in numbers), default=0)),
        "n_operators": float(len(re.findall(r"[+\-*/%]|\bmod\b", text))),
        "is_arithmetic": float(q["category"] == "arithmetic"),
    }


def update_run_info(path: str | Path, key: str, value: Any) -> None:
    path = Path(path)
    info = json.loads(path.read_text()) if path.exists() else {}
    info[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
