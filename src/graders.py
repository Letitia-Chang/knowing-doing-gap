from __future__ import annotations

import math
import re
import string
from typing import Any


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = s.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    s = re.sub(r"\s+", " ", s)
    return s


_NON_ANSWER_NUMBER = re.compile(r"\bin base\s+\d+\b|\bbase\s+\d+\b", flags=re.I)


def extract_last_number(text: str) -> float | None:
    # Strip annotations like "in base 10" so they aren't mistaken for the
    # answer itself (e.g. "63 in base 10" should grade as 63, not 10).
    text = _NON_ANSWER_NUMBER.sub("", text.replace("−", "-"))  # U+2212 minus sign
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def grade_arithmetic(answer: str, reference: str, tolerance: float = 1e-6) -> bool:
    got = extract_last_number(answer)
    exp = extract_last_number(reference)
    if got is None or exp is None:
        return False
    return math.isclose(got, exp, rel_tol=0, abs_tol=tolerance)


def grade_factual(answer: str, reference: str, accept: list[str] | None = None) -> bool:
    answer_n = normalize_text(answer)
    candidates = [reference] + (accept or [])
    # Whole-token match so short references ("au", "one") don't hit inside other words.
    return any(
        re.search(rf"(?<!\w){re.escape(n)}(?!\w)", answer_n)
        for n in (normalize_text(c) for c in candidates)
        if n
    )


def grade_answer(question: dict[str, Any], answer: str) -> bool:
    if question["category"] == "arithmetic":
        return grade_arithmetic(answer, str(question["reference_answer"]))
    return grade_factual(answer, str(question["reference_answer"]), question.get("accept"))
