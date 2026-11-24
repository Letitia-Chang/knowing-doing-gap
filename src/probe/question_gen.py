"""Question-set generation for the hidden-state probe extension (pure Python, no GPU).

Arithmetic questions follow the 13 problem families and sampling shares of the
paper's Appendix A (Table 2 / Algorithms 1-13), re-implemented here from that
description. Factual questions are sampled from records supplied by the caller
(``build_questions.py`` feeds TriviaQA), so this module has no network or
dataset dependency and is fully unit-testable.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graders import normalize_text  # noqa: E402
from tools import calculator  # noqa: E402

# family -> (share in percent, difficulty group). Shares sum to 100 (paper Table 2).
FAMILIES: dict[str, tuple[int, str]] = {
    "single_step": (8, "easy"),
    "two_step": (5, "easy"),
    "small_modulo": (5, "easy"),
    "negative_subtraction": (7, "larger_short"),
    "four_digit_addsub": (6, "larger_short"),
    "two_digit_mult": (9, "larger_short"),
    "three_by_two_mult": (9, "larger_short"),
    "three_by_three_mult": (7, "larger_short"),
    "precedence_chain": (12, "multi_step"),
    "one_digit_chain": (11, "multi_step"),
    "small_chain": (10, "multi_step"),
    "parenthesized": (6, "multi_step"),
    "mult_chain": (5, "multi_step"),
}


def _chain(rng: random.Random, operands: list[int], ops: list[str]) -> str:
    parts = [str(operands[0])]
    for op, val in zip(ops, operands[1:]):
        parts += [op, str(val)]
    return " ".join(parts)


def _single_step(r: random.Random) -> str:
    return f"{r.randint(1, 99)} {r.choice('+-')} {r.randint(1, 99)}"


def _two_step(r: random.Random) -> str:
    a, b, c = (r.randint(1, 99) for _ in range(3))
    return f"{a} + {b} - {c}" if r.random() < 0.5 else f"{a} - {b} + {c}"


def _small_modulo(r: random.Random) -> str:
    return f"{r.randint(100, 999)} % {r.randint(3, 19)}"


def _negative_subtraction(r: random.Random) -> str:
    if r.random() < 0.55:
        a = r.randint(100, 500)
        b = r.randint(a + 10, a + 250)
    else:
        a = r.randint(1000, 5000)
        b = r.randint(a + 100, a + 3000)
    return f"{a} - {b}"


def _four_digit_addsub(r: random.Random) -> str:
    a, b = r.randint(1000, 9999), r.randint(1000, 9999)
    return f"{a} + {b}" if r.random() < 0.6 else f"{a} - {b}"


def _two_digit_mult(r: random.Random) -> str:
    lo, hi = (15, 50) if r.random() < 0.45 else (30, 99)
    return f"{r.randint(lo, hi)} * {r.randint(lo, hi)}"


def _three_by_two_mult(r: random.Random) -> str:
    return f"{r.randint(100, 999)} * {r.randint(10, 99)}"


def _three_by_three_mult(r: random.Random) -> str:
    return f"{r.randint(100, 999)} * {r.randint(100, 999)}"


def _precedence_chain(r: random.Random) -> str:
    return _chain(r, [r.randint(10, 999) for _ in range(5)], [r.choice(["+", "-", "*"]) for _ in range(4)])


def _one_digit_chain(r: random.Random) -> str:
    n = r.randint(16, 22) if r.random() < 0.4 else r.randint(29, 39)
    ops = ["+" if r.random() < 0.53 else "-" for _ in range(n - 1)]
    return _chain(r, [r.randint(1, 9) for _ in range(n)], ops)


def _small_chain(r: random.Random) -> str:
    n = r.randint(21, 27)
    return _chain(r, [r.randint(1, 30) for _ in range(n)], [r.choice("+-") for _ in range(n - 1)])


def _parenthesized(r: random.Random) -> str:
    a, b, c, d = (r.randint(10, 99) for _ in range(4))
    return f"({a} + {b}) * ({c} - {d})"


def _mult_chain(r: random.Random) -> str:
    a, b, c, d, e = (r.randint(10, 99) for _ in range(5))
    return f"{a} + {b} * {c} - {d} * {e}"


GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "single_step": _single_step,
    "two_step": _two_step,
    "small_modulo": _small_modulo,
    "negative_subtraction": _negative_subtraction,
    "four_digit_addsub": _four_digit_addsub,
    "two_digit_mult": _two_digit_mult,
    "three_by_two_mult": _three_by_two_mult,
    "three_by_three_mult": _three_by_three_mult,
    "precedence_chain": _precedence_chain,
    "one_digit_chain": _one_digit_chain,
    "small_chain": _small_chain,
    "parenthesized": _parenthesized,
    "mult_chain": _mult_chain,
}
assert set(GENERATORS) == set(FAMILIES) and sum(s for s, _ in FAMILIES.values()) == 100


def expression_to_question(expr: str) -> str:
    return f"What is {expr.replace('%', 'mod')}?"


def allocate_quotas(total: int, shares: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of ``total`` items across families."""
    denom = sum(shares.values())
    raw = {k: total * v / denom for k, v in shares.items()}
    quotas = {k: int(x) for k, x in raw.items()}
    leftover = total - sum(quotas.values())
    for k in sorted(raw, key=lambda k: (raw[k] - quotas[k], k), reverse=True)[:leftover]:
        quotas[k] += 1
    return quotas


def generate_arithmetic(n: int, seed: int, seen: set[str] | None = None) -> list[dict[str, Any]]:
    """Generate ``n`` unique arithmetic questions with gold answers.

    Gold answers are computed with the repo's AST-whitelisted calculator, so the
    question text and the answer cannot drift apart.
    """
    seen = set(seen or ())
    rng = random.Random(seed)
    quotas = allocate_quotas(n, {k: s for k, (s, _) in FAMILIES.items()})
    out: list[dict[str, Any]] = []
    for family, quota in quotas.items():
        made = 0
        attempts = 0
        while made < quota:
            attempts += 1
            if attempts > quota * 200 + 1000:
                raise RuntimeError(f"Could not generate {quota} unique '{family}' questions")
            expr = GENERATORS[family](rng)
            question = expression_to_question(expr)
            if question in seen:
                continue
            seen.add(question)
            out.append({
                "category": "arithmetic",
                "difficulty": FAMILIES[family][1],
                "family": family,
                "question": question,
                "reference_answer": calculator(expr),
                "source": "generated",
            })
            made += 1
    return out


_MIN_ALIAS_CHARS = 3


def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize_text(text).split() if len(t) >= _MIN_ALIAS_CHARS}


def clean_aliases(value: str, aliases: list[str], limit: int = 10) -> list[str]:
    """Return [canonical value, *safe aliases], or [] if the canonical value is ungradeable.

    TriviaQA aliases include Wikipedia redirect titles and misspellings that are unrelated to
    the answer (e.g. "Duckling" for "Duck"), which would grade wrong answers as correct. An alias
    is therefore kept only if it shares at least one content word with the canonical value
    (so "Sir William Lyons" survives for "William Lyons", but "Quack (sound)" does not).
    """
    if len(normalize_text(value)) < _MIN_ALIAS_CHARS:
        return []
    anchor = _content_tokens(value)
    seen = {normalize_text(value)}
    out = [value]
    for cand in aliases:
        norm = normalize_text(cand)
        if len(norm) >= _MIN_ALIAS_CHARS and norm not in seen and anchor & _content_tokens(cand):
            seen.add(norm)
            out.append(cand)
        if len(out) >= limit:
            break
    return out


def select_factual(records: list[dict[str, Any]], n: int, seed: int, seen: set[str] | None = None) -> list[dict[str, Any]]:
    """Pick ``n`` factual questions from TriviaQA-style records.

    Each record needs ``question`` and ``answer`` = {"value": str, "aliases": [str]}.
    Records whose gold answer cannot be graded safely are skipped.
    """
    seen = set(seen or ())
    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    for i in order:
        rec = records[i]
        question = re.sub(r"\s+", " ", rec["question"]).strip()
        if question in seen or len(question) > 240:
            continue
        accepted = clean_aliases(rec["answer"]["value"], rec["answer"].get("aliases", []))
        if not accepted:
            continue
        seen.add(question)
        out.append({
            "category": "factual",
            "difficulty": "triviaqa",
            "question": question,
            "reference_answer": accepted[0],
            "accept": accepted[1:],
            "source": "triviaqa",
            "source_id": rec.get("question_id"),
        })
        if len(out) == n:
            return out
    raise RuntimeError(f"Only {len(out)} usable factual questions found, needed {n}")


def assemble(originals: list[dict[str, Any]], arithmetic: list[dict[str, Any]], factual: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Originals first (IDs/fields untouched), then new items shuffled together under neutral IDs."""
    kept = []
    for q in originals:
        q = dict(q)
        q["source"] = "original30"
        kept.append(q)
    new = [dict(q) for q in arithmetic + factual]
    random.Random(seed).shuffle(new)
    for i, q in enumerate(new, 1):
        q["id"] = f"Q{i:04d}"
    ordered = kept + new
    ids = [q["id"] for q in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question ids after assembly")
    return ordered
