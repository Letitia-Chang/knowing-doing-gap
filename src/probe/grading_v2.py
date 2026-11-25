"""Corrected factual grader (v2) for the probe extension. Applied to ALL saved answers, right and wrong.

Why: the first grader rejected correct answers over plurals, accents, hyphens, possessives and
ordinals ("Wind" vs "Winds", "Danaë" vs "Danae"), and the first alias filter dropped valid aliases
("Ducks", "twenty-one"). src/graders.py is left unchanged, so the behavioral results are unaffected.
"""
from __future__ import annotations

import re
import unicodedata

_ORDINALS = {w: f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}" for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split(), 1)}
_NUMBERS = {w: str(i) for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split(), 1)}
_ARTICLES = {"a", "an", "the"}


def _stem(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def tokens(text: str) -> list[str]:
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("&", " and ")
    s = re.sub(r"'s\b", "", s).replace("'", "")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)                 # 2,850 -> 2850
    s = re.sub(r"[^\w\s.]|_", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)            # keep decimal points only
    out = []
    for t in s.split():
        t = _ORDINALS.get(t, _NUMBERS.get(t, t))
        if t not in _ARTICLES:
            out.append(_stem(t))
    return out


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if len(t) >= 2}


def variants(candidate: str) -> list[list[str]]:
    """The candidate as written, and with bracketed text removed: "(Rt Rev) Paul Bayes" -> "Paul Bayes"."""
    out = []
    for c in (candidate, re.sub(r"\([^)]*\)", " ", candidate)):
        t = tokens(c)
        if t and t not in out:
            out.append(t)
    return out


def match_factual(answer: str, candidates: list[str]) -> str | None:
    """Return the first candidate whose token sequence appears contiguously in the answer, else None."""
    hay = " " + " ".join(tokens(answer)) + " "
    for c in candidates:
        if any(" " + " ".join(v) + " " in hay for v in variants(c)):
            return c
    return None


def filter_aliases(value: str, aliases: list[str]) -> list[str]:
    """Keep aliases sharing a (normalized) content word with the canonical value; no length cap."""
    anchor, seen, out = content_tokens(value), {tuple(tokens(value))}, []
    for a in aliases:
        key = tuple(tokens(a))
        if key and key not in seen and anchor & content_tokens(a):
            seen.add(key)
            out.append(a)
    return out
