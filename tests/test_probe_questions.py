import json
import re
from collections import Counter
from pathlib import Path

import pytest

from graders import extract_last_number, grade_factual
from question_gen import FAMILIES, allocate_quotas, assemble, clean_aliases, generate_arithmetic, select_factual

DATA = Path(__file__).resolve().parent.parent / "data"


def eval_question(question: str) -> int:
    expr = re.fullmatch(r"What is (.+)\?", question).group(1).replace(" mod ", " % ")
    return eval(expr, {"__builtins__": {}})  # generated, trusted strings only


def test_family_shares_sum_to_100():
    assert sum(s for s, _ in FAMILIES.values()) == 100


@pytest.mark.parametrize("total", [0, 1, 7, 100, 485])
def test_allocate_quotas_sums_to_total(total):
    q = allocate_quotas(total, {k: s for k, (s, _) in FAMILIES.items()})
    assert sum(q.values()) == total
    assert all(v >= 0 for v in q.values())


def test_generate_arithmetic_gold_matches_independent_eval():
    qs = generate_arithmetic(300, seed=1)
    assert len(qs) == 300
    assert all(int(q["reference_answer"]) == eval_question(q["question"]) for q in qs)


def test_generate_arithmetic_unique_deterministic_and_respects_seen():
    a = generate_arithmetic(120, seed=3)
    b = generate_arithmetic(120, seed=3)
    assert a == b
    assert len({q["question"] for q in a}) == 120
    assert a != generate_arithmetic(120, seed=4)
    seen = {q["question"] for q in a}
    assert not seen & {q["question"] for q in generate_arithmetic(120, seed=3, seen=seen)}


def test_generate_arithmetic_family_counts_follow_quotas():
    qs = generate_arithmetic(200, seed=0)
    expected = allocate_quotas(200, {k: s for k, (s, _) in FAMILIES.items()})
    assert Counter(q["family"] for q in qs) == Counter(expected)


def test_clean_aliases_drops_unrelated_and_short():
    assert clean_aliases("Duck", ["Quack (sound)", "Duckling", "Ducks"]) == ["Duck"]
    assert clean_aliases("William Lyons", ["Sir William Lyons", "Bill"]) == ["William Lyons", "Sir William Lyons"]
    assert clean_aliases("Au", ["gold"]) == []  # canonical value too short to grade safely


def _records(n):
    return [{"question": f"Question number {i}?", "question_id": f"tc_{i}", "answer": {"value": f"Answer{i}", "aliases": []}} for i in range(n)]


def test_select_factual_deterministic_dedup_and_error_when_short():
    recs = _records(50) + [{"question": "Bad?", "answer": {"value": "ab", "aliases": []}}]
    a = select_factual(recs, 20, seed=0)
    assert a == select_factual(recs, 20, seed=0)
    assert len({q["question"] for q in a}) == 20 and all(q["category"] == "factual" for q in a)
    assert not any(q["question"] == "Bad?" for q in select_factual(recs, 50, seed=0))
    with pytest.raises(RuntimeError):
        select_factual(recs, 60, seed=0)


def test_assemble_keeps_originals_and_uses_neutral_ids():
    orig = [{"id": "A01", "category": "arithmetic", "question": "What is 2 + 2?", "reference_answer": "4"}]
    out = assemble(orig, generate_arithmetic(5, 0), select_factual(_records(10), 5, 0), seed=0)
    assert out[0]["id"] == "A01" and out[0]["source"] == "original30"
    assert all(re.fullmatch(r"Q\d{4}", q["id"]) for q in out[1:])
    assert len({q["id"] for q in out}) == len(out) == 11


def test_committed_probe_set_integrity():
    path = DATA / "questions_probe.json"
    qs = json.loads(path.read_text())["questions"]
    originals = json.loads((DATA / "questions.json").read_text())["questions"]
    assert Counter(q["category"] for q in qs) == Counter({"arithmetic": 500, "factual": 500})
    assert len({q["id"] for q in qs}) == len(qs) == len({q["question"] for q in qs})
    for o, q in zip(originals, qs[:30]):  # originals are kept verbatim, in order, as a subset
        assert all(q[k] == v for k, v in o.items())
    for q in qs:
        if q["category"] == "arithmetic" and q["source"] == "generated":
            assert int(q["reference_answer"]) == eval_question(q["question"])
        if q["category"] == "factual":
            assert q["reference_answer"]


def test_grader_unicode_minus_and_word_boundary():
    assert extract_last_number("The result is −755") == -755
    assert grade_factual("The answer is 21", "one") is False
    assert grade_factual("A phone", "one") is False
    assert grade_factual("Paul", "Au") is False
    assert grade_factual("gold (Au)", "Au", ["gold (au)"]) is True
    assert grade_factual("approximately 2,850 km", "2850") is True
