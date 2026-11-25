"""Correction pass: v2 grading, regrading, prompt transfer and direction analysis (no GPU)."""
import numpy as np
import pytest

from directions import direction, split_cosines
from grading_v2 import filter_aliases, match_factual, tokens
from regrade import candidates, regrade_question
from transfer import boot_diff, transfer_scores
from cv import fast_auroc, stratified_folds


@pytest.mark.parametrize("answer,gold", [
    ("wind", "Winds"), ("Ducks", "Duck"), ("Cymbal", "Cymbals"), ("Twenty-One", "Twenty One"), ("Danaë", "DANAE"),
    ("Shylock's", "Shylock"), ("First Anniversary", "1st"), ("I Don't Like Mondays", "I Don’t Like Mondays"),
    ("Joe Frazier", "(Joe) FRAZIER"), ("Frazier", "(Joe) FRAZIER"), ("Tiffany and Co", "Tiffany & Co."),
    ("approximately 2,850 km", "2850"), ("Python 3.14.", "3.14"), ("Garonne River", "The Garonne River"),
])
def test_v2_accepts_surface_variants(answer, gold):
    assert match_factual(answer, [gold]) == gold


@pytest.mark.parametrize("answer,gold", [
    ("Duckling", "Duck"), ("A phone", "one"), ("Paul", "Au"), ("10", "One"), ("South Park", "The Simpsons"),
    ("York", "New York"), ("M1", "M11"), ("Enoch", "JOHN"), ("3.1", "3.14"), ("", "Paris"),
])
def test_v2_rejects_different_answers(answer, gold):
    assert match_factual(answer, [gold]) is None


def test_alias_filter_keeps_variants_and_drops_unrelated():
    assert filter_aliases("Duck", ["Quack (sound)", "Duckling", "True ducks"]) == ["True ducks"]
    assert filter_aliases("William Lyons", ["Sir William Lyons", "Bill"]) == ["Sir William Lyons"]


def test_tokens_empty_and_articles():
    assert tokens("the") == [] and tokens("The Cat's hats") == ["cat", "hat"]


def test_regrade_is_symmetric_and_uses_extra_accepts():
    q = {"id": "Q1", "category": "factual", "reference_answer": "Winds", "accept": []}
    flags, via = regrade_question(q, ["wind", "rain", "gale"], candidates(q, {}, {"Q1": ["gale"]}))
    assert flags == [True, False, True] and via == ["Winds", None, "gale"]
    a = {"id": "A1", "category": "arithmetic", "reference_answer": "-7"}
    assert regrade_question(a, ["-7", "7"], candidates(a, {}, {}))[0] == [True, False]


def _data(n=120, hidden=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(3, n, hidden)).astype(np.float32)
    X[1, :, 0] += 3 * (y - 0.5)
    return X, y, rng


def test_transfer_survives_a_constant_prompt_offset_when_recentred():
    X, y, rng = _data()
    XB = X + 5.0 + rng.normal(scale=0.05, size=X.shape).astype(np.float32)
    s = transfer_scores(X, XB, y, stratified_folds(y, 3, 1, 0), c_grid=(0.1,))
    assert fast_auroc(y, s["A"]) > 0.9 and fast_auroc(y, s["B_recentred"]) > 0.9
    assert np.abs(s["B_recentred"] - s["A"]).mean() < 0.3


def test_transfer_fails_when_the_signal_is_absent_in_the_other_prompt():
    X, y, rng = _data()
    XB = rng.normal(size=X.shape).astype(np.float32)
    s = transfer_scores(X, XB, y, stratified_folds(y, 3, 1, 0), c_grid=(0.1,))
    assert fast_auroc(y, s["A"]) > 0.9 and 0.3 < fast_auroc(y, s["B_recentred"]) < 0.7


def test_direction_cosines_separate_same_from_orthogonal_signals():
    rng = np.random.default_rng(0)
    need, act = rng.integers(0, 2, 400), rng.integers(0, 2, 400)
    Z = rng.normal(size=(400, 10))
    Z[:, 0] += 4 * need
    Z[:, 1] += 4 * act
    for kind in ("logreg", "meandiff"):
        x, cc, aa = np.mean([split_cosines(Z, need, act, kind, 0.1, s) for s in range(4)], axis=0)
        assert abs(x) < 0.3 and cc > 0.8 and aa > 0.8
        assert np.mean([split_cosines(Z, need, need, kind, 0.1, s)[0] for s in range(4)]) > 0.8
    assert np.linalg.norm(direction(Z, need, "logreg", 0.1)) == pytest.approx(1.0)


def test_boot_diff_sign_and_interval():
    x = np.r_[np.ones(50), np.zeros(50)] + np.random.default_rng(0).normal(scale=0.1, size=100)
    r = boot_diff(x, np.arange(100) < 50, n_boot=200)
    assert r["lo"] < r["diff"] < r["hi"] and r["lo"] > 0.8


def test_reject_rule_overrides_a_match():
    q = {"id": "Q0941", "category": "factual", "reference_answer": "ELECTRIC RAZOR", "accept": []}
    flags, _ = regrade_question(q, ["Schick Safety Razor", "Electric razor"], ["ELECTRIC RAZOR", "Razor"], ["Safety razor"])
    assert flags == [False, True]


def test_label_decisions_refer_to_real_factual_questions_and_do_not_conflict():
    import json
    from pathlib import Path

    data = Path(__file__).resolve().parent.parent / "data"
    dec = json.loads((data / "label_decisions.json").read_text())
    cat = {q["id"]: q["category"] for q in json.loads((data / "questions_probe.json").read_text())["questions"]}
    groups = [set(dec[k]) for k in ("exclude", "accept_extra", "reject_if_contains")]
    assert all(cat[i] == "factual" for g in groups for i in g)
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
