import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from cv import balanced_accuracy, fast_auroc, metrics_with_ci, run_nested, stratified_folds
from knowing_doing import analyse, bootstrap_rate, calibration_table, cell_name


def synthetic(n=120, layers=3, hidden=6, signal_layer=1, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    X = rng.normal(size=(layers, n, hidden)).astype(np.float32)
    X[signal_layer, :, 0] += 3.0 * (y - 0.5)  # only one layer carries the label
    return X, y


def test_fast_auroc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)  # rounding creates ties
    assert fast_auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(fast_auroc(np.ones(5), np.arange(5)))


def test_balanced_accuracy():
    y = np.array([1, 1, 1, 1, 0, 0])
    assert balanced_accuracy(y, np.array([1, 1, 0, 0, 0, 0])) == pytest.approx((0.5 + 1.0) / 2)


def test_folds_partition_every_repeat_and_stay_stratified():
    y = np.array([0] * 80 + [1] * 20)
    folds = stratified_folds(y, n_splits=5, repeats=3, seed=0)
    assert len(folds) == 15
    for r in range(3):
        tests = np.concatenate([te for rr, _, _, te in folds if rr == r])
        assert sorted(tests) == list(range(100))  # each sample tested exactly once per repeat
    for _, _, tr, te in folds:
        assert not set(tr) & set(te)
        assert y[te].sum() == 4  # 20% positives preserved in every test fold


def test_nested_selection_never_sees_outer_test_labels():
    """Flip the labels of one outer test fold: its layer selection and predictions must not change."""
    X, y = synthetic()
    folds = stratified_folds(y, 3, 1, seed=0)
    kw = dict(c_grid=(0.01, 1.0), inner_splits=3, seed=0, n_jobs=1)
    base = run_nested(X, y, folds, **kw)
    r, k, tr, te = folds[0]
    y2 = y.copy()
    y2[te] = 1 - y2[te]
    changed = run_nested(X, y2, folds, **kw)
    assert np.array_equal(base.inner_auc[r, k], changed.inner_auc[r, k])
    assert base.selected_layer[r, k] == changed.selected_layer[r, k]
    assert np.allclose(base.prob[r][:, te], changed.prob[r][:, te])


def test_nested_finds_signal_layer_and_nulls_stay_near_chance():
    X, y = synthetic()
    folds = stratified_folds(y, 5, 2, seed=0)
    kw = dict(c_grid=(0.01, 1.0), inner_splits=3, seed=0, n_jobs=1)
    res = run_nested(X, y, folds, **kw)
    assert (res.selected_layer == 1).mean() >= 0.8
    assert fast_auroc(y, res.nested_prob().mean(0)) > 0.9
    y_perm = np.random.default_rng(1).permutation(y)
    null = run_nested(X, y_perm, stratified_folds(y_perm, 5, 1, seed=0), **kw)
    assert fast_auroc(y_perm, null.nested_prob()[0]) < 0.7


def test_nested_prob_uses_layer_selected_for_each_samples_fold():
    X, y = synthetic(n=60)
    folds = stratified_folds(y, 3, 1, seed=0)
    res = run_nested(X, y, folds, c_grid=(0.1,), inner_splits=3, seed=0, n_jobs=1)
    for i in range(len(y)):
        assert res.nested_prob()[0, i] == res.prob[0, res.selected_layer[0, res.fold_of[0, i]], i]


def test_metrics_with_ci_perfect_and_random():
    y = np.array([0, 1] * 100)
    perfect = metrics_with_ci(y, y.astype(float), n_boot=50)
    assert perfect["auroc"] == 1.0 and perfect["bal_acc"] == 1.0 and perfect["auroc_lo"] > 0.95
    rnd = metrics_with_ci(y, np.random.default_rng(0).random(200), n_boot=200)
    assert rnd["auroc_lo"] < 0.5 < rnd["auroc_hi"]


def test_cell_name_four_cells():
    assert cell_name(True, True) == "knows_does" and cell_name(True, False) == "knows_doesnt"
    assert cell_name(False, True) == "unneeded_does" and cell_name(False, False) == "neither"


def test_analyse_counts_and_overuse_subset():
    df = pd.DataFrame({
        "id": list("abcdefgh"),
        "category": ["arithmetic"] * 4 + ["factual"] * 4,
        "tool_needed": [1, 1, 0, 0, 1, 0, 0, 1],
        "p_needed_oof": [0.9, 0.8, 0.2, 0.1, 0.3, 0.6, 0.4, 0.7],
        "tool_called": [True, False, True, True, False, True, False, True],
    })
    s, eight, cal = analyse(df)
    assert s["four_cell_counts"] == {"knows_does": 3, "knows_doesnt": 1, "unneeded_does": 2, "neither": 2}
    assert s["over_use"]["n"] == 3  # c, d, f: not needed but called
    assert s["over_use"]["probe_says_not_needed"]["rate"] == pytest.approx(2 / 3)  # c, d yes; f no
    assert s["under_use"]["n"] == 2 and eight.n.iloc[:8].sum() >= 8
    assert eight[eight.category == "all"].n.sum() == 8
    assert cal.n.sum() == 8


def test_calibration_and_bootstrap_helpers():
    p = np.repeat([0.05, 0.95], 100)
    needed = np.repeat([0.0, 1.0], 100)
    _, ece = calibration_table(p, needed, needed)
    assert ece == pytest.approx(0.05)
    assert bootstrap_rate(np.array([]))["n"] == 0
    r = bootstrap_rate(np.array([1, 1, 0, 0]))
    assert r["rate"] == 0.5 and r["lo"] <= 0.5 <= r["hi"]
