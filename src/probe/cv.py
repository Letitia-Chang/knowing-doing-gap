"""Cross-validated linear probing with nested layer selection (CPU only; numpy + scikit-learn).

Design (all of it exists to avoid optimistic bias):

* Outer loop: repeated stratified K-fold. Every question gets exactly one out-of-fold (OOF)
  probability per repeat per layer.
* Inner loop (training rows of an outer fold ONLY): stratified inner CV picks the L2 strength C
  for each layer and gives that layer an inner AUROC.
* Nested layer selection: the layer with the best inner AUROC is "the best layer" for that outer
  fold. The OOF probability a question receives comes from the layer chosen without ever seeing
  its fold, so the headline numbers and the Step-6 predictions are not selected on test data.
* The per-layer curve uses the same OOF probabilities (each layer's C is also tuned inside
  training folds). Its peak is still optimistic as a "best layer" estimate; the nested number is
  the honest one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_probe(C: float, seed: int = 0):
    # LogisticRegression's default penalty is L2. class_weight="balanced" because tool_needed is imbalanced.
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed))


def fast_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)  # average ranks handle ties
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    return float((pred[y].mean() + (~pred[~y]).mean()) / 2)


def stratified_folds(y: np.ndarray, n_splits: int, repeats: int, seed: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """[(repeat, fold, train_idx, test_idx)]; each repeat partitions all samples."""
    folds = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            folds.append((r, k, tr, te))
    return folds


def inner_select_C(X: np.ndarray, y: np.ndarray, c_grid: tuple[float, ...], inner_splits: int, seed: int) -> tuple[float, float]:
    """Return (best_inner_auroc, best_C) using only (X, y), i.e. training rows of an outer fold."""
    skf = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    best_auc, best_c = -1.0, c_grid[0]
    for C in c_grid:
        oof = np.zeros(len(y))
        for tr, va in splits:
            oof[va] = make_probe(C, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        auc = fast_auroc(y, oof)
        if auc > best_auc + 1e-12:  # ties -> smaller C (earlier in grid)
            best_auc, best_c = auc, C
    return best_auc, best_c


def _fold_layer_task(Xl: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, c_grid, inner_splits: int, seed: int):
    X32 = Xl.astype(np.float32)
    inner_auc, C = inner_select_C(X32[tr], y[tr], c_grid, inner_splits, seed)
    prob = make_probe(C, seed).fit(X32[tr], y[tr]).predict_proba(X32[te])[:, 1]
    return inner_auc, C, prob


@dataclass
class NestedResult:
    prob: np.ndarray          # (R, L, N) OOF P(tool_needed) for every layer
    inner_auc: np.ndarray     # (R, K, L) inner-CV AUROC used for layer selection
    best_C: np.ndarray        # (R, K, L)
    fold_of: np.ndarray       # (R, N) outer fold index of each sample in each repeat
    y: np.ndarray

    @property
    def selected_layer(self) -> np.ndarray:  # (R, K): chosen from inner (training-only) AUROC
        return np.argmax(self.inner_auc, axis=2)

    def nested_prob(self) -> np.ndarray:
        """(R, N): each sample's OOF probability taken from the layer selected for its outer fold."""
        R, L, N = self.prob.shape
        sel = self.selected_layer
        out = np.zeros((R, N), dtype=np.float32)
        for r in range(R):
            layers = sel[r][self.fold_of[r]]
            out[r] = self.prob[r, layers, np.arange(N)]
        return out


def run_nested(X: np.ndarray, y: np.ndarray, folds, c_grid=(1e-4, 1e-3, 1e-2), inner_splits: int = 3, seed: int = 0, n_jobs: int = 1) -> NestedResult:
    """X: (L, N, H). ``folds`` from :func:`stratified_folds` (or any explicit split list)."""
    from joblib import Parallel, delayed

    y = np.asarray(y).astype(int)
    L, N, _ = X.shape
    R = 1 + max(f[0] for f in folds)
    K = 1 + max(f[1] for f in folds)
    tasks = [(r, k, l) for (r, k, _, _) in folds for l in range(L)]
    fold_by_rk = {(r, k): (tr, te) for r, k, tr, te in folds}
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fold_layer_task)(X[l], y, *fold_by_rk[(r, k)], c_grid, inner_splits, seed + 1000 * r + k) for (r, k, l) in tasks
    )
    prob = np.zeros((R, L, N), dtype=np.float32)
    inner_auc = np.zeros((R, K, L))
    best_C = np.zeros((R, K, L))
    fold_of = np.zeros((R, N), dtype=int)
    for (r, k, l), (iauc, C, p) in zip(tasks, results):
        te = fold_by_rk[(r, k)][1]
        prob[r, l, te] = p
        inner_auc[r, k, l] = iauc
        best_C[r, k, l] = C
        fold_of[r, te] = k
    return NestedResult(prob, inner_auc, best_C, fold_of, y)


def oof_sklearn(S: np.ndarray, y: np.ndarray, folds, model_factory: Callable) -> np.ndarray:
    """(R, N) OOF probabilities for a tabular baseline (surface features)."""
    R = 1 + max(f[0] for f in folds)
    out = np.zeros((R, len(y)))
    for r, k, tr, te in folds:
        out[r, te] = model_factory().fit(S[tr], y[tr]).predict_proba(S[te])[:, 1]
    return out


def surface_models(seed: int = 0) -> dict[str, Callable]:
    return {
        "surface_logreg": lambda: make_probe(1.0, seed),
        "surface_gboost": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.1, class_weight="balanced", random_state=seed),
    }


def metrics_with_ci(y: np.ndarray, prob: np.ndarray, n_boot: int = 1000, seed: int = 0, threshold: float = 0.5) -> dict[str, float]:
    """AUROC and balanced accuracy with percentile-bootstrap 95% CIs over questions.

    ``prob`` should be the per-question probability averaged over CV repeats. Bootstrapping
    questions (not folds) is used because CV folds share training data and their scores are not
    independent, which makes fold-to-fold standard errors too small.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    out = {"auroc": fast_auroc(y, prob), "bal_acc": balanced_accuracy(y, prob >= threshold)}
    aucs, bas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs.append(fast_auroc(yb, prob[idx]))
        bas.append(balanced_accuracy(yb, prob[idx] >= threshold))
    for name, vals in (("auroc", aucs), ("bal_acc", bas)):
        out[f"{name}_lo"], out[f"{name}_hi"] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))
    return out
