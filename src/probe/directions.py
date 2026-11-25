"""Necessity vs action directions (adapts paper sections 5.2-5.3) on saved activations (CPU).

Per layer: a probe for tool_needed and a probe for tool_called, with out-of-fold AUROC and MCC, and the
cosine similarity between their weight vectors.

A raw cosine is hard to read here: with ~1,000 samples in 3,584 dimensions each direction is noisy, and
noise alone pushes cosines toward 0, which would look like "orthogonal". So directions are fitted on
disjoint random halves, and the cross-half cosine between necessity and action is reported next to its
ceiling: how well each direction agrees with ITSELF across halves. Directions are compared in one
label-free standardized space. Two estimators: L2 logistic regression and difference of class means.
The layer is not selected; every layer is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import fast_auroc, make_probe  # noqa: E402
from io_utils import load_questions  # noqa: E402

MIN_MINORITY = 40  # skip a (prompt, scope) when either label has fewer minority samples


def direction(X: np.ndarray, y: np.ndarray, kind: str, C: float) -> np.ndarray:
    if kind == "logreg":
        w = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(X, y).coef_[0]
    else:
        w = X[y == 1].mean(0) - X[y == 0].mean(0)
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def split_cosines(Z, need, act, kind, C, seed) -> tuple[float, float, float]:
    """(necessity-action, necessity-necessity, action-action) cosines across two disjoint halves."""
    strat = need * 2 + act
    if np.bincount(strat, minlength=4).min() < 2:
        strat = act
    a, b = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=seed).split(Z, strat))
    cA, cB, aA, aB = (direction(Z[i], lab[i], kind, C) for lab in (need, act) for i in (a, b))
    return (cA @ aB + cB @ aA) / 2, cA @ cB, aA @ aB


def oof(X, y, C, folds=5, seed=0) -> tuple[float, float]:
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        p[te] = make_probe(C, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return fast_auroc(y, p), float(matthews_corrcoef(y, p >= 0.5))


def layer_task(Xl, need, act, C, splits, seed) -> dict:
    X = Xl.astype(np.float32)
    sd = X.std(0)
    Z = (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)
    row: dict = {}
    row["need_auroc"], row["need_mcc"] = oof(X, need, C, seed=seed)
    row["action_auroc"], row["action_mcc"] = oof(X, act, C, seed=seed)
    rng = np.random.default_rng(seed + 77)
    for kind in ("logreg", "meandiff"):
        v = np.array([split_cosines(Z, need, act, kind, C, seed + s) for s in range(splits)])
        x, cc, aa = v.mean(0)
        row[f"{kind}_cos_need_action"], row[f"{kind}_cos_need_need"], row[f"{kind}_cos_action_action"] = x, cc, aa
        row[f"{kind}_cos_need_action_lo"], row[f"{kind}_cos_need_action_hi"] = np.percentile(v[:, 0], [2.5, 97.5])
        row[f"{kind}_cos_disattenuated"] = x / np.sqrt(cc * aa) if cc > 0.05 and aa > 0.05 else np.nan
        row[f"{kind}_cos_shuffled_action"] = np.mean([split_cosines(Z, need, rng.permutation(act), kind, C, seed + s)[0] for s in range(min(splits, 5))])
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/probe_corrected")
    p.add_argument("--actdir", default="results/probe")
    p.add_argument("--C", type=float, default=1e-3)
    p.add_argument("--splits", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    out, act_dir = Path(args.outdir), Path(args.actdir)

    qs = load_questions(out / "questions_probe_labeled.json")
    ids = [q["id"] for q in qs]
    need = np.array([int(q["tool_needed"]) for q in qs])
    cat = np.array([q["category"] for q in qs])
    rows, meta = [], []
    for prompt, f in (("baseline", "activations.npz"), ("intervention", "activations_intervention.npz")):
        z = np.load(act_dir / f, allow_pickle=False)
        pos = {str(i): k for k, i in enumerate(z["ids"])}
        X = z["last_token"][:, [pos[i] for i in ids], :]
        a = {r["id"]: int(r["tool_called"]) for r in json.loads((out / f"action_local_{prompt}.json").read_text())}
        act = np.array([a[i] for i in ids])
        for scope in ("all", "arithmetic", "factual"):
            m = np.ones(len(ids), bool) if scope == "all" else cat == scope
            n, c = need[m], act[m]
            minority = min(n.sum(), (1 - n).sum(), c.sum(), (1 - c).sum())
            info = {"prompt": prompt, "scope": scope, "n": int(m.sum()), "n_needed": int(n.sum()), "n_called": int(c.sum()), "skipped": bool(minority < MIN_MINORITY)}
            if not info["skipped"]:
                info["label_correlation_phi"] = float(matthews_corrcoef(n, c))
            meta.append(info)
            if info["skipped"]:
                continue
            for l, r in enumerate(Parallel(n_jobs=args.n_jobs)(delayed(layer_task)(X[l][m], n, c, args.C, args.splits, args.seed) for l in range(X.shape[0]))):
                rows.append({"prompt": prompt, "scope": scope, "layer": l, **r})
    d = pd.DataFrame(rows)
    d.to_csv(out / "directions.csv", index=False)
    lines = ["### Necessity and action directions (last prompt token)", "",
             f"Cosines are between directions fitted on disjoint halves, averaged over {args.splits} splits. 'Ceiling' = agreement of each direction with itself across halves. 'Corrected' = cosine / sqrt(product of ceilings).", "",
             "| Prompt | Scope | n (needed / called) | phi(needed, called) | Layer | Necessity AUROC (MCC) | Action AUROC (MCC) | LR cosine | LR ceiling need / action | LR corrected | Mean-diff cosine | Mean-diff corrected | Shuffled-action LR cosine |",
             "|---|---|---|--:|--:|---|---|---|---|--:|--:|--:|--:|"]
    for i in meta:
        if i["skipped"]:
            lines.append(f"| {i['prompt']} | {i['scope']} | {i['n']} ({i['n_needed']} / {i['n_called']}) | skipped: a label has fewer than {MIN_MINORITY} minority samples | | | | | | | | | |")
            continue
        g = d[(d.prompt == i["prompt"]) & (d.scope == i["scope"])].set_index("layer")
        last = int(g.index.max())
        for l in sorted({7, 14, 21, 24, last} & set(g.index)):
            r = g.loc[l]
            lines.append(f"| {i['prompt']} | {i['scope']} | {i['n']} ({i['n_needed']} / {i['n_called']}) | {i['label_correlation_phi']:.2f} | {l} | {r.need_auroc:.3f} ({r.need_mcc:.2f}) | {r.action_auroc:.3f} ({r.action_mcc:.2f}) | "
                         f"{r.logreg_cos_need_action:.2f} [{r.logreg_cos_need_action_lo:.2f}, {r.logreg_cos_need_action_hi:.2f}] | {r.logreg_cos_need_need:.2f} / {r.logreg_cos_action_action:.2f} | {r.logreg_cos_disattenuated:.2f} | "
                         f"{r.meandiff_cos_need_action:.2f} | {r.meandiff_cos_disattenuated:.2f} | {r.logreg_cos_shuffled_action:.2f} |")
    (out / "directions.json").write_text(json.dumps({"config": vars(args), "cells": meta}, indent=2))
    (out / "directions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
