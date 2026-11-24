"""Stage 5: train/evaluate probes for ``tool_needed`` on saved activations (CPU).

Outputs (in --outdir, suffixed by --tag):
  probe_layer_curve{tag}.csv  per-layer OOF AUROC / balanced accuracy (+bootstrap CIs) with shuffled-label null
  probe_nested{tag}.json      nested-CV headline numbers, layer-selection counts, surface-feature baselines, nulls
  oof_{scope}_{feature}{tag}.csv   per-question OOF P(tool_needed) from the nested pipeline (used by Step 6)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SENSITIVITY_THRESHOLDS, surface_features, threshold_column  # noqa: E402
from cv import fast_auroc, metrics_with_ci, oof_sklearn, run_nested, stratified_folds, surface_models  # noqa: E402
from io_utils import apply_limit, load_questions  # noqa: E402

MIN_CLASS_COUNT = 15  # skip a scope if its minority class is smaller than this


def parse_grid(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def surface_matrix(questions: list[dict]) -> tuple[np.ndarray, list[str]]:
    feats = [surface_features(q) for q in questions]
    names = list(feats[0])
    return np.array([[f[n] for n in names] for f in feats], dtype=float), names


def evaluate_scope(X, y, S, S_names, args, scope: str, feature: str) -> tuple[list[dict], dict, np.ndarray]:
    """Run the full battery on one (scope, feature). Returns (curve rows, summary dict, nested mean OOF prob)."""
    L = X.shape[0]
    folds = stratified_folds(y, args.folds, args.repeats, args.seed)
    kw = dict(c_grid=parse_grid(args.c_grid), inner_splits=args.inner_folds, seed=args.seed, n_jobs=args.n_jobs)
    res = run_nested(X, y, folds, **kw)

    # Shuffled-label null: identical pipeline, labels permuted (one permutation per null repeat).
    null_layer_auc = np.zeros((args.null_repeats, L))
    null_nested_auc, null_nested_bacc = [], []
    for j in range(args.null_repeats):
        y_perm = np.random.default_rng(args.seed + 9000 + j).permutation(y)
        res_null = run_nested(X, y_perm, stratified_folds(y_perm, args.folds, 1, args.seed + 500 + j), **{**kw, "seed": args.seed + 500 + j})
        null_layer_auc[j] = [fast_auroc(y_perm, res_null.prob[0, l]) for l in range(L)]
        pn = res_null.nested_prob()[0]
        null_nested_auc.append(fast_auroc(y_perm, pn))
        null_nested_bacc.append(metrics_with_ci(y_perm, pn, n_boot=1)["bal_acc"])

    rows = []
    for l in range(L):
        m = metrics_with_ci(y, res.prob[:, l, :].mean(0), n_boot=args.n_boot, seed=args.seed)
        rows.append({"scope": scope, "feature": feature, "layer": l, **m,
                     "null_auroc_mean": float(null_layer_auc[:, l].mean())})
    null_p95 = float(np.percentile(null_layer_auc, 95))
    for r in rows:
        r["null_auroc_p95_all_layers"] = null_p95

    nested = res.nested_prob()
    nested_mean = nested.mean(0)
    summary = {
        "scope": scope, "feature": feature, "n": int(len(y)), "n_needed": int(y.sum()), "needed_rate": float(y.mean()),
        "nested": metrics_with_ci(y, nested_mean, n_boot=args.n_boot, seed=args.seed),
        "nested_auroc_per_repeat": [fast_auroc(y, nested[r]) for r in range(nested.shape[0])],
        "selected_layer_counts": {int(k): int(v) for k, v in sorted(Counter(res.selected_layer.flatten().tolist()).items())},
        "peak_layer_by_curve": int(max(rows, key=lambda r: r["auroc"])["layer"]),
        "null_nested_auroc": null_nested_auc,
        "null_nested_bal_acc": null_nested_bacc,
        "config": {"folds": args.folds, "repeats": args.repeats, "inner_folds": args.inner_folds, "c_grid": list(parse_grid(args.c_grid)),
                   "null_repeats": args.null_repeats, "n_boot": args.n_boot, "seed": args.seed},
    }

    # Surface-feature baselines (no model involved) evaluated with the same outer folds.
    base = {}
    for name, factory in surface_models(args.seed).items():
        base[name] = metrics_with_ci(y, oof_sklearn(S, y, folds, factory).mean(0), n_boot=args.n_boot, seed=args.seed)
    if scope == "all":
        cat_col = S[:, [S_names.index("is_arithmetic")]]
        base["category_only"] = metrics_with_ci(y, oof_sklearn(cat_col, y, folds, surface_models(args.seed)["surface_logreg"]).mean(0), n_boot=args.n_boot, seed=args.seed)
    summary["baselines"] = base
    return rows, summary, nested_mean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activations", default=None, help="Default: <outdir>/activations.npz")
    p.add_argument("--questions", default=None, help="Default: <outdir>/questions_probe_labeled.json")
    p.add_argument("--outdir", default="results/probe")
    p.add_argument("--tag", default="", help="Suffix for output files, e.g. _intervention")
    p.add_argument("--label-threshold", type=float, default=1.0, choices=list(SENSITIVITY_THRESHOLDS), help="1.0 = needed if any no-tool sample was wrong (paper definition)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--c-grid", default="0.0001,0.001,0.01")
    p.add_argument("--null-repeats", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--no-mean-pool", action="store_true", help="Skip the mean-pooled comparison representation")
    p.add_argument("--scopes", default="all,arithmetic,factual", help="Comma-separated subset of all,arithmetic,factual")
    p.add_argument("--limit", type=int, default=None, help="Use only the first N questions (smoke test)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    args.activations = args.activations or str(outdir / "activations.npz")
    args.questions = args.questions or str(outdir / "questions_probe_labeled.json")
    questions = apply_limit(load_questions(args.questions), args.limit)
    npz = np.load(args.activations, allow_pickle=False)
    ids = [str(i) for i in npz["ids"]]
    pos = {i: k for k, i in enumerate(ids)}
    order = [pos[q["id"]] for q in questions]
    label_col = threshold_column(args.label_threshold)
    y_all = np.array([int(q[label_col]) for q in questions])
    cat = np.array([q["category"] for q in questions])
    S_all, S_names = surface_matrix(questions)

    features = ["last_token"] + ([] if args.no_mean_pool else ["mean_pool"])
    curve_rows, summaries = [], []
    for feature in features:
        arr = npz[feature][:, order, :]
        # last_token is run for every scope; the mean-pool comparison only for the pooled scope.
        wanted = args.scopes.split(",")
        scopes = [s for s in (["all", "arithmetic", "factual"] if feature == "last_token" else ["all"]) if s in wanted]
        for scope in scopes:
            idx = np.arange(len(y_all)) if scope == "all" else np.where(cat == scope)[0]
            y = y_all[idx]
            if min(y.sum(), (1 - y).sum()) < MIN_CLASS_COUNT:
                print(f"SKIP {feature}/{scope}: minority class has {min(y.sum(), (1 - y).sum())} < {MIN_CLASS_COUNT} samples")
                summaries.append({"scope": scope, "feature": feature, "skipped": True, "n": int(len(y)), "n_needed": int(y.sum())})
                continue
            print(f"== {feature}/{scope}: n={len(y)} needed={int(y.sum())} ({y.mean():.1%})")
            rows, summary, nested_mean = evaluate_scope(arr[:, idx, :], y, S_all[idx], S_names, args, scope, feature)
            curve_rows += rows
            summaries.append(summary)
            pd.DataFrame({"id": [questions[i]["id"] for i in idx], "category": cat[idx], "tool_needed": y, "p_needed_oof": nested_mean}) \
                .to_csv(outdir / f"oof_{scope}_{feature}{args.tag}.csv", index=False)
            n = summary["nested"]
            print(f"   nested AUROC {n['auroc']:.3f} [{n['auroc_lo']:.3f}, {n['auroc_hi']:.3f}]  bal.acc {n['bal_acc']:.3f}  "
                  f"null AUROC {np.mean(summary['null_nested_auroc']):.3f}  baselines { {k: round(v['auroc'], 3) for k, v in summary['baselines'].items()} }")

    pd.DataFrame(curve_rows).to_csv(outdir / f"probe_layer_curve{args.tag}.csv", index=False)
    (outdir / f"probe_nested{args.tag}.json").write_text(json.dumps({"label_col": label_col, "label_threshold": args.label_threshold, "activations": args.activations, "results": summaries}, indent=2))
    print(f"Saved probe results to {outdir}")


if __name__ == "__main__":
    main()
