"""Prompt transfer: score one prompt's activations with a probe trained on the OTHER prompt (CPU).

Two separately trained probes agree whenever both work, so their agreement says little about whether a
prompt change moved the necessity signal. Here one probe is frozen and applied to both prompts. Layer
and C are chosen on training rows of the training prompt only, and every score is out-of-fold.
  strict     the trained scaler and weights applied as they are
  recentred  test activations first shifted by the label-free difference of training-row means, which
             removes a constant offset caused by the different system prompt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import fast_auroc, make_probe, metrics_with_ci, run_nested, stratified_folds  # noqa: E402
from io_utils import load_questions  # noqa: E402


def transfer_scores(XA, XB, y, folds, n_jobs=1, seed=0, c_grid=(1e-4, 1e-3, 1e-2), inner_splits=3) -> dict[str, np.ndarray]:
    """Mean out-of-fold decision scores of probes trained on A: on A, on B (strict), on B (recentred)."""
    res = run_nested(XA, y, folds, c_grid=c_grid, inner_splits=inner_splits, seed=seed, n_jobs=n_jobs)
    sel, R = res.selected_layer, 1 + max(f[0] for f in folds)
    out = {k: np.zeros((R, len(y))) for k in ("A", "B_strict", "B_recentred")}
    for r, k, tr, te in folds:
        l = sel[r, k]
        a, b = XA[l].astype(np.float32), XB[l].astype(np.float32)
        probe = make_probe(res.best_C[r, k, l], seed).fit(a[tr], y[tr])
        out["A"][r, te] = probe.decision_function(a[te])
        out["B_strict"][r, te] = probe.decision_function(b[te])
        out["B_recentred"][r, te] = probe.decision_function(b[te] - b[tr].mean(0) + a[tr].mean(0))
    return {k: v.mean(0) for k, v in out.items()}


def boot_diff(x: np.ndarray, g: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of x in group g minus mean of x outside it, with a percentile bootstrap CI over questions."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if g[i].any() and (~g[i]).any():
            vals.append(x[i][g[i]].mean() - x[i][~g[i]].mean())
    return {"diff": float(x[g].mean() - x[~g].mean()), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_in": int(g.sum()), "n_out": int((~g).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/probe_corrected")
    p.add_argument("--actdir", default="results/probe")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    out, act = Path(args.outdir), Path(args.actdir)

    qs = load_questions(out / "questions_probe_labeled.json")
    ids = [q["id"] for q in qs]
    y = np.array([int(q["tool_needed"]) for q in qs])
    cat = np.array([q["category"] for q in qs])
    X, called = {}, {}
    for name, f in (("baseline", "activations.npz"), ("intervention", "activations_intervention.npz")):
        z = np.load(act / f, allow_pickle=False)
        pos = {str(i): k for k, i in enumerate(z["ids"])}
        X[name] = z["last_token"][:, [pos[i] for i in ids], :]
        a = {r["id"]: r["tool_called"] for r in json.loads((out / f"action_local_{name}.json").read_text())}
        called[name] = np.array([a[i] for i in ids])
    flipped = called["baseline"] & ~called["intervention"]
    stayed = called["baseline"] & called["intervention"]

    S: dict = {"n": len(ids), "n_flipped_called_to_not": int(flipped.sum()), "n_stayed_called": int(stayed.sum()), "directions": {}}
    per_q = pd.DataFrame({"id": ids, "category": cat, "tool_needed": y, "called_baseline": called["baseline"], "called_intervention": called["intervention"]})
    lines = ["### Prompt transfer: one frozen probe scored on both prompts", "",
             "| Scope | Trained on | AUROC on same prompt | AUROC on other prompt, strict | recentred | Spearman (strict) |", "|---|---|---|---|---|--:|"]
    for scope in ("all", "arithmetic", "factual"):
        m = np.ones(len(y), bool) if scope == "all" else cat == scope
        for A, B in (("baseline", "intervention"), ("intervention", "baseline")):
            folds = stratified_folds(y[m], args.folds, args.repeats, args.seed)
            s = transfer_scores(X[A][:, m, :], X[B][:, m, :], y[m], folds, args.n_jobs, args.seed)
            met = {k: metrics_with_ci(y[m], v, threshold=0.0) for k, v in s.items()}
            row = {"metrics": met, "spearman_strict": float(spearmanr(s["A"], s["B_strict"])[0]), "spearman_recentred": float(spearmanr(s["A"], s["B_recentred"])[0]),
                   "mean_score_A": float(s["A"].mean()), "mean_score_B_strict": float(s["B_strict"].mean())}
            if A == "baseline":
                grp = (flipped | stayed)[m]
                d = (s["B_recentred"] - s["A"])[grp]
                row["score_change_flipped_minus_stayed"] = boot_diff(d, flipped[m][grp])
                row["baseline_score_flipped_minus_stayed"] = boot_diff(s["A"][grp], flipped[m][grp])
                row["auroc_baseline_score_predicts_staying_called"] = fast_auroc(stayed[m][grp], s["A"][grp])
            S["directions"][f"{scope}:{A}->{B}"] = row
            if scope == "all":
                for k, v in s.items():
                    per_q[f"{A}_probe_{k}"] = v
            c = lambda k: f"{met[k]['auroc']:.3f} [{met[k]['auroc_lo']:.3f}, {met[k]['auroc_hi']:.3f}]"  # noqa: E731
            lines.append(f"| {scope} | {A} | {c('A')} | {c('B_strict')} | {c('B_recentred')} | {row['spearman_strict']:.3f} |")
    lines += ["", "Baseline-trained probe, questions that called a tool under the baseline prompt; scores are log-odds of need. 'Flipped' = stopped calling under the intervention prompt, 'stayed' = still called.", "",
              "| Scope | n flipped / stayed | Baseline score, flipped minus stayed | Score change across prompts (recentred), flipped minus stayed | AUROC of baseline score for staying |", "|---|---|---|---|--:|"]
    for scope in ("all", "arithmetic", "factual"):
        r = S["directions"][f"{scope}:baseline->intervention"]
        a, b = r["baseline_score_flipped_minus_stayed"], r["score_change_flipped_minus_stayed"]
        lines.append(f"| {scope} | {a['n_in']} / {a['n_out']} | {a['diff']:+.2f} [{a['lo']:+.2f}, {a['hi']:+.2f}] | {b['diff']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}] | {r['auroc_baseline_score_predicts_staying_called']:.3f} |")
    (out / "transfer.json").write_text(json.dumps(S, indent=2))
    (out / "transfer.md").write_text("\n".join(lines) + "\n")
    per_q.to_csv(out / "transfer_per_question.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
