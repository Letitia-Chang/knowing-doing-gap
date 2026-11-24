"""Stage 7b: collect every number quoted in the README into results/probe/summary.json and summary.md.

Nothing is typed by hand: the README's probe section is copied from summary.md, and the resume
lines are derived from summary.json, so every figure traces back to a script in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SENSITIVITY_THRESHOLDS, threshold_column  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def ci(m: dict, key: str = "auroc") -> str:
    return f"{m[key]:.3f} [{m[f'{key}_lo']:.3f}, {m[f'{key}_hi']:.3f}]"


def pct(r: dict) -> str:
    return f"{r['rate']:.0%} [{r['lo']:.0%}, {r['hi']:.0%}] (n={r['n']})"


def find(results: list[dict], scope: str, feature: str = "last_token") -> dict | None:
    return next((r for r in results if r["scope"] == scope and r["feature"] == feature), None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/probe")
    args = p.parse_args()
    out = Path(args.outdir)

    qs = pd.DataFrame(load(out / "questions_probe_labeled.json")["questions"])
    S: dict = {"run_info": load(out / "run_info.json")}
    lines: list[str] = []

    # --- dataset & necessity -------------------------------------------------
    ds = {"n": int(len(qs)), "by_category": qs.category.value_counts().to_dict()}
    for t in SENSITIVITY_THRESHOLDS:
        col = threshold_column(t)
        ds[col] = {"overall": float(qs[col].mean()), **{c: float(g[col].mean()) for c, g in qs.groupby("category")}}
    S["dataset"] = ds
    lines += ["### Question set and local necessity labels", "",
              f"{ds['n']} questions ({ds['by_category']}); tool_needed = any of N={int(qs.necessity_n.iloc[0])} no-tool samples wrong.", "",
              "| Label rule | overall | arithmetic | factual |", "|---|--:|--:|--:|"]
    for t in SENSITIVITY_THRESHOLDS:
        d = ds[threshold_column(t)]
        rule = "any wrong (primary)" if t == 1.0 else f"accuracy < {t:.0%}"
        lines.append(f"| {rule} | {d['overall']:.1%} | {d['arithmetic']:.1%} | {d['factual']:.1%} |")

    together = ROOT / "results" / "questions_labeled.json"
    if together.exists():
        tg = pd.DataFrame(load(together)["questions"])[["id", "tool_needed"]].rename(columns={"tool_needed": "together"})
        m = qs.merge(tg, on="id")
        S["agreement_with_together_labels"] = {"n": int(len(m)), "agree": int((m.tool_needed == m.together).sum()),
                                               "together_needed": int(m.together.sum()), "local_needed": int(m.tool_needed.sum())}
        a = S["agreement_with_together_labels"]
        lines += ["", f"On the original 30 questions, local-bf16 labels agree with the TogetherAI-Turbo labels on {a['agree']}/{a['n']} "
                      f"(needed: {a['local_needed']} local vs {a['together_needed']} Together)."]

    # --- probe ---------------------------------------------------------------
    probe = load(out / "probe_nested.json")
    if probe:
        res = probe["results"]
        S["probe"] = {"label_col": probe["label_col"], "results": res}
        lines += ["", "### Probe: nested-CV out-of-fold performance (95% bootstrap CI over questions)", "",
                  "| Scope | Representation | n (needed) | Nested AUROC | Nested balanced acc. | Shuffled-label AUROC | Surface LR AUROC | Surface GBM AUROC | Layers chosen (count) |",
                  "|---|---|--:|--:|--:|--:|--:|--:|---|"]
        for r in res:
            if r.get("skipped"):
                lines.append(f"| {r['scope']} | {r['feature']} | {r['n']} ({r['n_needed']}) | skipped: too few samples in one class | | | | | |")
                continue
            top = sorted(r["selected_layer_counts"].items(), key=lambda kv: -kv[1])[:3]
            b = r["baselines"]
            lines.append(f"| {r['scope']} | {r['feature']} | {r['n']} ({r['n_needed']}) | {ci(r['nested'])} | {ci(r['nested'], 'bal_acc')} | "
                         f"{np.mean(r['null_nested_auroc']):.3f} | {b['surface_logreg']['auroc']:.3f} | {b['surface_gboost']['auroc']:.3f} | "
                         f"{', '.join(f'L{k} ({v})' for k, v in top)} |")
        sens = {}
        for t in SENSITIVITY_THRESHOLDS[1:]:
            tag = f"_acc{int(round(t * 100))}"
            pj = load(out / f"probe_nested{tag}.json")
            if pj and find(pj["results"], "all") and not find(pj["results"], "all").get("skipped"):
                sens[tag] = find(pj["results"], "all")["nested"]
        if sens:
            S["probe_label_sensitivity"] = sens
            lines += ["", "Label-threshold sensitivity (pooled scope, last-token probe): " +
                      "; ".join(f"tool_needed = accuracy < {int(k[4:])}% -> nested AUROC {ci(v)}" for k, v in sens.items())]

    # --- knowing-doing -------------------------------------------------------
    kd_runs = {}
    for name, sfx in (("baseline", ""), ("intervention", "_intervention")):
        kd = load(out / f"knowing_doing{sfx}.json")
        if kd:
            kd_runs[name] = kd
    S["knowing_doing"] = kd_runs
    if kd_runs:
        lines += ["", "### Probe-predicted need vs actual tool calls (out-of-fold probe, threshold 0.5)", "",
                  "| Prompt | n | Tool-call rate | Needed + called | Needed + NOT called | Not needed + called (over-use) | Not needed + not called |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for name, kd in kd_runs.items():
            # Behaviour-vs-necessity cells straight from the per-question file (ground truth, not the probe).
            pq = pd.read_csv(out / f"knowing_doing_per_question{'' if name == 'baseline' else '_intervention'}.csv")
            nn, cc = pq.tool_needed.astype(bool), pq.tool_called.astype(bool)
            lines.append(f"| {name} | {kd['n']} | {kd['tool_call_rate']:.1%} | {int((nn & cc).sum())} | {int((nn & ~cc).sum())} | {int((~nn & cc).sum())} | {int((~nn & ~cc).sum())} |")
        lines += ["", "| Prompt | Probe cell: knows & does | knows, doesn't | not-needed & does | neither | Over-use items where probe says NOT needed | Needed+not-called items where probe says needed |",
                  "|---|--:|--:|--:|--:|---|---|"]
        for name, kd in kd_runs.items():
            fc = kd["four_cell_counts"]
            lines.append(f"| {name} | {fc['knows_does']} | {fc['knows_doesnt']} | {fc['unneeded_does']} | {fc['neither']} | "
                         f"{pct(kd['over_use']['probe_says_not_needed'])} | {pct(kd['under_use']['probe_says_needed'])} |")
        lines += ["", "Probe score vs behaviour: " + "; ".join(
            f"{n}: AUROC(probe, tool_needed)={kd['auroc_probe_vs_tool_needed']:.3f}, AUROC(probe, tool_called)={kd['auroc_probe_vs_tool_called']:.3f}, ECE={kd['ece_vs_tool_needed']:.3f}" for n, kd in kd_runs.items())]

    # --- baseline vs intervention (secondary) -------------------------------
    if len(kd_runs) == 2:
        b = pd.read_csv(out / "knowing_doing_per_question.csv")
        i = pd.read_csv(out / "knowing_doing_per_question_intervention.csv")
        m = b.merge(i, on="id", suffixes=("_b", "_i"))
        flip = m[m.tool_called_b.astype(bool) & ~m.tool_called_i.astype(bool)]
        S["prompt_comparison"] = {
            "n": int(len(m)),
            "call_rate_baseline": float(m.tool_called_b.mean()), "call_rate_intervention": float(m.tool_called_i.mean()),
            "n_called_to_not_called": int(len(flip)),
            "spearman_probe_prob_between_prompts": float(m.p_needed_oof_b.corr(m.p_needed_oof_i, method="spearman")),
            "mean_abs_probe_prob_change": float((m.p_needed_oof_b - m.p_needed_oof_i).abs().mean()),
            "mean_probe_prob_change_on_flipped": float((flip.p_needed_oof_i - flip.p_needed_oof_b).mean()) if len(flip) else None,
        }
        c = S["prompt_comparison"]
        lines += ["", "### Same questions, two prompts (secondary)", "",
                  f"Tool-call rate {c['call_rate_baseline']:.1%} (baseline) -> {c['call_rate_intervention']:.1%} (intervention); "
                  f"{c['n_called_to_not_called']} questions flipped from calling to not calling. "
                  f"Probe P(needed) across the two prompts: Spearman rho = {c['spearman_probe_prob_between_prompts']:.3f}, mean |change| = {c['mean_abs_probe_prob_change']:.3f}"
                  + (f", mean change on the flipped questions = {c['mean_probe_prob_change_on_flipped']:+.3f}." if c["mean_probe_prob_change_on_flipped"] is not None else ".")]

    ri = S["run_info"] or {}
    if ri.get("necessity"):
        n = ri["necessity"]
        lines += ["", f"Run info: model {n['model']}, precision {n['precision']}, GPU {n['gpu']}, torch {n['torch']}, transformers {n['transformers']}."]

    (out / "summary.json").write_text(json.dumps(S, indent=2, default=float))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
