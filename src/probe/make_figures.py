"""Stage 7a: figures from saved probe results (CPU, matplotlib only; no model or retraining)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CELL_LABELS = {
    "knows_does": "Probe: needed\nModel: called tool\n(knows & does)",
    "knows_doesnt": "Probe: needed\nModel: NO call\n(knows, doesn't)",
    "unneeded_does": "Probe: not needed\nModel: called tool\n(over-use)",
    "neither": "Probe: not needed\nModel: no call\n(neither)",
}
CELL_COLORS = {"knows_does": "#2a9d8f", "knows_doesnt": "#e76f51", "unneeded_does": "#e9c46a", "neither": "#8d99ae"}


def layer_curve(outdir: Path, tag: str) -> None:
    curve = pd.read_csv(outdir / f"probe_layer_curve{tag}.csv")
    nested = {(r["scope"], r["feature"]): r for r in json.loads((outdir / f"probe_nested{tag}.json").read_text())["results"] if not r.get("skipped")}
    scopes = [s for s in ["all", "arithmetic", "factual"] if (s, "last_token") in nested]
    fig, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.2), squeeze=False, sharey=True)
    for ax, scope in zip(axes[0], scopes):
        d = curve[(curve.scope == scope) & (curve.feature == "last_token")]
        ax.plot(d.layer, d.auroc, color="#1d4e89", lw=2, label="probe (last prompt token)")
        ax.fill_between(d.layer, d.auroc_lo, d.auroc_hi, color="#1d4e89", alpha=0.2)
        if scope == "all" and ((curve.scope == "all") & (curve.feature == "mean_pool")).any():
            m = curve[(curve.scope == "all") & (curve.feature == "mean_pool")]
            ax.plot(m.layer, m.auroc, color="#6a994e", lw=1.5, ls="--", label="probe (mean-pooled)")
        ax.plot(d.layer, d.null_auroc_mean, color="grey", lw=1.5, label="shuffled labels (mean)")
        ax.axhline(d.null_auroc_p95_all_layers.iloc[0], color="grey", ls=":", lw=1, label="shuffled 95th pct")
        for name, b in nested[(scope, "last_token")]["baselines"].items():
            ax.axhline(b["auroc"], ls="-.", lw=1, color={"surface_logreg": "#e76f51", "surface_gboost": "#bc4749", "category_only": "#f4a261"}[name], label=name.replace("_", " "))
        n = nested[(scope, "last_token")]["nested"]
        ax.set_title(f"{scope}: nested AUROC {n['auroc']:.2f} [{n['auroc_lo']:.2f}, {n['auroc_hi']:.2f}]", fontsize=10)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("out-of-fold AUROC for tool_needed")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / f"probe_layer_curve{tag}.png", dpi=160)
    plt.close(fig)


def four_cell(outdir: Path, tag: str) -> None:
    kd = json.loads((outdir / f"knowing_doing{tag}.json").read_text())
    cats = ["all"] + sorted(kd["by_category"])
    fig, axes = plt.subplots(1, len(cats), figsize=(4.6 * len(cats), 4.2), squeeze=False, sharey=False)
    for ax, c in zip(axes[0], cats):
        counts = kd["four_cell_counts"] if c == "all" else kd["by_category"][c]["four_cell_counts"]
        keys = list(CELL_LABELS)
        bars = ax.bar(range(4), [counts[k] for k in keys], color=[CELL_COLORS[k] for k in keys])
        ax.bar_label(bars)
        ax.set_xticks(range(4))
        ax.set_xticklabels([CELL_LABELS[k] for k in keys], fontsize=6.5)
        ax.set_title(f"{c} (n={sum(counts.values())})", fontsize=10)
        ax.set_ylabel("questions")
    fig.suptitle(f"Probe-predicted need vs actual tool call — prompt: {kd['prompt']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / f"four_cell{tag}.png", dpi=160)
    plt.close(fig)


def calibration(outdir: Path, tag: str) -> None:
    cal = pd.read_csv(outdir / f"calibration{tag}.csv")
    kd = json.loads((outdir / f"knowing_doing{tag}.json").read_text())
    ok = cal.n > 0
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color="grey", ls=":", label="perfect calibration")
    ax.plot(cal.mean_p[ok], cal.frac_tool_needed[ok], "o-", color="#1d4e89", label="fraction truly tool-needed")
    ax.plot(cal.mean_p[ok], cal.frac_tool_called[ok], "s--", color="#e76f51", label="fraction where model called a tool")
    for x, y, n in zip(cal.mean_p[ok], cal.frac_tool_needed[ok], cal.n[ok]):
        ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), fontsize=6, ha="center")
    ax.set_xlabel("probe P(tool needed), out-of-fold (class-balanced probe)")
    ax.set_ylabel("observed fraction")
    ax.set_title(f"Calibration — ECE vs necessity {kd['ece_vs_tool_needed']:.3f}\n(point labels = n per bin)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / f"calibration{tag}.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/probe")
    p.add_argument("--tag", default="", help="e.g. _intervention")
    args = p.parse_args()
    outdir = Path(args.outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    layer_curve(outdir, args.tag)
    four_cell(outdir, args.tag)
    calibration(outdir, args.tag)
    print(f"Figures written to {outdir / 'figures'}")


if __name__ == "__main__":
    main()
