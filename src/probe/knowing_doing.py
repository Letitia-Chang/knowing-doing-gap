"""Stage 6: compare probe-predicted need against actual tool calls (CPU).

Uses the nested-CV out-of-fold probe probabilities (oof_all_last_token*.csv): each question's
prediction comes from a probe and a layer that never saw that question's fold, so nothing here
is selected on the data being scored.

Cell naming (probe belief x behaviour):
  knows_does      probe says tool needed,     model called a tool
  knows_doesnt    probe says tool needed,     model did NOT call   <- classic knowing-doing gap
  unneeded_does   probe says tool NOT needed, model called a tool  <- over-use that matches the probe's belief
  neither         probe says tool NOT needed, model did not call
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import fast_auroc  # noqa: E402

THRESHOLD = 0.5  # P(needed) >= 0.5 means "probe says tool needed" (probe is class-balanced)


def cell_name(probe_needed: bool, called: bool) -> str:
    if probe_needed:
        return "knows_does" if called else "knows_doesnt"
    return "unneeded_does" if called else "neither"


def bootstrap_rate(flags: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    flags = np.asarray(flags, dtype=float)
    if len(flags) == 0:
        return {"n": 0, "rate": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = flags[rng.integers(0, len(flags), (n_boot, len(flags)))].mean(1)
    return {"n": int(len(flags)), "rate": float(flags.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def calibration_table(p: np.ndarray, needed: np.ndarray, called: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        rows.append({"bin": b, "p_lo": edges[b], "p_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else np.nan,
                     "frac_tool_needed": float(needed[m].mean()) if m.any() else np.nan,
                     "frac_tool_called": float(called[m].mean()) if m.any() else np.nan})
    tab = pd.DataFrame(rows)
    ok = tab["n"] > 0
    ece = float((tab.loc[ok, "n"] * (tab.loc[ok, "mean_p"] - tab.loc[ok, "frac_tool_needed"]).abs()).sum() / tab["n"].sum())
    return tab, ece


def analyse(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["probe_needed"] = df["p_needed_oof"] >= THRESHOLD
    df["cell"] = [cell_name(a, b) for a, b in zip(df["probe_needed"], df["tool_called"].astype(bool))]
    needed, called = df["tool_needed"].astype(bool).to_numpy(), df["tool_called"].astype(bool).to_numpy()
    probe_needed = df["probe_needed"].to_numpy()

    cells = (df.groupby(["category", "tool_needed", "probe_needed", "tool_called"]).size().rename("n").reset_index())
    cells_all = (df.groupby(["tool_needed", "probe_needed", "tool_called"]).size().rename("n").reset_index())
    cells_all.insert(0, "category", "all")
    eight_cell = pd.concat([cells_all, cells], ignore_index=True)

    over_use = ~needed & called
    under_use = needed & ~called
    cal, ece = calibration_table(df["p_needed_oof"].to_numpy(), needed.astype(float), called.astype(float))
    summary = {
        "n": int(len(df)),
        "tool_call_rate": float(called.mean()),
        "tool_needed_rate": float(needed.mean()),
        "probe_says_needed_rate": float(probe_needed.mean()),
        "four_cell_counts": {c: int((df["cell"] == c).sum()) for c in ["knows_does", "knows_doesnt", "unneeded_does", "neither"]},
        "behavioural_mismatch_rate": float((needed != called).mean()),
        "probe_vs_necessity_mismatch_rate": float((needed != probe_needed).mean()),
        "probe_vs_action_mismatch_rate": float((probe_needed != called).mean()),
        "over_use": {"definition": "tool_needed=0 and tool_called=1", "n": int(over_use.sum()),
                     "probe_says_not_needed": bootstrap_rate(~probe_needed[over_use])},
        "needed_and_called": {"n": int((needed & called).sum()), "probe_says_needed": bootstrap_rate(probe_needed[needed & called])},
        "under_use": {"definition": "tool_needed=1 and tool_called=0", "n": int(under_use.sum()),
                      "probe_says_needed": bootstrap_rate(probe_needed[under_use])},
        "not_needed_and_not_called": {"n": int((~needed & ~called).sum()), "probe_says_not_needed": bootstrap_rate(~probe_needed[~needed & ~called])},
        "auroc_probe_vs_tool_needed": fast_auroc(needed, df["p_needed_oof"].to_numpy()),
        "auroc_probe_vs_tool_called": fast_auroc(called, df["p_needed_oof"].to_numpy()),
        "ece_vs_tool_needed": ece,
        "by_category": {},
    }
    for c, g in df.groupby("category"):
        n_g, c_g, pn = g["tool_needed"].astype(bool).to_numpy(), g["tool_called"].astype(bool).to_numpy(), g["probe_needed"].to_numpy()
        summary["by_category"][c] = {
            "n": int(len(g)), "tool_call_rate": float(c_g.mean()), "tool_needed_rate": float(n_g.mean()),
            "over_use_n": int((~n_g & c_g).sum()), "over_use_probe_says_not_needed": bootstrap_rate(~pn[~n_g & c_g]),
            "four_cell_counts": {k: int((g["cell"] == k).sum()) for k in ["knows_does", "knows_doesnt", "unneeded_does", "neither"]},
        }
    return summary, eight_cell, cal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results/probe")
    p.add_argument("--prompt", default="baseline", choices=["baseline", "intervention"])
    p.add_argument("--tag", default="", help="Must match the --tag used for run_probe.py (e.g. _intervention)")
    p.add_argument("--feature", default="last_token")
    args = p.parse_args()

    outdir = Path(args.outdir)
    oof = pd.read_csv(outdir / f"oof_all_{args.feature}{args.tag}.csv")
    action = pd.DataFrame(json.loads((outdir / f"action_local_{args.prompt}.json").read_text()))[["id", "tool_called", "malformed"]]
    df = oof.merge(action, on="id", how="inner")
    if len(df) != len(oof):
        raise SystemExit(f"Only {len(df)}/{len(oof)} probe questions have an action record")
    summary, eight_cell, cal = analyse(df)
    summary.update({"prompt": args.prompt, "feature": args.feature, "threshold": THRESHOLD, "malformed_rate": float(df["malformed"].mean())})

    sfx = f"{args.tag}" if args.feature == "last_token" else f"_{args.feature}{args.tag}"
    eight_cell.to_csv(outdir / f"knowing_doing_cells{sfx}.csv", index=False)
    cal.to_csv(outdir / f"calibration{sfx}.csv", index=False)
    df.to_csv(outdir / f"knowing_doing_per_question{sfx}.csv", index=False)
    (outdir / f"knowing_doing{sfx}.json").write_text(json.dumps(summary, indent=2))
    fc = summary["four_cell_counts"]
    ou = summary["over_use"]["probe_says_not_needed"]
    print(f"[{args.prompt}] call rate {summary['tool_call_rate']:.1%}  four cells {fc}")
    print(f"  over-use n={summary['over_use']['n']}: probe says NOT needed on {ou['rate']:.1%} [{ou['lo']:.1%}, {ou['hi']:.1%}]")


if __name__ == "__main__":
    main()
