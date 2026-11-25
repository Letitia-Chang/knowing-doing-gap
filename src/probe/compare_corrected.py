"""Original vs corrected labels, side by side, plus the averaged-embedding baseline per scope (CPU)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(p: Path):
    return json.loads(p.read_text())


def find(res: list[dict], scope: str, feature: str) -> dict:
    return next(r for r in res if r["scope"] == scope and r["feature"] == feature)


def fmt(m: dict) -> str:
    return f"{m['auroc']:.3f} [{m['auroc_lo']:.3f}, {m['auroc_hi']:.3f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", default="results/probe")
    p.add_argument("--new", default="results/probe_corrected")
    args = p.parse_args()
    old, new = Path(args.old), Path(args.new)
    R = load(new / "regrade_summary.json")
    a, b = load(old / "probe_nested.json")["results"], load(new / "probe_nested.json")["results"]
    curve = pd.read_csv(new / "probe_layer_curve.csv")
    S: dict = {"regrade": R, "scopes": {}}
    lines = ["### Label correction: original vs corrected", "",
             f"{R['n_excluded']} questions excluded ({R['n_kept']} kept: {R['kept_by_category']}); among the kept ones, {len(R['needed_to_not_needed'])} labels changed from needed to not needed and {len(R['not_needed_to_needed'])} the other way.", "",
             "| Scope | Original: n (needed) | Probe AUROC, original | Corrected: n (needed) | Probe AUROC, corrected | Best surface baseline, corrected | Averaged input embeddings, corrected |", "|---|---|---|---|---|--:|---|"]
    for scope in ("all", "arithmetic", "factual"):
        o, n = find(a, scope, "last_token"), find(b, scope, "last_token")
        surf = max(v["auroc"] for k, v in n["baselines"].items())
        e = curve[(curve.scope == scope) & (curve.feature == "mean_pool") & (curve.layer == 0)].iloc[0]
        S["scopes"][scope] = {"original": o["nested"], "corrected": n["nested"], "corrected_surface_best": surf,
                              "corrected_embedding_layer0": {k: float(e[k]) for k in ("auroc", "auroc_lo", "auroc_hi")},
                              "corrected_null_auroc": n["null_nested_auroc"], "original_needed": o["n_needed"], "corrected_needed": n["n_needed"], "corrected_n": n["n"]}
        lines.append(f"| {scope} | {o['n']} ({o['n_needed']}) | {fmt(o['nested'])} | {n['n']} ({n['n_needed']}) | {fmt(n['nested'])} | {surf:.3f} | {e.auroc:.3f} [{e.auroc_lo:.3f}, {e.auroc_hi:.3f}] |")
    lines += ["", "| Prompt | Over-use n, original -> corrected | Probe says not needed on over-use, original -> corrected | ...on not-needed and not called, original -> corrected |", "|---|---|---|---|"]
    for name, sfx in (("baseline", ""), ("intervention", "_intervention")):
        x, y = load(old / f"knowing_doing{sfx}.json"), load(new / f"knowing_doing{sfx}.json")
        r = lambda d, k: f"{d[k]['probe_says_not_needed']['rate']:.0%} (n={d[k]['n']})"  # noqa: E731
        lines.append(f"| {name} | {x['over_use']['n']} -> {y['over_use']['n']} | {r(x, 'over_use')} -> {r(y, 'over_use')} | {r(x, 'not_needed_and_not_called')} -> {r(y, 'not_needed_and_not_called')} |")
    (new / "correction.json").write_text(json.dumps(S, indent=2))
    (new / "correction.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
