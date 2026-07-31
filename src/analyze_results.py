from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from io_utils import load_questions, read_json


def outcome(tool_needed: bool, tool_called: bool) -> str:
    if tool_needed and tool_called:
        return "Needed tool + called tool"
    if tool_needed and not tool_called:
        return "Needed tool + did NOT call tool"
    if (not tool_needed) and tool_called:
        return "No tool needed + DID call tool"
    return "No tool needed + no tool called"


def make_bar(series: pd.Series, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and analyze Knowing-Doing Gap results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--confidence-threshold", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    q_df = pd.DataFrame(load_questions(run_dir / "questions_labeled.json"))
    cog_df = pd.DataFrame(read_json(run_dir / "cognition.json")) if (run_dir / "cognition.json").exists() else pd.DataFrame()
    base_df = pd.DataFrame(read_json(run_dir / "action_baseline.json"))

    merged = q_df.merge(cog_df, on="id", how="left").merge(base_df, on="id", how="left", suffixes=("", "_baseline"))
    merged["outcome"] = merged.apply(lambda r: outcome(bool(r["tool_needed"]), bool(r["tool_called"])), axis=1)
    merged["necessity_action_mismatch"] = merged["tool_needed"] != merged["tool_called"]
    if "cognition_needs_tool" in merged.columns:
        merged["cognition_action_mismatch"] = merged["cognition_needs_tool"] != merged["tool_called"]
        merged["cognition_necessity_mismatch"] = merged["cognition_needs_tool"] != merged["tool_needed"]

    if (run_dir / "action_intervention.json").exists():
        inter_df = pd.DataFrame(read_json(run_dir / "action_intervention.json"))
        inter_df = inter_df.rename(columns={
            "tool_called": "tool_called_intervention",
            "tools_used": "tools_used_intervention",
            "final_answer": "final_answer_intervention",
            "action_correct": "action_correct_intervention",
            "malformed_action_json": "malformed_action_json_intervention",
        })
        merged = merged.merge(inter_df.drop(columns=[c for c in ["mode"] if c in inter_df.columns]), on="id", how="left")
        merged["outcome_intervention"] = merged.apply(
            lambda r: outcome(bool(r["tool_needed"]), bool(r["tool_called_intervention"]))
            if pd.notna(r.get("tool_called_intervention")) else None,
            axis=1,
        )
        merged["necessity_action_mismatch_intervention"] = merged["tool_needed"] != merged["tool_called_intervention"]

    merged.to_csv(run_dir / "merged_results.csv", index=False)

    rows: list[dict[str, Any]] = []
    n = len(merged)
    rows.append({"metric": "n_questions", "value": n})
    rows.append({"metric": "baseline_tool_call_rate", "value": merged["tool_called"].mean()})
    rows.append({"metric": "baseline_necessity_action_mismatch_rate", "value": merged["necessity_action_mismatch"].mean()})
    rows.append({"metric": "baseline_action_accuracy", "value": merged["action_correct"].mean()})
    if "cognition_needs_tool" in merged.columns:
        low = merged[merged["cognition_needs_tool"] == True]
        rows.append({"metric": "cognition_low_confidence_count", "value": len(low)})
        rows.append({"metric": "among_low_confidence_tool_called_rate", "value": low["tool_called"].mean() if len(low) else None})
        rows.append({"metric": "cognition_action_mismatch_rate", "value": merged["cognition_action_mismatch"].mean()})
        rows.append({"metric": "cognition_necessity_mismatch_rate", "value": merged["cognition_necessity_mismatch"].mean()})
    if "tool_called_intervention" in merged.columns:
        rows.append({"metric": "intervention_tool_call_rate", "value": merged["tool_called_intervention"].mean()})
        rows.append({"metric": "intervention_necessity_action_mismatch_rate", "value": merged["necessity_action_mismatch_intervention"].mean()})
        rows.append({"metric": "intervention_action_accuracy", "value": merged["action_correct_intervention"].mean()})

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)

    # Tables
    outcome_order = [
        "Needed tool + called tool",
        "Needed tool + did NOT call tool",
        "No tool needed + DID call tool",
        "No tool needed + no tool called",
    ]

    baseline_four_cell = merged["outcome"].value_counts().reindex(outcome_order).fillna(0).astype(int)
    baseline_four_cell.to_csv(run_dir / "four_cell_baseline.csv")

    pd.crosstab(merged["category"], merged["outcome"]).to_csv(run_dir / "breakdown_by_category.csv")
    pd.crosstab(merged["tool_needed"], merged["tool_called"], rownames=["tool_needed"], colnames=["tool_called"]).to_csv(run_dir / "necessity_action_crosstab_baseline.csv")

    make_bar(baseline_four_cell, "Baseline four-cell outcomes", figures_dir / "baseline_four_cell_outcomes.png")
    make_bar(pd.crosstab(merged["category"], merged["necessity_action_mismatch"]).rename(columns={False: "aligned", True: "mismatch"})["mismatch"], "Baseline mismatch count by category", figures_dir / "baseline_mismatch_by_category.png")

    if "necessity_action_mismatch_intervention" in merged.columns:
        intervention_four_cell = merged["outcome_intervention"].value_counts().reindex(outcome_order).fillna(0).astype(int)
        intervention_four_cell.to_csv(run_dir / "four_cell_intervention.csv")
        pd.crosstab(merged["category"], merged["outcome_intervention"]).to_csv(run_dir / "breakdown_by_category_intervention.csv")
        pd.crosstab(merged["tool_needed"], merged["tool_called_intervention"], rownames=["tool_needed"], colnames=["tool_called_intervention"]).to_csv(run_dir / "necessity_action_crosstab_intervention.csv")
        make_bar(intervention_four_cell, "Intervention four-cell outcomes", figures_dir / "intervention_four_cell_outcomes.png")

        comparison = pd.Series({
            "baseline": merged["necessity_action_mismatch"].mean(),
            "intervention": merged["necessity_action_mismatch_intervention"].mean(),
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        comparison.plot(kind="bar", ax=ax)
        ax.set_title("Necessity-action mismatch rate")
        ax.set_ylabel("Mismatch rate")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=0)
        fig.tight_layout()
        fig.savefig(figures_dir / "baseline_vs_intervention_mismatch.png", dpi=160)
        plt.close(fig)

    print("Saved:")
    print(f"  {run_dir / 'merged_results.csv'}")
    print(f"  {run_dir / 'summary_metrics.csv'}")
    print(f"  {figures_dir}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
