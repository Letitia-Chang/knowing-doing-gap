"""Recompute necessity labels from the SAVED no-tool answers with the v2 grader (CPU, no model).

Every one of the 10 saved answers for every factual question is regraded, whether it was right or
wrong before. Arithmetic grading is unchanged. Question-level decisions (bad/outdated/ambiguous gold,
extra accepted answers) come from data/label_decisions.json, which was written from questions and
answers only, before any probe was rerun.

Outputs in --outdir: questions_probe_labeled.json, regrade_changes.csv, label_review.csv (no probe
information), regrade_summary.json, plus copies of the action files and run_info.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import necessity_labels  # noqa: E402
from grading_v2 import match_factual, tokens  # noqa: E402
from graders import grade_arithmetic  # noqa: E402
from io_utils import load_questions, read_json, write_json  # noqa: E402


def candidates(q: dict, accept: dict, extra: dict) -> list[str]:
    base = accept.get(q["id"], q.get("accept") or [])
    return [str(q["reference_answer"]), *base, *extra.get(q["id"], [])]


def regrade_question(q: dict, answers: list[str], cands: list[str]) -> tuple[list[bool], list[str | None]]:
    if q["category"] == "arithmetic":
        flags = [grade_arithmetic(a, str(q["reference_answer"])) for a in answers]
        return flags, [None] * len(answers)
    via = [match_factual(a, cands) for a in answers]
    return [v is not None for v in via], via


def fmt(counter: Counter, k: int) -> str:
    return " || ".join(f"{a.replace(chr(10), ' ')[:90]} x{n}" for a, n in counter.most_common(k))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", default="results/probe")
    p.add_argument("--outdir", default="results/probe_corrected")
    p.add_argument("--questions", default="data/questions_probe.json")
    p.add_argument("--accept", default="data/accept_lists_v2.json")
    p.add_argument("--decisions", default="data/label_decisions.json")
    args = p.parse_args()
    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accept = read_json(args.accept)
    dec = read_json(args.decisions) if Path(args.decisions).exists() else {}
    exclude, extra = dec.get("exclude", {}), dec.get("accept_extra", {})
    runs = {r["id"]: r for r in read_json(indir / "necessity_runs_local.json")}
    old = {q["id"]: q for q in load_questions(indir / "questions_probe_labeled.json")}

    labeled, changes, review = [], [], []
    for q in load_questions(args.questions):
        answers = runs[q["id"]]["answers"]
        flags, via = regrade_question(q, answers, candidates(q, accept, extra))
        lab = necessity_labels(sum(flags), len(flags))
        o = old[q["id"]]
        if flags != runs[q["id"]]["correct"]:
            changes.append([q["id"], q["category"], o["necessity_correct_count"], sum(flags), o["tool_needed"], lab["tool_needed"], q["id"] in exclude])
        if q["category"] == "factual":
            wrong = Counter(a for a, f in zip(answers, flags) if not f)
            right = Counter(a for a, f in zip(answers, flags) if f)
            review.append([q["id"], q["question"], q["reference_answer"], o["necessity_correct_count"], sum(flags), fmt(wrong, 4), fmt(right, 2),
                           "; ".join(sorted({v for v in via if v and v != q["reference_answer"]}))[:120]])
        if q["id"] not in exclude:
            labeled.append({**q, **lab})

    write_json({"questions": labeled}, outdir / "questions_probe_labeled.json")
    for name, head, rows in (
        ("regrade_changes.csv", ["id", "category", "old_correct", "new_correct", "old_tool_needed", "new_tool_needed", "excluded"], changes),
        ("label_review.csv", ["id", "question", "gold", "old", "new", "wrong", "right", "matched_via_alias"], review),
    ):
        with open(outdir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(rows)
    for name in ("action_local_baseline.json", "action_local_intervention.json", "run_info.json"):
        shutil.copy(indir / name, outdir / name)

    seen: dict[str, list[str]] = {}
    for q in load_questions(args.questions):
        seen.setdefault(" ".join(tokens(q["question"])), []).append(q["id"])
    kept = {q["id"] for q in labeled}
    new = {q["id"]: q["tool_needed"] for q in labeled}
    summary = {
        "n_questions": len(old), "n_excluded": len(set(exclude) & set(old)), "n_kept": len(labeled),
        "kept_by_category": dict(Counter(q["category"] for q in labeled)),
        "questions_with_any_changed_grade": len(changes),
        "needed_to_not_needed": sorted(i for i in kept if old[i]["tool_needed"] and not new[i]),
        "not_needed_to_needed": sorted(i for i in kept if not old[i]["tool_needed"] and new[i]),
        "original30_label_changes": sorted(i for i in kept if i[0] in "AF" and old[i]["tool_needed"] != new[i]),
        "duplicate_questions": [v for v in seen.values() if len(v) > 1],
        "n_extra_accept": len(extra),
    }
    write_json(summary, outdir / "regrade_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
