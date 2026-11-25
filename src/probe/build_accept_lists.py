"""Rebuild alias lists for the already-selected TriviaQA questions with the v2 filter (needs `datasets`).

The questions, ids and gold values in data/questions_probe.json do not change; only accepted aliases do.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_questions import load_triviaqa  # noqa: E402
from grading_v2 import filter_aliases  # noqa: E402
from io_utils import load_questions, write_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="data/questions_probe.json")
    p.add_argument("--out", default="data/accept_lists_v2.json")
    p.add_argument("--split", default="validation")
    args = p.parse_args()
    by_q = {re.sub(r"\s+", " ", r["question"]).strip(): r for r in load_triviaqa(args.split)}
    out = {}
    for q in load_questions(args.questions):
        if q.get("source") != "triviaqa":
            continue
        rec = by_q[q["question"]]
        if rec["answer"]["value"] != q["reference_answer"]:
            raise SystemExit(f"{q['id']}: gold value mismatch {rec['answer']['value']!r} vs {q['reference_answer']!r}")
        out[q["id"]] = filter_aliases(q["reference_answer"], rec["answer"]["aliases"])
    write_json(out, args.out)
    print(f"Wrote alias lists for {len(out)} questions to {args.out}")


if __name__ == "__main__":
    main()
