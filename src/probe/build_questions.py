"""Build data/questions_probe.json: the original 30 questions plus generated arithmetic and TriviaQA factual QA.

Needs the ``datasets`` package and internet access (TriviaQA download). The output file is
committed, so GPU machines only need to read it; they never run this script.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from io_utils import load_questions, write_json  # noqa: E402
from question_gen import assemble, generate_arithmetic, select_factual  # noqa: E402


def load_triviaqa(split: str) -> list[dict]:
    from datasets import load_dataset  # lazy: only needed for building

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    return [{"question": r["question"], "question_id": r["question_id"], "answer": {"value": r["answer"]["value"], "aliases": r["answer"]["aliases"]}} for r in ds]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--originals", default="data/questions.json")
    p.add_argument("--out", default="data/questions_probe.json")
    p.add_argument("--n-per-category", type=int, default=500, help="Total per category, including the 15 originals.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--triviaqa-split", default="validation")
    args = p.parse_args()

    originals = load_questions(args.originals)
    n_orig = Counter(q["category"] for q in originals)
    seen = {q["question"] for q in originals}

    arithmetic = generate_arithmetic(args.n_per_category - n_orig["arithmetic"], seed=args.seed, seen=seen)
    factual = select_factual(load_triviaqa(args.triviaqa_split), args.n_per_category - n_orig["factual"], seed=args.seed, seen=seen)
    questions = assemble(originals, arithmetic, factual, seed=args.seed)

    write_json({"_meta": {"seed": args.seed, "n_per_category": args.n_per_category, "triviaqa_split": args.triviaqa_split,
                          "arithmetic": "Paper Appendix A families, re-implemented (src/probe/question_gen.py)",
                          "factual": "Original 15 + TriviaQA rc.nocontext sample (dataset-supplied gold answers)"},
                "questions": questions}, args.out)
    print(f"Wrote {len(questions)} questions to {args.out}: {dict(Counter(q['category'] for q in questions))}")


if __name__ == "__main__":
    main()
