from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from graders import grade_answer
from io_utils import apply_limit, load_questions, write_json
from model_client import TogetherChatClient

SYSTEM = "You are a helpful assistant. Answer the user's question directly and concisely. Do not use external tools."


def build_prompt(q: dict[str, Any]) -> str:
    return f"Question: {q['question']}\nGive only the final answer, with no explanation."


def main() -> None:
    parser = argparse.ArgumentParser(description="Empirically label model-adaptive tool necessity.")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--outdir", default="results", help="Output folder. Defaults to results/.")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    questions = apply_limit(load_questions(args.questions), args.limit)
    client = TogetherChatClient()

    labeled = []
    raw = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] necessity {q['id']}")
        runs = []
        correct_flags = []
        for r in range(args.n):
            answer = client.ask(build_prompt(q), system=SYSTEM, temperature=args.temp, max_tokens=128)
            correct = grade_answer(q, answer)
            runs.append({"run": r + 1, "answer": answer, "correct": correct})
            correct_flags.append(correct)
            print(f"  run {r + 1}: correct={correct} answer={answer[:80]!r}")
        tool_needed = not all(correct_flags)
        labeled_q = dict(q)
        labeled_q["tool_needed"] = tool_needed
        labeled_q["necessity_correct_count"] = sum(correct_flags)
        labeled_q["necessity_n"] = args.n
        labeled.append(labeled_q)
        raw.append({"id": q["id"], "question": q["question"], "runs": runs, "tool_needed": tool_needed})

    write_json({"questions": labeled}, outdir / "questions_labeled.json")
    write_json(raw, outdir / "raw_necessity.json")
    write_json({
        "model": client.config.model,
        "provider": "TogetherAI",
        "necessity_n": args.n,
        "necessity_temperature": args.temp,
        "question_count": len(questions),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, outdir / "metadata.json")
    print(f"\nSaved to {outdir}")


if __name__ == "__main__":
    main()
