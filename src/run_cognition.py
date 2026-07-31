from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from io_utils import apply_limit, load_questions, write_json
from model_client import TogetherChatClient

SYSTEM = "You are a careful assistant. Follow the requested output format exactly."

PROMPT = """Before answering the question, estimate whether you can answer it correctly WITHOUT looking anything up or using a calculator.

Use this exact format:
Confidence: <integer from 1 to 5>
Answer: <your final answer>

Scale:
1 = very unlikely to answer correctly without a tool
2 = somewhat unlikely / should use a tool
3 = unsure
4 = likely can answer without a tool
5 = definitely can answer without a tool

Question: {question}
"""


def parse_confidence(text: str) -> int | None:
    m = re.search(r"confidence\s*[:=]\s*([1-5])", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit cognition/confidence probe.")
    parser.add_argument("--questions", required=True, help="Usually results/questions_labeled.json")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=int, default=2)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    questions = apply_limit(load_questions(args.questions), args.limit)
    client = TogetherChatClient()

    results = []
    raw = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] cognition {q['id']}")
        text = client.ask(PROMPT.format(question=q["question"]), system=SYSTEM, temperature=0.0, max_tokens=256)
        conf = parse_confidence(text)
        needs_tool = None if conf is None else conf <= args.threshold
        results.append({
            "id": q["id"],
            "confidence": conf,
            "cognition_needs_tool": needs_tool,
            "cognition_answer": text,
        })
        raw.append({"id": q["id"], "raw": text})
        print(f"  confidence={conf} cognition_needs_tool={needs_tool}")

    write_json(results, outdir / "cognition.json")
    write_json(raw, outdir / "raw_cognition.json")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
