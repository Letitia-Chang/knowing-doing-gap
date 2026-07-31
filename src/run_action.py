from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from graders import grade_answer
from io_utils import apply_limit, load_questions, write_json
from model_client import TogetherChatClient
from tools import execute_tool


BASELINE_SYSTEM = """You are an assistant in an agentic setting. You may answer directly or call exactly one tool first.
Available tools:
- calculator: use for arithmetic. Arguments: {"expression": "..."}
- search: use for factual lookup. Arguments: {"query": "..."}

Do not use a tool unless you decide it is useful. Respond ONLY as valid JSON in one of these forms:
{"action":"answer","answer":"..."}
{"action":"tool","tool":"calculator","arguments":{"expression":"..."}}
{"action":"tool","tool":"search","arguments":{"query":"..."}}
"""


INTERVENTION_SYSTEM = """You have access to tools.

Answer directly when the question is simple or well-known.
Use a tool when the question is hard, calculation-heavy, recent, obscure, or uncertain.

Do not call tools by default, but do not avoid them when they are useful.

Respond ONLY as valid JSON in one of these forms:
{"action":"answer","answer":"..."}
{"action":"tool","tool":"calculator","arguments":{"expression":"..."}}
{"action":"tool","tool":"search","arguments":{"query":"..."}}
"""


FINAL_SYSTEM = (
    "You are a helpful assistant. Use the provided tool result to answer the original question. "
    "Give only the final answer."
)


def extract_json(text: str) -> Any | None:
    """
    Extract JSON from the model response.

    The expected output is a dict like:
      {"action":"answer","answer":"..."}
      {"action":"tool","tool":"calculator","arguments":{"expression":"..."}}

    But some models may return valid JSON primitives, like:
      63
      "Paris"

    So this function returns Any, not just dict.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON-looking object from a longer response.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    return None


def choose_action(
    client: TogetherChatClient,
    q: dict[str, Any],
    intervention: bool,
) -> tuple[Any | None, str]:
    system = INTERVENTION_SYSTEM if intervention else BASELINE_SYSTEM
    prompt = f"Question ID: {q['id']}\nQuestion: {q['question']}"
    raw = client.ask(prompt, system=system, temperature=0.0, max_tokens=256)
    return extract_json(raw), raw


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agentic action probe with manual JSON tool protocol."
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Usually results/questions_labeled.json",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--intervention", action="store_true")
    parser.add_argument("--mock-search", default="data/mock_search.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    questions = apply_limit(load_questions(args.questions), args.limit)
    client = TogetherChatClient()

    mode = "intervention" if args.intervention else "baseline"
    results = []
    raw_records = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] action/{mode} {q['id']}")

        action_obj, raw_choice = choose_action(client, q, args.intervention)

        tool_called = False
        tools_used: list[str] = []
        tool_result = None
        malformed = action_obj is None

        # Case 1: model returned valid JSON but not a dict, e.g. 63 or "Paris".
        # Treat it as a direct answer instead of crashing.
        if action_obj is not None and not isinstance(action_obj, dict):
            final_text = str(action_obj)

        # Case 2: model returned the expected JSON object.
        elif isinstance(action_obj, dict):
            action = action_obj.get("action")

            # Preferred protocol:
            #   {"action":"tool","tool":"calculator","arguments":{...}}
            # Robust fallback for occasional model outputs like:
            #   {"action":"calculator","arguments":{...}}
            if action == "tool" or action in {"calculator", "search"}:
                tool_called = True
                tool = str(action_obj.get("tool", "")) if action == "tool" else str(action)
                tools_used.append(tool)

                arguments = action_obj.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {}

                try:
                    tool_result = execute_tool(
                        tool,
                        arguments,
                        question_id=q["id"],
                        mock_path=args.mock_search,
                    )
                except Exception as e:
                    tool_result = f"TOOL_ERROR: {type(e).__name__}: {e}"

                final_prompt = (
                    f"Original question: {q['question']}\n"
                    f"Tool used: {tool}\n"
                    f"Tool result: {tool_result}\n"
                    "Final answer:"
                )
                final_text = client.ask(
                    final_prompt,
                    system=FINAL_SYSTEM,
                    temperature=0.0,
                    max_tokens=128,
                )

            elif action == "answer":
                final_text = str(action_obj.get("answer", ""))

            else:
                # Valid JSON object, but not in the required protocol.
                malformed = True
                final_text = raw_choice

        # Case 3: no parseable JSON at all.
        else:
            final_text = raw_choice

        correct = grade_answer(q, final_text)

        results.append(
            {
                "id": q["id"],
                "mode": mode,
                "tool_called": tool_called,
                "tools_used": tools_used,
                "final_answer": final_text,
                "action_correct": correct,
                "malformed_action_json": malformed,
            }
        )

        raw_records.append(
            {
                "id": q["id"],
                "mode": mode,
                "raw_choice": raw_choice,
                "parsed_choice": action_obj,
                "tool_result": tool_result,
                "final_answer": final_text,
            }
        )

        print(f"  tool_called={tool_called} tools={tools_used} correct={correct}")

    write_json(results, outdir / f"action_{mode}.json")
    write_json(raw_records, outdir / f"raw_action_{mode}.json")

    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()