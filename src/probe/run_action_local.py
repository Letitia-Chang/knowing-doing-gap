"""Stage 3: behavioural action probe with the LOCAL model (GPU).

Same manual-JSON tool protocol, system prompts and greedy decoding as src/run_action.py.
Only the first-turn *decision* is needed for the probe study (did the model call a tool?),
so tools are not executed and no second turn is run.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTION_SYSTEMS, action_user_prompt, extract_json, parse_tool_decision, update_run_info  # noqa: E402
from graders import grade_answer  # noqa: E402
from hf_runtime import DEFAULT_MODEL, chat_text, generate, load_model  # noqa: E402
from io_utils import apply_limit, load_questions, write_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default=None, help="Default: <outdir>/questions_probe_labeled.json")
    p.add_argument("--outdir", default="results/probe")
    p.add_argument("--prompt", default="baseline", choices=sorted(ACTION_SYSTEMS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--precision", default="bf16", choices=["bf16", "8bit"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    questions = apply_limit(load_questions(args.questions or outdir / "questions_probe_labeled.json"), args.limit)
    tok, model, info = load_model(args.model, args.precision)
    system = ACTION_SYSTEMS[args.prompt]

    results = []
    t0 = time.time()
    for start in range(0, len(questions), args.batch_size):
        batch = questions[start:start + args.batch_size]
        texts = [chat_text(tok, system, action_user_prompt(q)) for q in batch]
        raws = [r[0] for r in generate(tok, model, texts, max_new_tokens=256, do_sample=False)]
        for q, raw in zip(batch, raws):
            d = parse_tool_decision(extract_json(raw), raw)
            results.append({
                "id": q["id"],
                "prompt": args.prompt,
                "tool_called": d["tool_called"],
                "tool": d["tool"],
                "malformed": d["malformed"],
                "direct_answer": d["direct_answer"],
                "direct_answer_correct": (grade_answer(q, d["direct_answer"]) if d["direct_answer"] is not None else None),
                "raw_choice": raw,
            })
        print(f"[{min(start + args.batch_size, len(questions))}/{len(questions)}] {time.time() - t0:.0f}s")

    write_json(results, outdir / f"action_local_{args.prompt}.json")
    update_run_info(outdir / "run_info.json", f"action_{args.prompt}", {**info, "decoding": "greedy", "batch_size": args.batch_size, "n_questions": len(results), "seconds": round(time.time() - t0)})
    rate = sum(r["tool_called"] for r in results) / len(results)
    print(f"Tool-call rate ({args.prompt}): {rate:.1%}  malformed: {sum(r['malformed'] for r in results)}")


if __name__ == "__main__":
    main()
