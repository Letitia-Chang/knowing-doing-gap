"""Stage 2: model-adaptive tool necessity, recomputed with the LOCAL model (GPU).

Same definition as src/label_necessity.py: N no-tool samples per question at temperature T,
same prompt, same grader. ``tool_needed`` = any sample wrong. The accuracy fraction and
two stricter/looser thresholds are stored for sensitivity checks.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NECESSITY_SYSTEM, necessity_labels, necessity_user_prompt, update_run_info  # noqa: E402
from graders import grade_answer  # noqa: E402
from hf_runtime import DEFAULT_MODEL, chat_text, generate, load_model  # noqa: E402
from io_utils import apply_limit, load_questions, write_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default="data/questions_probe.json")
    p.add_argument("--outdir", default="results/probe")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--precision", default="bf16", choices=["bf16", "8bit"])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=8, help="Questions per generate() call (x n sequences each).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    import torch

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    questions = apply_limit(load_questions(args.questions), args.limit)
    tok, model, info = load_model(args.model, args.precision)
    torch.manual_seed(args.seed)
    print(f"Loaded {info}")

    labeled, runs_out = [], []
    t0 = time.time()
    for start in range(0, len(questions), args.batch_size):
        batch = questions[start:start + args.batch_size]
        texts = [chat_text(tok, NECESSITY_SYSTEM, necessity_user_prompt(q)) for q in batch]
        answers = generate(tok, model, texts, max_new_tokens=128, do_sample=True, temperature=args.temp, num_return_sequences=args.n)
        for q, ans in zip(batch, answers):
            flags = [grade_answer(q, a) for a in ans]
            labeled.append({**q, **necessity_labels(sum(flags), args.n)})
            runs_out.append({"id": q["id"], "question": q["question"], "answers": ans, "correct": flags})
        print(f"[{min(start + args.batch_size, len(questions))}/{len(questions)}] {time.time() - t0:.0f}s")

    write_json({"questions": labeled}, outdir / "questions_probe_labeled.json")
    write_json(runs_out, outdir / "necessity_runs_local.json")

    # Human-audit sheet: factual questions where all N answers were "wrong". Either the model truly
    # does not know, or the dataset gold answer is bad/ungradeable; a person should spot-check these.
    with open(outdir / "gold_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold", "accepted_aliases", "sample_answers_distinct"])
        for q, r in zip(labeled, runs_out):
            if q["category"] == "factual" and q["necessity_correct_count"] == 0:
                distinct = list(dict.fromkeys(a.replace("\n", " ") for a in r["answers"]))[:4]
                w.writerow([q["id"], q["question"], q["reference_answer"], "; ".join(q.get("accept", [])[:5]), " || ".join(distinct)])

    update_run_info(outdir / "run_info.json", "necessity", {**info, "n": args.n, "temperature": args.temp, "seed": args.seed, "batch_size": args.batch_size,
                                                           "sampling": "temperature only (top_p=1, top_k off, repetition_penalty=1)",
                                                           "n_questions": len(labeled), "seconds": round(time.time() - t0)})
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
