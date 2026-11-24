"""Stage 4: hidden-state extraction (GPU).

For every question, run ONE forward pass over the exact templated action prompt (system prompt +
"Question ID/Question" user turn + generation prompt) and save, for every layer (embeddings + each
transformer block):
  - ``last_token``: hidden state at the final prompt token, i.e. the position that produces the
    first generated token (the model's decision point; "t = -1" in the paper's notation)
  - ``mean_pool``: mean over the user-turn + generation-prompt tokens (the constant system prompt
    is excluded; causal attention makes its states identical across questions anyway)
Arrays have shape (n_layers + 1, n_questions, hidden) and are stored as float16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ACTION_SYSTEMS, action_user_prompt, update_run_info  # noqa: E402
from hf_runtime import DEFAULT_MODEL, chat_text, load_model  # noqa: E402
from io_utils import apply_limit, load_questions  # noqa: E402


def to_float16(x: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: non-finite activations")
    if np.abs(x).max() > 6e4:  # float16 max is 65504
        print(f"WARNING: {name} exceeds float16 range; keeping float32")
        return x.astype(np.float32)
    return x.astype(np.float16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", default=None, help="Default: <outdir>/questions_probe_labeled.json")
    p.add_argument("--outdir", default="results/probe", help="Writes <outdir>/activations[_<prompt>].npz")
    p.add_argument("--prompt", default="baseline", choices=sorted(ACTION_SYSTEMS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--precision", default="bf16", choices=["bf16", "8bit"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    import torch

    outdir = Path(args.outdir)
    out = outdir / ("activations.npz" if args.prompt == "baseline" else f"activations_{args.prompt}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    questions = apply_limit(load_questions(args.questions or outdir / "questions_probe_labeled.json"), args.limit)
    tok, model, info = load_model(args.model, args.precision)
    system = ACTION_SYSTEMS[args.prompt]

    # Number of tokens in the system block, so mean pooling can start at the user turn.
    prefix_text = tok.apply_chat_template([{"role": "system", "content": system}], tokenize=False)
    n_prefix = len(tok(prefix_text, add_special_tokens=False)["input_ids"])

    last_chunks, mean_chunks = [], []
    last_tokens_seen = set()
    t0 = time.time()
    for start in range(0, len(questions), args.batch_size):
        batch = questions[start:start + args.batch_size]
        texts = [chat_text(tok, system, action_user_prompt(q)) for q in batch]
        assert all(t.startswith(prefix_text) for t in texts), "system block is not a shared prefix"
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states  # tuple of (B, T, H), len = n_layers + 1
        mask = enc["attention_mask"]
        T = mask.shape[1]
        pad_len = T - mask.sum(1)
        keep = (torch.arange(T, device=mask.device)[None, :] >= (pad_len + n_prefix)[:, None]).float()
        denom = keep.sum(1, keepdim=True)
        last_tokens_seen.update(tok.convert_ids_to_tokens(enc["input_ids"][:, -1]))
        last_chunks.append(torch.stack([h[:, -1, :].float() for h in hs]).cpu().numpy())
        mean_chunks.append(torch.stack([(h.float() * keep[:, :, None]).sum(1) / denom for h in hs]).cpu().numpy())
        print(f"[{min(start + args.batch_size, len(questions))}/{len(questions)}] {time.time() - t0:.0f}s")

    last = to_float16(np.concatenate(last_chunks, axis=1), "last_token")
    mean = to_float16(np.concatenate(mean_chunks, axis=1), "mean_pool")
    np.savez(out, last_token=last, mean_pool=mean, ids=np.array([q["id"] for q in questions]),
             meta=json.dumps({"prompt": args.prompt, "n_prefix_tokens": n_prefix, "last_token_strings": sorted(last_tokens_seen), **info}))
    update_run_info(outdir / "run_info.json", f"activations_{args.prompt}",
                    {**info, "shape_last_token": list(last.shape), "dtype": str(last.dtype), "batch_size": args.batch_size, "last_token_strings": sorted(last_tokens_seen), "seconds": round(time.time() - t0)})
    print(f"Saved {out}  last_token{last.shape} {last.dtype}")


if __name__ == "__main__":
    main()
