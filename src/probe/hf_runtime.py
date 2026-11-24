"""Hugging Face model helpers for the GPU stages. torch/transformers are imported lazily."""
from __future__ import annotations

from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_model(model_name: str = DEFAULT_MODEL, precision: str = "bf16"):
    """Load tokenizer + model. ``precision`` is "bf16" (default) or "8bit" (only if bf16 does not fit).

    Returns (tokenizer, model, info) where ``info`` records what was actually used.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: dict[str, Any] = {"device_map": "auto"}
    if precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown precision {precision!r}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    info = {
        "model": model_name,
        "precision": precision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else None,
    }
    return tok, model, info


def chat_text(tok, system: str, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(tok, model, texts: list[str], *, max_new_tokens: int, do_sample: bool, temperature: float = 1.0, num_return_sequences: int = 1) -> list[list[str]]:
    """Generate for a batch of already-templated prompts; returns ``num_return_sequences`` strings per prompt.

    Sampling settings are passed explicitly because Qwen's generation_config.json otherwise
    injects top_p=0.8, top_k=20 and repetition_penalty=1.05, which would silently change the
    "temperature 0.7" necessity definition. Here: plain temperature sampling, or pure greedy.
    """
    import torch

    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "repetition_penalty": 1.0, "pad_token_id": tok.pad_token_id}
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=num_return_sequences)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
    k = num_return_sequences if do_sample else 1
    return [[d.strip() for d in decoded[i * k:(i + 1) * k]] for i in range(len(texts))]
