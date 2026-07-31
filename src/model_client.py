from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI


@dataclass
class ModelConfig:
    model: str = os.getenv("TOGETHER_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    base_url: str = "https://api.together.xyz/v1"
    max_tokens: int = 512
    timeout: float = 60.0
    max_retries: int = 4


class TogetherChatClient:
    """Small OpenAI-compatible wrapper for TogetherAI chat completions."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url, timeout=self.config.timeout)
        self.stats = {"calls": 0, "seconds": 0.0, "retries": 0}

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                )
                self.stats["calls"] += 1
                self.stats["seconds"] += time.time() - t0
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                self.stats["retries"] += 1
                wait = min(2 ** attempt, 10)
                print(f"[TogetherAI] attempt {attempt + 1} failed: {type(e).__name__}. Retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"TogetherAI call failed after retries: {last_err}")

    def ask(self, user_prompt: str, system: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
