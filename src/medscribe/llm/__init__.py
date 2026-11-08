"""A small LLM interface so the pipeline never imports a provider SDK directly.

Tests use FakeLLM; real runs use OpenRouterLLM (OpenAI-compatible API).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from medscribe.config import AppConfig, ModelConfig

Message = dict[str, str]


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class LLM(Protocol):
    def complete(self, messages: list[Message]) -> LLMResponse: ...


class FakeLLM:
    """Returns canned responses in order (or from a function of the messages)."""

    def __init__(self, responses: list[str] | Callable[[list[Message]], str]):
        self._responses = responses
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if callable(self._responses):
            text = self._responses(messages)
        else:
            if not self._responses:
                raise RuntimeError("FakeLLM ran out of responses")
            text = self._responses.pop(0)
        return LLMResponse(text=text, model="fake")


class OpenRouterLLM:
    def __init__(self, model: ModelConfig, base_url: str, api_key: str):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    @classmethod
    def from_config(cls, cfg: AppConfig, model_key: str = "default") -> OpenRouterLLM:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"Set {cfg.api_key_env} (see .env.example)")
        return cls(cfg.model(model_key), cfg.base_url, key)

    def complete(self, messages: list[Message]) -> LLMResponse:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model.id,
            messages=messages,
            temperature=self.model.temperature,
            max_tokens=self.model.max_tokens,
            response_format={"type": "json_object"},
            extra_body={"usage": {"include": True}},
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        tin = getattr(usage, "prompt_tokens", 0) or 0
        tout = getattr(usage, "completion_tokens", 0) or 0
        cost = getattr(usage, "cost", None)
        if cost is None:
            cost = (tin * self.model.price_in_per_mtok + tout * self.model.price_out_per_mtok) / 1e6
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=resp.model or self.model.id,
            input_tokens=tin,
            output_tokens=tout,
            cost_usd=float(cost),
            latency_s=latency,
        )
