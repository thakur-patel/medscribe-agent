"""C0: the fixed baseline. One LLM call, Pydantic validation, one retry, plain-code post-check."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from medscribe.llm import LLM, LLMResponse, Message
from medscribe.pipeline.post_check import Drop, post_check
from medscribe.schemas import Prescription, Transcript

PIPELINE_VERSION = "c0"
MAX_ATTEMPTS = 2  # first call plus one retry with the validation errors


@dataclass
class DraftResult:
    ok: bool
    prescription: Prescription | None = None  # after post-check
    raw_prescription: Prescription | None = None  # as the model returned it, validated
    drops: list[Drop] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    attempts: int = 0
    calls: list[LLMResponse] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def summary(self) -> dict:
        return {
            "ok": self.ok,
            "pipeline": PIPELINE_VERSION,
            "attempts": self.attempts,
            "errors": self.errors,
            "drops": [d.to_dict() for d in self.drops],
            "cost_usd": self.cost_usd,
            "latency_s": self.latency_s,
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
        }


def build_messages(prompt_template: str, transcript: Transcript) -> list[Message]:
    text = prompt_template.replace("{{TRANSCRIPT}}", transcript.to_prompt_text())
    return [{"role": "user", "content": text}]


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse(text: str) -> Prescription:
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    return Prescription.model_validate(json.loads(text))


def _format_errors(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return [f"{type(exc).__name__}: {exc}"]


def draft(transcript: Transcript, llm: LLM, prompt_template: str) -> DraftResult:
    start = time.perf_counter()
    result = DraftResult(ok=False)
    messages = build_messages(prompt_template, transcript)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result.attempts = attempt
        try:
            resp = llm.complete(messages)
        except Exception as exc:  # network or provider failure: report, do not crash
            result.errors = _format_errors(exc)
            break
        result.calls.append(resp)
        try:
            raw = parse(resp.text)
        except (json.JSONDecodeError, ValidationError) as exc:
            result.errors = _format_errors(exc)
            messages = messages + [
                {"role": "assistant", "content": resp.text},
                {
                    "role": "user",
                    "content": "Your JSON failed validation with these errors:\n"
                    + "\n".join(f"- {e}" for e in result.errors)
                    + "\nReturn the corrected JSON object only.",
                },
            ]
            continue
        result.raw_prescription = raw
        result.prescription, result.drops = post_check(raw, transcript)
        result.errors = []
        result.ok = True
        break

    result.latency_s = time.perf_counter() - start
    return result
