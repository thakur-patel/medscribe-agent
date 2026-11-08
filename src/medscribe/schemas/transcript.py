"""Transcript contract: an ordered list of speaker-labelled lines with stable IDs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator

Speaker = Literal["doctor", "patient", "unclear"]


class Line(BaseModel):
    id: str
    speaker: Speaker
    text: str


class Transcript(BaseModel):
    lines: list[Line]

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_list(cls, data: Any) -> Any:
        # The contract is "a list of lines"; accept both [..] and {"lines": [..]}.
        if isinstance(data, list):
            return {"lines": data}
        return data

    @model_validator(mode="after")
    def _unique_ids(self) -> Transcript:
        ids = [line.id for line in self.lines]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate line ids: {sorted(dupes)}")
        return self

    def by_id(self) -> dict[str, Line]:
        return {line.id: line for line in self.lines}

    def to_prompt_text(self) -> str:
        return "\n".join(f"[{ln.id}] {ln.speaker}: {ln.text}" for ln in self.lines)
