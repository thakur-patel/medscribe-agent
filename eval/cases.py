"""Golden case loading. One case per JSONL line; lines starting with '#' are comments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from medscribe.schemas import Prescription, Transcript


@dataclass
class Case:
    case_id: str
    split: Literal["dev", "heldout"]
    tags: list[str]
    transcript: Transcript
    expected: Prescription


def load_cases(golden_dir: str | Path, split: str) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(Path(golden_dir, split).glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
                case = Case(
                    case_id=raw["case_id"],
                    split=raw["split"],
                    tags=list(raw.get("tags", [])),
                    transcript=Transcript.model_validate(raw["transcript"]),
                    expected=Prescription.model_validate(raw["expected_prescription"]),
                )
            except Exception as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
            if case.split != split:
                raise ValueError(f"{path}:{n}: case {case.case_id} has split {case.split!r}")
            cases.append(case)
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in golden set")
    return cases
