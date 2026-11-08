"""Plain-code checks applied after the model. The model proposes; this code decides.

Rules:
1. A non-null field must cite at least one evidence ID, and every cited ID must exist.
2. Diagnosis and medication fields must cite doctor lines only.
A field that breaks a rule is dropped: value -> null, evidence -> [], flag -> needs_doctor_check.
"""

from __future__ import annotations

from dataclasses import dataclass

from medscribe.schemas import (
    DOCTOR_ONLY_SECTIONS,
    Field,
    Line,
    Prescription,
    Transcript,
    section_of,
)


@dataclass
class Drop:
    path: str
    reason: str
    original_value: object
    original_evidence: list[str]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "reason": self.reason,
            "original_value": self.original_value,
            "original_evidence": self.original_evidence,
        }


def evidence_problem(field: Field, lines: dict[str, Line]) -> str | None:
    """A non-null value with no evidence, or citing IDs that do not exist, is invented."""
    if field.value is None:
        return None
    if not field.evidence:
        return "no evidence cited"
    unknown = [i for i in field.evidence if i not in lines]
    if unknown:
        return f"unknown evidence ids {unknown}"
    return None


def speaker_problem(path: str, field: Field, lines: dict[str, Line]) -> str | None:
    """Diagnosis and medication fields may cite doctor lines only."""
    if field.value is None or section_of(path) not in DOCTOR_ONLY_SECTIONS:
        return None
    non_doctor = [i for i in field.evidence if i in lines and lines[i].speaker != "doctor"]
    if non_doctor:
        return f"cites non-doctor lines {non_doctor}"
    return None


def violation(path: str, field: Field, transcript: Transcript) -> str | None:
    """Return why a non-null field breaks a rule, or None if it is fine."""
    lines = transcript.by_id()
    return evidence_problem(field, lines) or speaker_problem(path, field, lines)


def post_check(rx: Prescription, transcript: Transcript) -> tuple[Prescription, list[Drop]]:
    checked = rx.model_copy(deep=True)
    drops: list[Drop] = []
    for path, field in checked.iter_fields():
        reason = violation(path, field, transcript)
        if reason is None:
            continue
        drops.append(Drop(path, reason, field.value, list(field.evidence)))
        field.value = None
        field.evidence = []
        field.flag = "needs_doctor_check"
    return checked, drops
