"""Prescription contract.

Every leaf is a Field: a value, the transcript line IDs it came from, and an optional flag.
A null value must carry a flag, so "not stated" is always explicit and never silently blank.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydField

T = TypeVar("T")
Flag = Literal["missing", "unclear", "needs_doctor_check"]


class Field(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    evidence: list[str] = PydField(default_factory=list)
    flag: Flag | None = "missing"

    @model_validator(mode="after")
    def _null_needs_flag(self) -> Field[T]:
        if self.value is None and self.flag is None:
            raise ValueError("a null value must have a flag")
        return self


def _missing() -> Field:
    return Field(value=None, evidence=[], flag="missing")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatientDetails(_Section):
    name: Field[str] = PydField(default_factory=_missing)
    age: Field[int] = PydField(default_factory=_missing)
    gender: Field[str] = PydField(default_factory=_missing)


MEDICATION_ATTRS = (
    "medicine_name",
    "strength",
    "dose",
    "frequency",
    "duration",
    "route",
    "instructions",
)


class Medication(_Section):
    medicine_name: Field[str] = PydField(default_factory=_missing)
    strength: Field[str] = PydField(default_factory=_missing)
    dose: Field[str] = PydField(default_factory=_missing)
    frequency: Field[str] = PydField(default_factory=_missing)
    duration: Field[str] = PydField(default_factory=_missing)
    route: Field[str] = PydField(default_factory=_missing)
    instructions: Field[str] = PydField(default_factory=_missing)


SCALAR_SECTIONS = ("history_of_present_illness", "duration", "follow_up")
LIST_SECTIONS = (
    "chief_complaints",
    "associated_symptoms",
    "past_medical_history",
    "medication_history",
    "allergies",
    "examination_findings",
    "diagnosis",
    "investigations",
    "advice",
)
# Sections whose evidence must come from doctor lines only.
DOCTOR_ONLY_SECTIONS = ("diagnosis", "medications")


class Prescription(_Section):
    patient_details: PatientDetails = PydField(default_factory=PatientDetails)
    chief_complaints: list[Field[str]] = PydField(default_factory=list)
    history_of_present_illness: Field[str] = PydField(default_factory=_missing)
    duration: Field[str] = PydField(default_factory=_missing)
    associated_symptoms: list[Field[str]] = PydField(default_factory=list)
    past_medical_history: list[Field[str]] = PydField(default_factory=list)
    medication_history: list[Field[str]] = PydField(default_factory=list)
    allergies: list[Field[str]] = PydField(default_factory=list)
    examination_findings: list[Field[str]] = PydField(default_factory=list)
    diagnosis: list[Field[str]] = PydField(default_factory=list)
    medications: list[Medication] = PydField(default_factory=list)
    investigations: list[Field[str]] = PydField(default_factory=list)
    advice: list[Field[str]] = PydField(default_factory=list)
    follow_up: Field[str] = PydField(default_factory=_missing)

    def iter_fields(self) -> Iterator[tuple[str, Field]]:
        """Yield (path, field) for every leaf, e.g. 'medications[0].dose', 'diagnosis[1]'."""
        for attr in ("name", "age", "gender"):
            yield f"patient_details.{attr}", getattr(self.patient_details, attr)
        for section in SCALAR_SECTIONS:
            yield section, getattr(self, section)
        for section in LIST_SECTIONS:
            for i, f in enumerate(getattr(self, section)):
                yield f"{section}[{i}]", f
        for i, med in enumerate(self.medications):
            for attr in MEDICATION_ATTRS:
                yield f"medications[{i}].{attr}", getattr(med, attr)


def section_of(path: str) -> str:
    """'medications[0].dose' -> 'medications'; 'patient_details.age' -> 'patient_details'."""
    return path.split("[")[0].split(".")[0]
