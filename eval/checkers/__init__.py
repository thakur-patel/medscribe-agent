"""Plain-code checkers. No LLM judges here: every metric is deterministic and explainable.

Metrics per case:
- field precision and recall: non-null predicted values vs non-null expected values.
- medication exact match: name, strength, dose, frequency and duration all equal.
- invented field rate: non-null values with no valid evidence (no IDs, or unknown IDs).
- missing-info recall: expected-null fields that came back null and flagged, not filled.
- speaker violations: diagnosis or medication fields citing a non-doctor line.
Invented rate and speaker violations are reported on the raw model output (what the model did)
and on the final output (what reaches the doctor after the post-check).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from medscribe.pipeline.post_check import evidence_problem, speaker_problem
from medscribe.schemas import (
    LIST_SECTIONS,
    MEDICATION_ATTRS,
    SCALAR_SECTIONS,
    Field,
    Prescription,
    Transcript,
)

EXACT_MED_ATTRS = ("medicine_name", "strength", "dose", "frequency", "duration")
# Free-text sections compare by token overlap; everything else must match after normalising.
FUZZY_SECTIONS = {
    "history_of_present_illness",
    "chief_complaints",
    "associated_symptoms",
    "past_medical_history",
    "medication_history",
    "examination_findings",
    "diagnosis",
    "advice",
    "follow_up",
    "investigations",
    "allergies",
}
FUZZY_THRESHOLD = 0.5

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "fourteen": "14",
    "thirty": "30",
}
_PLURALS = {"days": "day", "weeks": "week", "months": "month", "tablets": "tablet"}
_FREQUENCY = {
    "od": "od",
    "once a day": "od",
    "once daily": "od",
    "daily": "od",
    "once a day in the morning": "od morning",
    "once daily in the morning": "od morning",
    "in the morning": "od morning",
    "bd": "bd",
    "bid": "bd",
    "twice a day": "bd",
    "twice daily": "bd",
    "2 times a day": "bd",
    "tds": "tds",
    "tid": "tds",
    "thrice daily": "tds",
    "thrice a day": "tds",
    "3 times a day": "tds",
    "3 times daily": "tds",
    "hs": "hs",
    "at night": "hs",
    "at bedtime": "hs",
    "once at night": "hs",
    "once a day at night": "hs",
    "once daily at night": "hs",
    "sos": "sos",
    "prn": "sos",
    "as needed": "sos",
    "when needed": "sos",
    "when required": "sos",
}


def normalize(value: object, attr: str | None = None) -> str:
    text = str(value).lower()
    text = re.sub(r"(\d)\s*(mg|ml|g|mcg)\b", r"\1 \2", text)
    text = re.sub(r"[^\w\s./%]", " ", text)
    words = [_NUMBER_WORDS.get(w, w) for w in text.split()]
    words = [_PLURALS.get(w, w) for w in words]
    text = " ".join(words).strip(" .")
    if attr == "frequency":
        text = _FREQUENCY.get(text, text)
    return text


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"\W+", text) if t}


def values_match(a: str, b: str, fuzzy: bool) -> bool:
    if a == b:
        return True
    if not fuzzy:
        return False
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= FUZZY_THRESHOLD


@dataclass(frozen=True)
class Item:
    key: str  # what slot this value fills, e.g. "diagnosis" or "medications.dose|paracetamol"
    value: str
    fuzzy: bool


def items(rx: Prescription) -> list[Item]:
    out: list[Item] = []

    def add(key: str, f: Field, section: str, attr: str | None = None) -> None:
        if f.value is not None:
            out.append(Item(key, normalize(f.value, attr), section in FUZZY_SECTIONS))

    for attr in ("name", "age", "gender"):
        add(f"patient_details.{attr}", getattr(rx.patient_details, attr), "patient_details")
    for section in SCALAR_SECTIONS:
        add(section, getattr(rx, section), section)
    for section in LIST_SECTIONS:
        for f in getattr(rx, section):
            add(section, f, section)
    for med in rx.medications:
        name = normalize(med.medicine_name.value) if med.medicine_name.value else "?"
        for attr in MEDICATION_ATTRS:
            add(f"medications.{attr}|{name}", getattr(med, attr), "medications", attr)
    return out


def match_count(pred: list[Item], expected: list[Item]) -> int:
    unused = list(pred)
    tp = 0
    for exp in expected:
        for i, p in enumerate(unused):
            if p.key == exp.key and values_match(p.value, exp.value, exp.fuzzy):
                tp += 1
                del unused[i]
                break
    return tp


def _med_index(rx: Prescription) -> dict[str, dict[str, str | None]]:
    index = {}
    for med in rx.medications:
        if med.medicine_name.value is None:
            continue
        index[normalize(med.medicine_name.value)] = {
            a: (normalize(getattr(med, a).value, a) if getattr(med, a).value is not None else None)
            for a in MEDICATION_ATTRS
        }
    return index


def _expected_null_paths(rx: Prescription) -> list[tuple[str, str | None, str]]:
    """(slot, medicine name or None, attr) for every expected-null scalar or medication field."""
    out: list[tuple[str, str | None, str]] = []
    for attr in ("name", "age", "gender"):
        if getattr(rx.patient_details, attr).value is None:
            out.append(("patient_details", None, attr))
    for section in SCALAR_SECTIONS:
        if getattr(rx, section).value is None:
            out.append((section, None, ""))
    for med in rx.medications:
        if med.medicine_name.value is None:
            continue
        for attr in MEDICATION_ATTRS:
            if getattr(med, attr).value is None:
                out.append(("medications", normalize(med.medicine_name.value), attr))
    return out


def _lookup(rx: Prescription, slot: str, med: str | None, attr: str) -> Field | None:
    if slot == "patient_details":
        return getattr(rx.patient_details, attr)
    if slot != "medications":
        return getattr(rx, slot)
    for m in rx.medications:
        if m.medicine_name.value is not None and normalize(m.medicine_name.value) == med:
            return getattr(m, attr)
    return None  # medicine not predicted at all: not counted for missing-info recall


def safety_counts(rx: Prescription | None, transcript: Transcript) -> dict[str, int]:
    if rx is None:
        return {"non_null": 0, "invented": 0, "speaker_violations": 0}
    lines = transcript.by_id()
    non_null = invented = speaker = 0
    for path, f in rx.iter_fields():
        if f.value is None:
            continue
        non_null += 1
        invented += evidence_problem(f, lines) is not None
        speaker += speaker_problem(path, f, lines) is not None
    return {"non_null": non_null, "invented": invented, "speaker_violations": speaker}


@dataclass
class CaseMetrics:
    field_tp: int
    field_pred: int
    field_expected: int
    med_exact: int
    med_pred: int
    med_expected: int
    missing_recalled: int
    missing_expected: int
    raw_non_null: int
    raw_invented: int
    raw_speaker_violations: int
    final_non_null: int
    final_invented: int
    final_speaker_violations: int

    def to_dict(self) -> dict:
        return asdict(self)


def score_case(
    expected: Prescription,
    final: Prescription | None,
    raw: Prescription | None,
    transcript: Transcript,
) -> CaseMetrics:
    """Score one case. `final is None` means the run failed: it scores zero, it is not skipped."""
    exp_items = items(expected)
    pred_items = items(final) if final is not None else []
    exp_meds = _med_index(expected)
    pred_meds = _med_index(final) if final is not None else {}
    med_exact = sum(
        1
        for name, attrs in exp_meds.items()
        if name in pred_meds and all(pred_meds[name][a] == attrs[a] for a in EXACT_MED_ATTRS)
    )
    null_paths = _expected_null_paths(expected)
    recalled = 0
    counted = 0
    for slot, med, attr in null_paths:
        if final is None:
            counted += 1
            continue
        got = _lookup(final, slot, med, attr)
        if got is None:
            continue
        counted += 1
        recalled += got.value is None and got.flag is not None
    raw_s = safety_counts(raw, transcript)
    fin_s = safety_counts(final, transcript)
    return CaseMetrics(
        field_tp=match_count(pred_items, exp_items),
        field_pred=len(pred_items),
        field_expected=len(exp_items),
        med_exact=med_exact,
        med_pred=len(pred_meds),
        med_expected=len(exp_meds),
        missing_recalled=recalled,
        missing_expected=counted,
        raw_non_null=raw_s["non_null"],
        raw_invented=raw_s["invented"],
        raw_speaker_violations=raw_s["speaker_violations"],
        final_non_null=fin_s["non_null"],
        final_invented=fin_s["invented"],
        final_speaker_violations=fin_s["speaker_violations"],
    )


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(cases: list[CaseMetrics]) -> dict[str, float | int | None]:
    """Micro-averaged over all fields in all cases."""
    total = {k: sum(getattr(c, k) for c in cases) for k in CaseMetrics.__dataclass_fields__}
    return {
        "field_precision": _ratio(total["field_tp"], total["field_pred"]),
        "field_recall": _ratio(total["field_tp"], total["field_expected"]),
        "med_exact_recall": _ratio(total["med_exact"], total["med_expected"]),
        "med_exact_precision": _ratio(total["med_exact"], total["med_pred"]),
        "missing_info_recall": _ratio(total["missing_recalled"], total["missing_expected"]),
        "invented_rate_raw": _ratio(total["raw_invented"], total["raw_non_null"]),
        "invented_rate_final": _ratio(total["final_invented"], total["final_non_null"]),
        "speaker_violations_raw": total["raw_speaker_violations"],
        "speaker_violations_final": total["final_speaker_violations"],
    }
