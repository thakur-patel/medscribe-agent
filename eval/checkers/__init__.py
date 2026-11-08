"""Plain-code checkers. No LLM judges here: every metric is deterministic and explainable.

Metrics per case:
- field precision and recall: non-null predicted values vs non-null expected values,
  excluding stated negatives ("no cough"), which are scored on their own as negatives recall.
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

CHECKER_VERSION = "v2"

EXACT_MED_ATTRS = ("medicine_name", "strength", "dose", "frequency", "duration")
# Free-text slots compare by content-word overlap; everything else must match after normalising.
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
FUZZY_MED_ATTRS = {"instructions"}
# Overlap coefficient: shared content words / words in the shorter value.
FUZZY_THRESHOLD = 0.5
# Neighbouring sections that hold the same kind of item. A value in either one counts.
SECTION_GROUPS = {
    "chief_complaints": "symptoms",
    "associated_symptoms": "symptoms",
    "advice": "plan",
    "follow_up": "plan",
}
NEGATION_WORDS = {"no", "not", "none", "nil", "nkda", "denies", "denied", "without", "never"}
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "with", "from", "by",
    "is", "are", "was", "be", "been", "it", "this", "that", "if", "as", "any", "some", "since",
    "when", "while", "you", "your", "i", "my", "he", "she", "they", "take", "do", "does", "get",
    "known", "mild", "little", "very", "about", "around", "like", "other", "patient",
}  # fmt: skip
_SUFFIXES = ("ness", "ing", "ed", "es", "s")

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
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}  # fmt: skip
_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9}  # fmt: skip
_COMPOUND = re.compile(r"\b(" + "|".join(_TENS) + r")[- ](" + "|".join(_UNITS) + r")\b")
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
    "if needed": "sos",
    "as needed for fever": "sos",
    "when you have fever": "sos",
    "if fever": "sos",
    "only if fever": "sos",
}


def normalize(value: object, attr: str | None = None) -> str:
    text = str(value).lower()
    text = re.sub(r"(\d)\s*(mg|ml|g|mcg)\b", r"\1 \2", text)
    text = _COMPOUND.sub(lambda m: str(_TENS[m[1]] + _UNITS[m[2]]), text)  # forty-five -> 45
    text = re.sub(r"\b(" + "|".join(_TENS) + r")\b", lambda m: str(_TENS[m[1]]), text)
    text = re.sub(r"[^\w\s./%]", " ", text)
    words = [_NUMBER_WORDS.get(w, w) for w in text.split()]
    words = [_PLURALS.get(w, w) for w in words]
    text = " ".join(words).strip(" .")
    if attr == "frequency":
        text = _FREQUENCY.get(text, text)
    return text


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    words = (w for w in re.split(r"[^\w.]+", text) if w and w != ".")
    return {_stem(w) for w in words if w not in STOPWORDS and w not in NEGATION_WORDS}


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def is_negative(text: str) -> bool:
    """A stated negative such as 'no cough' or 'none': the first word is a negation."""
    first = text.split()[0] if text.split() else ""
    return first in NEGATION_WORDS


def values_match(a: str, b: str, fuzzy: bool) -> bool:
    if a == b:
        return True
    if not fuzzy:
        return False
    if is_negative(a) != is_negative(b):  # "no fever" never matches "fever"
        return False
    na, nb = _numbers(a), _numbers(b)
    if na and nb and na != nb:  # "after 3 days" never matches "after 7 days"
        return False
    ca, cb = content_words(a), content_words(b)
    if not ca or not cb:
        # Bare negatives like "none" match any negative in the same slot ("no known allergies").
        return is_negative(a) and is_negative(b)
    return len(ca & cb) / min(len(ca), len(cb)) >= FUZZY_THRESHOLD


@dataclass(frozen=True)
class Item:
    key: str  # the slot this value fills, e.g. "symptoms" or "medications.dose|paracetamol"
    value: str
    fuzzy: bool
    negative: bool = False


def items(rx: Prescription) -> list[Item]:
    out: list[Item] = []

    def add(key: str, f: Field, fuzzy: bool, attr: str | None = None, neg: bool = False) -> None:
        if f.value is None:
            return
        value = normalize(f.value, attr)
        out.append(Item(key, value, fuzzy, neg and is_negative(value)))

    for attr in ("name", "age", "gender"):
        add(f"patient_details.{attr}", getattr(rx.patient_details, attr), False)
    for section in SCALAR_SECTIONS:
        add(SECTION_GROUPS.get(section, section), getattr(rx, section), section in FUZZY_SECTIONS)
    for section in LIST_SECTIONS:
        for f in getattr(rx, section):
            add(SECTION_GROUPS.get(section, section), f, section in FUZZY_SECTIONS, neg=True)
    for med in rx.medications:
        name = normalize(med.medicine_name.value) if med.medicine_name.value else "?"
        for attr in MEDICATION_ATTRS:
            key = f"medications.{attr}|{name}"
            add(key, getattr(med, attr), attr in FUZZY_MED_ATTRS, attr)
    return out


def match_count(pred: list[Item], expected: list[Item]) -> int:
    """One-to-one matching: each predicted item can satisfy at most one expected item."""
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
    neg_tp: int
    neg_pred: int
    neg_expected: int
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
    all_exp = items(expected)
    all_pred = items(final) if final is not None else []
    exp_items = [i for i in all_exp if not i.negative]
    pred_items = [i for i in all_pred if not i.negative]
    exp_neg = [i for i in all_exp if i.negative]
    pred_neg = [i for i in all_pred if i.negative]
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
        neg_tp=match_count(pred_neg, exp_neg),
        neg_pred=len(pred_neg),
        neg_expected=len(exp_neg),
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
        "negatives_recall": _ratio(total["neg_tp"], total["neg_expected"]),
        "negatives_extra": total["neg_pred"] - total["neg_tp"],
        "med_exact_recall": _ratio(total["med_exact"], total["med_expected"]),
        "med_exact_precision": _ratio(total["med_exact"], total["med_pred"]),
        "missing_info_recall": _ratio(total["missing_recalled"], total["missing_expected"]),
        "invented_rate_raw": _ratio(total["raw_invented"], total["raw_non_null"]),
        "invented_rate_final": _ratio(total["final_invented"], total["final_non_null"]),
        "speaker_violations_raw": total["raw_speaker_violations"],
        "speaker_violations_final": total["final_speaker_violations"],
    }
