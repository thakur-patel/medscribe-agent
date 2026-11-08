import pytest
from pydantic import ValidationError

from medscribe.schemas import Field, Medication, Prescription, Transcript


def test_transcript_accepts_bare_list_and_wrapped():
    lines = [{"id": "u01", "speaker": "doctor", "text": "Hello"}]
    assert Transcript.model_validate(lines).lines[0].id == "u01"
    assert Transcript.model_validate({"lines": lines}).lines[0].speaker == "doctor"


def test_transcript_rejects_bad_speaker_and_duplicate_ids():
    with pytest.raises(ValidationError):
        Transcript.model_validate([{"id": "u01", "speaker": "nurse", "text": "x"}])
    with pytest.raises(ValidationError):
        Transcript.model_validate(
            [
                {"id": "u01", "speaker": "doctor", "text": "a"},
                {"id": "u01", "speaker": "patient", "text": "b"},
            ]
        )


def test_null_value_requires_flag():
    with pytest.raises(ValidationError):
        Field[str](value=None, evidence=[], flag=None)
    assert Field[str](value=None, flag="unclear").flag == "unclear"
    assert Field[str](value="500 mg", evidence=["u03"], flag=None).value == "500 mg"


def test_field_rejects_unknown_flag_and_extra_keys():
    with pytest.raises(ValidationError):
        Field[str](value="x", evidence=["u01"], flag="guess")
    with pytest.raises(ValidationError):
        Field[str].model_validate({"value": "x", "evidence": [], "flag": None, "note": "?"})


def test_defaults_are_flagged_missing():
    rx = Prescription()
    assert rx.follow_up.value is None and rx.follow_up.flag == "missing"
    assert Medication().dose.flag == "missing"


def test_age_is_typed():
    rx = Prescription.model_validate(
        {"patient_details": {"age": {"value": "34", "evidence": ["u01"], "flag": None}}}
    )
    assert rx.patient_details.age.value == 34
    with pytest.raises(ValidationError):
        Prescription.model_validate(
            {"patient_details": {"age": {"value": "thirty", "evidence": [], "flag": None}}}
        )


def test_iter_fields_paths():
    rx = Prescription.model_validate(
        {
            "diagnosis": [{"value": "Viral fever", "evidence": ["u05"], "flag": None}],
            "medications": [
                {"medicine_name": {"value": "Paracetamol", "evidence": ["u06"], "flag": None}}
            ],
        }
    )
    paths = dict(rx.iter_fields())
    assert paths["diagnosis[0]"].value == "Viral fever"
    assert paths["medications[0].medicine_name"].value == "Paracetamol"
    assert paths["medications[0].dose"].flag == "missing"
    assert "patient_details.age" in paths
