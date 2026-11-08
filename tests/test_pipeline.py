import json

import pytest

from medscribe.llm import FakeLLM
from medscribe.pipeline import draft, post_check
from medscribe.pipeline.c0 import parse
from medscribe.schemas import Prescription


def test_prompt_has_six_parts_and_placeholder(prompt_text):
    for part in ("# Role", "# Context", "# Task", "# Constraints", "# Format", "# Examples"):
        assert part in prompt_text
    assert "{{TRANSCRIPT}}" in prompt_text
    assert chr(0x2014) not in prompt_text  # no em dashes


def test_happy_path(transcript, rx_json, prompt_text):
    llm = FakeLLM([rx_json])
    result = draft(transcript, llm, prompt_text)
    assert result.ok and result.attempts == 1
    assert result.drops == []
    assert result.prescription.medications[0].strength.value == "500 mg"
    assert "[u05] doctor: Take paracetamol" in llm.calls[0][0]["content"]


def test_parse_accepts_code_fence(rx_json):
    assert parse(f"```json\n{rx_json}\n```").diagnosis[0].value == "Viral fever"


def test_retry_once_with_validation_errors(transcript, rx_json, prompt_text):
    llm = FakeLLM(['{"patient_details": {"age": {"value": null, "flag": null}}}', rx_json])
    result = draft(transcript, llm, prompt_text)
    assert result.ok and result.attempts == 2
    retry_msg = llm.calls[1][-1]["content"]
    assert "failed validation" in retry_msg and "null value must have a flag" in retry_msg


def test_second_failure_returns_errors(transcript, prompt_text):
    llm = FakeLLM(["not json", "still not json"])
    result = draft(transcript, llm, prompt_text)
    assert not result.ok and result.attempts == 2
    assert result.errors and result.prescription is None


def test_provider_error_is_reported(transcript, prompt_text):
    def boom(_):
        raise ConnectionError("offline")

    result = draft(transcript, FakeLLM(boom), prompt_text)
    assert not result.ok and "offline" in result.errors[0]


def test_post_check_drops_bad_evidence_and_patient_cited_meds(transcript, rx_dict):
    rx_dict["follow_up"]["evidence"] = ["u99"]  # unknown id
    rx_dict["advice"][0]["evidence"] = []  # no evidence
    rx_dict["diagnosis"].append({"value": "Typhoid", "evidence": ["u03"], "flag": None})
    rx_dict["medications"].append(
        {"medicine_name": {"value": "Azithromycin", "evidence": ["u03"], "flag": None}}
    )
    rx = Prescription.model_validate(rx_dict)
    checked, drops = post_check(rx, transcript)
    dropped = {d.path for d in drops}
    assert dropped == {"follow_up", "advice[0]", "diagnosis[1]", "medications[1].medicine_name"}
    for path, field in checked.iter_fields():
        if path in dropped:
            assert field.value is None and field.flag == "needs_doctor_check"
    assert checked.diagnosis[0].value == "Viral fever"  # good fields untouched
    assert rx.follow_up.value is not None  # original not mutated


def test_post_check_allows_patient_cited_history(transcript, rx_dict):
    rx_dict["allergies"] = [{"value": "Sulfa drugs", "evidence": ["u03"], "flag": None}]
    _, drops = post_check(Prescription.model_validate(rx_dict), transcript)
    assert drops == []


@pytest.mark.parametrize("bad", ["[]", json.dumps({"diagnosis": "flu"})])
def test_wrong_shapes_fail_validation(transcript, prompt_text, bad):
    result = draft(transcript, FakeLLM([bad, bad]), prompt_text)
    assert not result.ok
