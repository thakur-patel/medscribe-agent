import json

from eval.cases import load_cases
from eval.checkers import aggregate, normalize, score_case, values_match
from eval.run import run_eval
from medscribe.config import REPO_ROOT
from medscribe.llm import FakeLLM
from medscribe.schemas import Prescription

CASES = load_cases(REPO_ROOT / "eval/golden", "dev")


def test_seed_cases_cover_the_required_mix():
    tags = [t for c in CASES for t in c.tags]
    assert len(CASES) == 10
    assert sum("routine" in c.tags for c in CASES) == 4
    for tag in ("sound-alike", "patient-requests-drug", "patient-diagnosis-guess", "allergy"):
        assert tag in tags
    assert sum(any(t.startswith("missing-") for t in c.tags) for c in CASES) == 2
    header = (REPO_ROOT / "eval/golden/dev/seed_cases.jsonl").read_text().splitlines()[0]
    assert header.startswith("# DRAFT: needs clinical review")


def test_normalize():
    assert normalize("500mg") == normalize("500 mg")
    assert normalize("Three days") == "3 day" == normalize("3 days")
    assert normalize("TDS", "frequency") == normalize("three times a day", "frequency")
    assert normalize("Once at night", "frequency") == normalize("HS", "frequency")
    assert normalize("walk forty-five minutes") == "walk 45 minutes"
    assert normalize("thirty minutes") == "30 minutes"
    assert normalize("When you have fever", "frequency") == normalize("SOS", "frequency")


def test_perfect_prediction_scores_perfectly():
    for case in CASES:
        m = score_case(case.expected, case.expected, case.expected, case.transcript)
        assert m.field_tp == m.field_expected == m.field_pred, case.case_id
        assert m.med_exact == m.med_expected
        assert m.missing_recalled == m.missing_expected
        assert m.raw_invented == 0 and m.raw_speaker_violations == 0, case.case_id


def test_bad_prediction_is_caught():
    case = next(c for c in CASES if "patient-requests-drug" in c.tags)
    bad = case.expected.model_dump()
    bad["medications"][0]["frequency"] = {"value": "BD", "evidence": ["u08"], "flag": None}
    bad["medications"].append(
        {"medicine_name": {"value": "Azithromycin", "evidence": ["u04"], "flag": None}}
    )
    bad["follow_up"] = {"value": "1 week", "evidence": [], "flag": None}  # invented
    pred = Prescription.model_validate(bad)
    m = score_case(case.expected, pred, pred, case.transcript)
    assert m.med_exact == m.med_expected - 1
    assert m.raw_speaker_violations == 1 and m.raw_invented == 1
    assert m.field_tp < m.field_pred


def test_failed_run_counts_as_zero():
    case = CASES[0]
    m = score_case(case.expected, None, None, case.transcript)
    agg = aggregate([m])
    assert agg["field_recall"] == 0 and agg["missing_info_recall"] == 0


def _oracle(messages):
    prompt = messages[0]["content"]
    for case in CASES:
        if case.transcript.to_prompt_text() in prompt:
            return case.expected.model_dump_json()
    raise AssertionError("unknown transcript")


def test_runner_end_to_end_offline(tmp_path):
    out = run_eval(REPO_ROOT / "eval/configs/c0.yaml", llm=FakeLLM(_oracle), out_root=tmp_path)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cases"] == 10 and summary["failed_runs"] == 0
    assert summary["field_precision"] == summary["field_recall"] == 1.0
    assert summary["med_exact_recall"] == 1.0 and summary["missing_info_recall"] == 1.0
    assert summary["invented_rate_raw"] == 0 and summary["speaker_violations_final"] == 0
    assert len((out / "cases.jsonl").read_text().splitlines()) == 10
    assert "| dev-010 |" in (out / "summary.md").read_text()


def test_runner_counts_failures(tmp_path):
    out = run_eval(
        REPO_ROOT / "eval/configs/c0.yaml", llm=FakeLLM(lambda _: "oops"), out_root=tmp_path
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cases"] == 10 and summary["failed_runs"] == 10
    assert summary["field_recall"] == 0


# Checker v2: fair on wording, strict on meaning.
FAIR = [
    ("sometimes acidity", "occasional acidity"),
    ("throat pain", "sore throat"),
    ("a little breathless when climbing stairs", "breathlessness on climbing stairs"),
    ("diabetes", "diabetes follow up"),
    ("none", "no known drug allergies"),
]
STRICT = [
    ("no fever", "fever"),  # negation must agree
    ("after 3 day", "after 7 day"),  # numbers must agree
    ("bp 152/96 mmhg", "bp 128/82 mmhg"),
    ("chest x ray", "complete blood count"),
    ("no cough", "no fever"),
]


def test_values_match_is_fair_on_wording():
    for a, b in FAIR:
        assert values_match(normalize(a), normalize(b), fuzzy=True), (a, b)


def test_values_match_is_strict_on_meaning():
    for a, b in STRICT:
        assert not values_match(normalize(a), normalize(b), fuzzy=True), (a, b)


def test_medication_fields_stay_exact():
    exp = Prescription.model_validate(
        {
            "medications": [
                {
                    "medicine_name": {"value": "Ondansetron", "evidence": ["u01"], "flag": None},
                    "frequency": {"value": "Twice a day", "evidence": ["u01"], "flag": None},
                }
            ]
        }
    )
    pred = exp.model_copy(deep=True)
    pred.medications[0].frequency.value = "Twice a day before food"
    m = score_case(exp, pred, pred, CASES[0].transcript)
    assert m.med_exact == 0 and m.field_tp == 1


def test_neighbouring_sections_and_negatives():
    case = CASES[0]  # dev-001
    pred = case.expected.model_copy(deep=True)
    # move a symptom to chief complaints and follow-up text into advice
    pred.chief_complaints.append(pred.associated_symptoms.pop())
    pred.advice.append(pred.follow_up.model_copy())
    pred.follow_up.value, pred.follow_up.evidence, pred.follow_up.flag = None, [], "missing"
    # add an extra stated negative: scored apart from field precision
    pred.associated_symptoms.append(
        Prescription.model_validate(
            {"advice": [{"value": "No cough", "evidence": ["u06"], "flag": None}]}
        ).advice[0]
    )
    m = score_case(case.expected, pred, pred, case.transcript)
    assert m.field_tp == m.field_expected == m.field_pred
    assert m.neg_tp == m.neg_expected == 1 and m.neg_pred == 2


def test_rescore_matches_a_fresh_run(tmp_path):
    from eval.rescore import rescore

    run = run_eval(REPO_ROOT / "eval/configs/c0.yaml", llm=FakeLLM(_oracle), out_root=tmp_path)
    again = json.loads((rescore(run) / "summary.json").read_text())
    first = json.loads((run / "summary.json").read_text())
    for key in ("field_precision", "field_recall", "med_exact_recall", "negatives_recall"):
        assert again[key] == first[key] == 1.0
    assert again["rescored_from"] == run.name and again["checker_version"] == "v2"
