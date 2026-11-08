import json

from eval.cases import load_cases
from eval.checkers import aggregate, normalize, score_case
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
