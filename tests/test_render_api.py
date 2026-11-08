import pytest
from fastapi.testclient import TestClient

from medscribe.api import create_app
from medscribe.llm import FakeLLM
from medscribe.render import render_html, render_pdf
from medscribe.schemas import Prescription
from tests.conftest import weasyprint_available

needs_weasyprint = pytest.mark.skipif(
    not weasyprint_available(), reason="WeasyPrint system libraries (pango) not installed"
)


def test_html_highlights_flagged_fields(rx_json):
    html = render_html(Prescription.model_validate_json(rx_json), "abc")
    assert "Paracetamol" in html and "[u05]" in html
    # gender and route are flagged missing in the fixture
    assert html.count('class="flagged"') == 2
    assert "missing" in html and "DRAFT for doctor review" in html


@needs_weasyprint
def test_pdf_renders(rx_json):
    pdf = render_pdf(Prescription.model_validate_json(rx_json), "abc")
    assert pdf[:4] == b"%PDF"


@pytest.fixture
def client(rx_json):
    return TestClient(create_app(llm=FakeLLM(lambda _: rx_json)))


def test_post_draft_then_html(client, transcript):
    resp = client.post("/draft", json=transcript.model_dump()["lines"])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] and body["prescription"]["diagnosis"][0]["value"] == "Viral fever"
    html = client.get(f"/draft/{body['id']}/html")
    assert html.status_code == 200 and 'class="flagged"' in html.text


@needs_weasyprint
def test_get_pdf(client, transcript):
    draft_id = client.post("/draft", json=transcript.model_dump()).json()["id"]
    pdf = client.get(f"/draft/{draft_id}/pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_bad_transcript_and_unknown_draft(client):
    assert client.post("/draft", json=[{"id": "u01", "speaker": "nurse"}]).status_code == 422
    assert client.get("/draft/nope/html").status_code == 404


def test_model_failure_returns_502(transcript):
    app = create_app(llm=FakeLLM(["bad", "bad"]))
    resp = TestClient(app).post("/draft", json=transcript.model_dump())
    assert resp.status_code == 502 and resp.json()["detail"]["errors"]
