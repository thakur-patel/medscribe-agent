"""FastAPI app: POST /draft, GET /draft/{id}/html, GET /draft/{id}/pdf. Drafts live in memory."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import ValidationError

from medscribe.config import AppConfig, load_config
from medscribe.llm import LLM, OpenRouterLLM
from medscribe.pipeline import PIPELINE_VERSION, DraftResult, draft
from medscribe.render import render_html, render_pdf
from medscribe.schemas import Transcript


def create_app(llm: LLM | None = None, config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    prompt = cfg.prompt("c0_extract")
    app = FastAPI(title="MedScribe Agent", version="0.1.0")
    drafts: dict[str, DraftResult] = {}
    state: dict[str, LLM | None] = {"llm": llm}

    def get_llm() -> LLM:
        if state["llm"] is None:
            state["llm"] = OpenRouterLLM.from_config(cfg)
        return state["llm"]

    def get_ok_draft(draft_id: str) -> DraftResult:
        result = drafts.get(draft_id)
        if result is None or not result.ok:
            raise HTTPException(404, "draft not found")
        return result

    @app.post("/draft")
    def create_draft(payload: Annotated[Any, Body()]) -> dict:
        try:
            transcript = Transcript.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False)) from exc
        result = draft(transcript, get_llm(), prompt.text)
        if not result.ok:
            raise HTTPException(502, {"errors": result.errors, "attempts": result.attempts})
        draft_id = uuid.uuid4().hex[:12]
        drafts[draft_id] = result
        return {
            "id": draft_id,
            "prompt_version": prompt.version,
            "config_version": cfg.version,
            "prescription": result.prescription.model_dump(),
            **result.summary(),
        }

    @app.get("/draft/{draft_id}/html", response_class=HTMLResponse)
    def draft_html(draft_id: str) -> str:
        return render_html(get_ok_draft(draft_id).prescription, draft_id, PIPELINE_VERSION)

    @app.get("/draft/{draft_id}/pdf")
    def draft_pdf(draft_id: str) -> Response:
        pdf = render_pdf(get_ok_draft(draft_id).prescription, draft_id, PIPELINE_VERSION)
        return Response(pdf, media_type="application/pdf")

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory medscribe.api:app_factory`."""
    from dotenv import load_dotenv

    load_dotenv()
    return create_app()
