"""Render a Prescription to HTML (Jinja2) and PDF (WeasyPrint). Flagged fields are highlighted."""

from __future__ import annotations

from jinja2 import Environment, PackageLoader, select_autoescape

from medscribe.schemas import Prescription

_env = Environment(
    loader=PackageLoader("medscribe.render", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
)


def render_html(rx: Prescription, draft_id: str = "-", pipeline: str = "c0") -> str:
    return _env.get_template("prescription.html.j2").render(
        rx=rx, draft_id=draft_id, pipeline=pipeline
    )


def render_pdf(rx: Prescription, draft_id: str = "-", pipeline: str = "c0") -> bytes:
    # Imported lazily: WeasyPrint needs system libraries (pango); HTML works without them.
    from weasyprint import HTML

    return HTML(string=render_html(rx, draft_id, pipeline)).write_pdf()
