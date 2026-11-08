# MedScribe Agent

A GenAI capstone for IIT Bombay (Generative AI: From Foundations to Autonomous Agents). MedScribe turns a doctor and patient outpatient consultation into a **draft ePrescription** that the doctor reviews and approves.

**Principle: the model proposes; code and the doctor decide.** Nothing on the prescription may be invented. Every field points to the transcript lines it came from, and anything not said is left empty and flagged.

> All data in this repository is synthetic. Never add real patient data.

This repository currently holds **C0**, the fixed baseline: contracts, a one-call pipeline, rendering, an API and the evaluation harness that will score every later version.

## Setup

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Then copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` (only needed for real model calls; tests run offline).

PDF output uses WeasyPrint, which needs Pango from the system: `brew install pango` on macOS, `apt install libpango-1.0-0 libpangoft2-1.0-0` on Debian or Ubuntu. On macOS, Homebrew's libraries must match Python's CPU architecture (Apple Silicon Python needs the `/opt/homebrew` install), and you may need `export DYLD_FALLBACK_LIBRARY_PATH=$(brew --prefix)/lib`. Without Pango, everything except PDF works and the two PDF tests are skipped.

## Run the tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Tests never call a model. The LLM sits behind a small interface (`medscribe.llm.LLM`) and tests use `FakeLLM`, which returns fixtures.

## Run the app

```bash
uv run uvicorn --factory medscribe.api:app_factory --reload
```

| Endpoint | What it does |
|---|---|
| `POST /draft` | Transcript JSON in (a list of lines, or `{"lines": [...]}`), Prescription JSON out, plus an `id`, the post-check drops, cost and latency |
| `GET /draft/{id}/html` | The draft as HTML, flagged fields highlighted |
| `GET /draft/{id}/pdf` | The same as PDF |

Try it with the test fixture:

```bash
curl -s localhost:8000/draft -H 'content-type: application/json' \
  -d @tests/fixtures/transcript_fever.json | python -m json.tool
```

Drafts are kept in memory, so they disappear when the server restarts.

## Run the eval

```bash
uv run python -m eval.run --config eval/configs/c0.yaml
```

This runs every case in `eval/golden/dev/` and writes `eval/results/<timestamp>-c0/` with `cases.jsonl` (one record per case, including raw and final output), `summary.json` and `summary.md`. Failed runs score zero and are counted, never skipped.

| Metric | Meaning |
|---|---|
| `field_precision`, `field_recall` | Non-null predicted values vs expected values. Free-text sections match on token overlap (Jaccard 0.5 or more); names, numbers and medication fields must match after normalising (units, number words, `TDS` = `three times a day`). |
| `med_exact_recall`, `med_exact_precision` | A medication counts only if name, strength, dose, frequency and duration all match. |
| `missing_info_recall` | Of the fields that were never stated, how many came back null and flagged instead of filled. |
| `invented_rate_raw`, `invented_rate_final` | Share of non-null values with no evidence or with line IDs that do not exist. Raw is what the model wrote; final is after the post-check. |
| `speaker_violations_raw`, `speaker_violations_final` | Diagnosis or medication fields citing a patient line. |
| `cost_usd_*`, `latency_s_median`, `latency_s_p95` | Per case. Cost is OpenRouter's reported cost, or computed from the prices in `configs/models.yaml`. |

### Golden cases

- `eval/golden/dev/seed_cases.jsonl`: 10 synthetic seed cases (4 routine visits, 2 with a dose or frequency never stated, a sound-alike drug, a patient-reported allergy, a patient asking for a drug the doctor does not give, a patient guessing a diagnosis the doctor does not confirm). They are a **draft that needs clinical review** by the team's healthcare member.
- `eval/golden/heldout/`: written by people only and never read during development. See its README.

## What C0 is

```
transcript JSON
  -> one LLM call (prompts/c0_extract.v1.md)
  -> Pydantic validation (one retry with the errors appended)
  -> post-check in plain code
  -> Prescription JSON -> HTML / PDF
```

- **Contracts** (`src/medscribe/schemas/`): every leaf is a `Field` with `value`, `evidence` (line IDs) and `flag` (`missing`, `unclear`, `needs_doctor_check`). A null value must have a flag.
- **Post-check** (`src/medscribe/pipeline/post_check.py`): drops any non-null field with no evidence or unknown line IDs, and any diagnosis or medication field that cites a non-doctor line. Dropped fields become null with flag `needs_doctor_check`, and the drop is logged with the original value.
- **Config** (`configs/models.yaml`): model IDs and prompt versions in one versioned file. Default model `openai/gpt-4.1-mini` through OpenRouter's OpenAI-compatible API.
- **Prompts** (`prompts/`): plain files with six parts: role, context, task, constraints, format, examples.

### Decisions worth knowing

- **Strict speaker rule.** A diagnosis or medication field is dropped if *any* cited line is not a doctor line, even if a doctor line is also cited. This is conservative: a patient saying "yes" to a dose can cause a drop. Worth revisiting once we see real failures.
- **Route is not inferred.** If the doctor says "tablet" but not "orally", the expected route is null and flagged. This keeps "never invent" strict; the healthcare member may prefer to allow obvious routes.
- **List sections.** Sections like `diagnosis` and `advice` are lists of fields, one per item, so each item has its own evidence. An empty list means nothing was said; missing-info recall only covers single-value fields and the attributes of expected medications.
- **Missing fields in golden cases** are simply left out of `expected_prescription`; the schema fills them as null with flag `missing`.

### C0 baseline (2026-11-08)

`openai/gpt-4.1-mini`, prompt `c0_extract@v1`, 10 dev seed cases (still a draft pending clinical review).

| Metric | C0 |
|---|---|
| Failed runs / retries | 0 / 0 |
| Field precision / recall | 0.777 / 0.804 |
| Medication exact match | 0.750 (12 of 16) |
| Missing-info recall | 0.979 |
| Invented rate (raw and final) | 0.000 |
| Speaker violations (raw and final) | 0 |
| Cost per case | $0.0018 |
| Latency median / p95 | 6.1 s / 7.8 s |

What the misses are:

- **Safety held on every case.** No invented values, no patient-requested drug (dev-009), no confirmed dengue (dev-010), no hydralazine mix-up (dev-007), and unstated doses and frequencies came back null and flagged (dev-005, dev-006).
- **Real model errors: merged fields.** 3 of the 4 medication misses are values put in the wrong attribute, for example "twice a day before food" as the frequency or "1 glass after every loose motion" as the dose. One medicine got its purpose ("helps itching") as its instruction.
- **Most field misses are the scoring, not the model.** Same meaning in other words ("sometimes acidity" vs "occasional acidity", "none" vs "no known drug allergies"), the same item in a neighbouring section (follow-up written under advice, a symptom under chief complaints instead of associated symptoms), and stated negatives ("no cough") counted as extra items.

## What comes next

C0 is the baseline every later version is measured against. Planned, not built yet:

1. A LangGraph supervisor with 4 specialist agents.
2. 4 FastMCP servers: formulary, patient records, rules, prescription.
3. Hybrid retrieval (for example, formulary lookups to catch sound-alike drugs).
4. A doctor approval gate.
5. Whisper speech-to-text so the input can be audio.
6. Docker Compose.

Each step gets its own eval config and is compared with C0 on the same dev cases.

## Layout

```
configs/models.yaml        model IDs, provider, prompt versions
prompts/                   prompt files
src/medscribe/
  schemas/                 Transcript, Prescription, Field
  llm/                     LLM interface, OpenRouterLLM, FakeLLM
  pipeline/                C0 pipeline and post-check
  render/                  Jinja2 HTML and WeasyPrint PDF
  api/                     FastAPI app
eval/
  golden/{dev,heldout}/    golden cases (JSONL)
  checkers/                metrics in plain code
  configs/                 eval configs (c0.yaml)
  run.py                   runner
tests/                     offline tests and fixtures
```
