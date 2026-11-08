"""Eval runner.

    uv run python -m eval.run --config eval/configs/c0.yaml

Runs every case in the configured split, writes eval/results/<timestamp>-<name>/ with
cases.jsonl (one record per case), summary.json and summary.md, and prints the summary.
Failed runs are scored as zero and counted, never skipped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from eval.cases import Case, load_cases
from eval.checkers import CHECKER_VERSION, CaseMetrics, aggregate, score_case
from medscribe.config import REPO_ROOT, load_config
from medscribe.llm import LLM, OpenRouterLLM
from medscribe.pipeline import PIPELINE_VERSION, draft


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=20, method="inclusive")[-1]


def run_case(case: Case, llm: LLM, prompt: str) -> dict:
    result = draft(case.transcript, llm, prompt)
    metrics = score_case(
        case.expected, result.prescription, result.raw_prescription, case.transcript
    )
    return {
        "case_id": case.case_id,
        "tags": case.tags,
        **result.summary(),
        "metrics": metrics.to_dict(),
        "prescription": result.prescription.model_dump() if result.prescription else None,
        "raw_prescription": (
            result.raw_prescription.model_dump() if result.raw_prescription else None
        ),
    }


def summarize(records: list[dict], meta: dict) -> dict:
    metrics = aggregate([CaseMetrics(**r["metrics"]) for r in records])
    latencies = [r["latency_s"] for r in records]
    costs = [r["cost_usd"] for r in records]
    return {
        **meta,
        "cases": len(records),
        "failed_runs": sum(not r["ok"] for r in records),
        "retried_runs": sum(r["attempts"] > 1 for r in records),
        **metrics,
        "cost_usd_total": round(sum(costs), 6),
        "cost_usd_per_case": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "latency_s_median": round(statistics.median(latencies), 3) if latencies else 0.0,
        "latency_s_p95": round(_p95(latencies), 3),
    }


def _fmt(v: object) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if v < 1 or v != int(v) else f"{v:.0f}"
    return str(v)


def to_markdown(summary: dict, records: list[dict]) -> str:
    lines = [f"# Eval summary: {summary['name']} ({summary['split']})", ""]
    lines += ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()]
    lines += ["", "## Per case", ""]
    lines += [
        "| case | tags | ok | field P | field R | med exact | missing recall | "
        "invented raw | speaker raw | drops | latency s | cost USD |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r["metrics"]
        p = m["field_tp"] / m["field_pred"] if m["field_pred"] else None
        rc = m["field_tp"] / m["field_expected"] if m["field_expected"] else None
        mr = m["missing_recalled"] / m["missing_expected"] if m["missing_expected"] else None
        lines.append(
            f"| {r['case_id']} | {', '.join(r['tags'])} | {'yes' if r['ok'] else 'NO'} | "
            f"{_fmt(p)} | {_fmt(rc)} | {m['med_exact']}/{m['med_expected']} | {_fmt(mr)} | "
            f"{m['raw_invented']} | {m['raw_speaker_violations']} | {len(r['drops'])} | "
            f"{r['latency_s']:.2f} | {r['cost_usd']:.5f} |"
        )
    return "\n".join(lines) + "\n"


def run_eval(config_path: str | Path, llm: LLM | None = None, out_root: Path | None = None) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    app_cfg = load_config()
    prompt = app_cfg.prompt(cfg["prompt"])
    model = app_cfg.model(cfg.get("model", "default"))
    cases = load_cases(REPO_ROOT / cfg["golden_dir"], cfg["split"])
    if not cases:
        raise SystemExit(f"no cases found for split {cfg['split']!r}")
    if llm is None:
        llm = OpenRouterLLM.from_config(app_cfg, cfg.get("model", "default"))

    with ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 1))) as pool:
        records = list(pool.map(lambda c: run_case(c, llm, prompt.text), cases))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        out_root or REPO_ROOT / cfg.get("results_dir", "eval/results")
    ) / f"{stamp}-{cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": cfg["name"],
        "split": cfg["split"],
        "pipeline": PIPELINE_VERSION,
        "model": model.id,
        "prompt": f"{prompt.key}@{prompt.version}",
        "models_config_version": app_cfg.version,
        "checker_version": CHECKER_VERSION,
        "timestamp": stamp,
    }
    summary = summarize(records, meta)
    with (out_dir / "cases.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = to_markdown(summary, records)
    (out_dir / "summary.md").write_text(md)
    print(md)
    print(f"Results written to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MedScribe eval harness.")
    parser.add_argument(
        "--config", required=True, help="eval config YAML, e.g. eval/configs/c0.yaml"
    )
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv()
    run_eval(args.config)


if __name__ == "__main__":
    main()
