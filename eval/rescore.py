"""Re-score saved eval results with the current checkers. No model calls.

    uv run python -m eval.rescore eval/results/<run-dir>

Writes a sibling folder <run-dir>-rescored-<checker version> with the same layout as a normal
run, so model outputs stay fixed while the scoring changes. Use it whenever the checkers or the
golden cases change, to keep comparisons between versions fair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.cases import load_cases
from eval.checkers import CHECKER_VERSION, score_case
from eval.run import summarize, to_markdown
from medscribe.config import REPO_ROOT
from medscribe.schemas import Prescription

META_KEYS = ("name", "split", "pipeline", "model", "prompt", "models_config_version", "timestamp")


def _rx(data: dict | None) -> Prescription | None:
    return Prescription.model_validate(data) if data is not None else None


def rescore(run_dir: str | Path, golden_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    old = json.loads((run_dir / "summary.json").read_text())
    cases = {
        c.case_id: c for c in load_cases(golden_dir or REPO_ROOT / "eval/golden", old["split"])
    }
    records = []
    for line in (run_dir / "cases.jsonl").read_text().splitlines():
        r = json.loads(line)
        case = cases[r["case_id"]]
        metrics = score_case(
            case.expected, _rx(r["prescription"]), _rx(r["raw_prescription"]), case.transcript
        )
        records.append({**r, "metrics": metrics.to_dict()})
    missing = set(cases) - {r["case_id"] for r in records}
    if missing:
        raise SystemExit(f"golden cases not in this run: {sorted(missing)}; run the eval again")

    out_dir = run_dir.with_name(f"{run_dir.name}-rescored-{CHECKER_VERSION}")
    out_dir.mkdir(exist_ok=True)
    meta = {k: old[k] for k in META_KEYS if k in old}
    meta |= {"checker_version": CHECKER_VERSION, "rescored_from": run_dir.name}
    summary = summarize(records, meta)
    with (out_dir / "cases.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = to_markdown(summary, records)
    (out_dir / "summary.md").write_text(md)
    print(md)
    print(f"Re-scored results written to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score saved eval results.")
    parser.add_argument("run_dir", help="an eval/results/<run> folder")
    rescore(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
