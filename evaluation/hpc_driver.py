"""Resumable entry points used by the Slurm experiment graph.

Every subcommand writes only below ``--results``. JSONL producers append one
record per completed unit and skip keys already present, which makes requeues
safe. The toy path is the executable golden oracle; benchmark paths call the
same evaluation modules used outside Slurm.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lethe.extract import extract

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "fixtures" / "shop"

TOY_RESULTS = {
    "single": {"localized_proposal": 13, "rounds": 13, "success": True},
    "parallel4": {"localized_proposal": None, "rounds": 6, "success": False},
    "team4": {"localized_proposal": 15, "rounds": 4, "success": True},
    "lethe": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_candidate_only": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_views": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_raw": {"localized_proposal": 15, "rounds": 4, "success": True},
    "lethe_fixprobe_only": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_spectrum_judge": {"localized_proposal": None, "rounds": 6, "success": False},
    "lethe_fixed2": {"localized_proposal": 15, "rounds": 4, "success": True},
    "lethe_fixed3": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_fixed4": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_fixed6": {"localized_proposal": 11, "rounds": 3, "success": True},
    "lethe_plateau": {"localized_proposal": 11, "rounds": 3, "success": True},
    "continue": {"localized_proposal": 15, "rounds": 4, "success": True},
    "forget": {"localized_proposal": 11, "rounds": 3, "success": True},
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(errors="replace").splitlines():
        try:
            out.add(str(json.loads(line)["record_key"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare(results: Path) -> None:
    pm = extract(TOY)
    source = ROOT / "evaluation" / "data" / "swe-bench_lite.parquet"
    data = results / "data"
    data.mkdir(parents=True, exist_ok=True)
    if source.exists() and not (data / source.name).exists():
        shutil.copy2(source, data / source.name)
    _write_json(
        results / "setup" / "verify.json",
        {
            "schema_version": "1.0",
            "toy_components": len(pm.components),
            "toy_functions": len(pm.functions),
            "toy_statements": len(pm.statements),
            "toy_source_hash": pm.source_hash,
            "swebench_parquet": (data / source.name).exists(),
        },
    )


def run_tests(results: Path) -> None:
    out = results / "setup" / "tests.json"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_core.py", "tests/test_shop_golden.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    record = {
        "schema_version": "1.0",
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    _write_json(out, record)
    if proc.returncode:
        raise SystemExit(proc.returncode)


def precompute(results: Path, dataset: str, n_items: int) -> None:
    if dataset == "toy":
        pm = extract(TOY)
        _write_json(
            results / "precompute" / dataset / "manifest.json",
            {
                "schema_version": "1.0",
                "dataset": dataset,
                "n_items": 1,
                "program_model": pm.model_dump(),
            },
        )
        return
    if dataset != "swebench-lite":
        raise ValueError(f"unsupported dataset: {dataset}")
    eval_data = results / "data"
    os.environ.update(
        LETHE_EVAL_DATA=str(eval_data),
        LETHE_PRECOMP=str(results / "precompute" / dataset),
        LETHE_SWEBENCH_PARQUET=str(eval_data / "swe-bench_lite.parquet"),
        LETHE_PYTHON=sys.executable,
    )
    from evaluation import precompute as pc
    from evaluation import swebench as swe

    bugs = [b for b in swe.load_dataset() if b.instance_id in swe.CANDIDATES][:n_items]
    selected: list[str] = []
    for bug in bugs:
        status = swe.setup(bug)
        if not status.get("ok"):
            continue
        doc = pc.precompute_bug(bug)
        if doc.get("faulty_statements"):
            selected.append(bug.instance_id)
    _write_json(Path(os.environ["LETHE_PRECOMP"]) / "_selected.json", selected)
    if not selected:
        raise RuntimeError("no SWE-bench bugs passed setup and precompute gates")


def run_study(
    results: Path,
    dataset: str,
    model_tier: str,
    study: str,
    configs: list[str],
    reps: int,
    n_items: int,
    output: Path,
) -> None:
    done = _keys(output)
    if dataset == "toy":
        for config in configs:
            expected = TOY_RESULTS.get(config, TOY_RESULTS["lethe"])
            for rep in range(1, reps + 1):
                key = f"toy|{model_tier}|{study}|{config}|{rep}"
                if key in done:
                    continue
                _append(
                    output,
                    {
                        "schema_version": "1.0",
                        "record_key": key,
                        "source": "toy_golden",
                        "reportable": False,
                        "dataset": dataset,
                        "model_tier": model_tier,
                        "study": study,
                        "config_name": config,
                        "replicate": rep,
                        "bug_id": "toy_shop",
                        "success": expected["success"],
                        "method_top1": float(expected["success"]),
                        "stmt_top1": float(expected["success"]),
                        "localized_proposal": expected["localized_proposal"],
                        "rounds": expected["rounds"],
                        "seconds": 0.0,
                    },
                )
        return

    eval_data = results / "data"
    precomp = results / "precompute" / dataset
    study_results = results / "runs" / model_tier / dataset / study
    os.environ.update(
        LETHE_EVAL_DATA=str(eval_data),
        LETHE_PRECOMP=str(precomp),
        LETHE_SELECTED=str(precomp / "_selected.json"),
        LETHE_RESULTS=str(study_results),
        LETHE_SWEBENCH_PARQUET=str(eval_data / "swe-bench_lite.parquet"),
        LETHE_LLM_MODEL=os.environ.get(f"LETHE_MODEL_{model_tier.upper()}", model_tier),
        LETHE_LLM_BASE=os.environ.get("LETHE_LLM_BASE", "http://127.0.0.1:11434/v1"),
    )
    from evaluation import run_grid

    old = sys.argv
    try:
        sys.argv = [
            "run_grid",
            "--configs",
            *configs,
            "--reps",
            str(reps),
            "--workers",
            os.environ.get("LETHE_CELL_WORKERS", "1"),
        ]
        run_grid.main()
    finally:
        sys.argv = old
    for metrics in sorted((study_results / "runs").glob("*/*/rep*/metrics.json")):
        row = json.loads(metrics.read_text())
        parts = metrics.relative_to(study_results / "runs").parts
        key = f"{parts[0]}|{model_tier}|{study}|{parts[1]}|{parts[2]}"
        if key in done:
            continue
        _append(
            output,
            {
                "schema_version": "1.0",
                "record_key": key,
                "source": "benchmark",
                "reportable": True,
                "dataset": dataset,
                "model_tier": model_tier,
                "study": study,
                "config_name": parts[1],
                "replicate": int(parts[2].removeprefix("rep")),
                "bug_id": parts[0],
                "success": bool(row["method"]["top1"]),
                "method_top1": row["method"]["top1"],
                "stmt_top1": row["stmt"]["top1"],
                "seconds": 0.0,
            },
        )


def analyze(results: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(results.glob("studies/**/*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    reportable = [r for r in rows if r.get("reportable")]
    summary = {
        "schema_version": "1.0",
        "records": len(rows),
        "reportable_records": len(reportable),
        "successes": sum(bool(r.get("success")) for r in reportable),
        "note": "Toy/smoke records are excluded from reportable counts.",
    }
    _write_json(results / "analysis" / "summary.json", summary)
    md = ["# LETHE experiment analysis", "", f"Records: {len(rows)}", f"Reportable: {len(reportable)}"]
    (results / "analysis" / "summary.md").write_text("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "test", "precompute", "run", "analyze"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", default="toy")
    parser.add_argument("--model-tier", default="mock")
    parser.add_argument("--study", default="main")
    parser.add_argument("--configs", default="single,parallel4,team4,lethe")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--n-items", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.command == "prepare":
        prepare(args.results)
    elif args.command == "test":
        run_tests(args.results)
    elif args.command == "precompute":
        precompute(args.results, args.dataset, args.n_items)
    elif args.command == "run":
        output = args.out or args.results / "studies" / args.study / f"{args.model_tier}__{args.dataset}.jsonl"
        run_study(
            args.results,
            args.dataset,
            args.model_tier,
            args.study,
            [x for x in args.configs.split(",") if x],
            args.reps,
            args.n_items,
            output,
        )
    else:
        analyze(args.results)
    print(f"{args.command} completed in {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
