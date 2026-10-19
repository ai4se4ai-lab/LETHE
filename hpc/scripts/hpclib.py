"""hpc/scripts/hpclib.py — shared helpers for submit.py, run_stage.py, audit.py and collect_results.py.

Standard library only (Python >= 3.8), so it is safe to import on a login node.

Concepts
  RUN_ID      one execution of the experiment graph (am-YYYYmmdd-HHMMSS, or *-smoke-*)
  stage       one node of the graph = one sbatch submission (a job array when it has a matrix)
  cell        one element of a stage's matrix (= one array task); stages without a matrix have one cell
  step        one command (or built-in action) inside a stage, run in order for every cell

Layout
  hpc/logs/<RUN_ID>/            (on /project, persistent)  config/ (frozen), jobs.tsv, steps.tsv, jobs_finished.tsv,
                                 slurm/, steps/<stage>/<cell>/<step>.log, gpu/, env/, audit.txt|json, results/ (collected)
  $WORK/runs/<RUN_ID>/          (on scratch, fast)          results/ ($RESULTS), state/<stage>/<cell>/<step>.done
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
from pathlib import Path

HPC = Path(os.environ.get("HPC_DIR") or Path(__file__).resolve().parents[1])
REPO = HPC.parent
_PH = re.compile(r"(?<![\$\\])\{([A-Za-z_][A-Za-z0-9_]*)\}")


# ── configuration ────────────────────────────────────────────────────────────
def source_env(path: Path) -> dict:
    """Variables of a bash env file, evaluated by bash exactly as the jobs see them (only the file's own names)."""
    if not path.exists():
        return {}
    script = f'set -a; source "{path}" >/dev/null 2>&1; env -0'
    base = subprocess.run(["bash", "-c", "env -0"], capture_output=True, check=False).stdout.decode(errors="replace")
    out = subprocess.run(["bash", "-c", script], capture_output=True, check=False).stdout.decode(errors="replace")
    parse = lambda s: dict(kv.split("=", 1) for kv in s.split("\0") if "=" in kv)  # noqa: E731
    before, after = parse(base), parse(out)
    names = set(re.findall(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=", path.read_text(), re.M))
    return {k: v for k, v in after.items() if k in names or before.get(k) != v}


def log_root(run_id: str) -> Path:
    return HPC / "logs" / run_id


def config_dir(run_id: str | None) -> Path:
    """The run's frozen config if it exists, else the live hpc/config/."""
    if run_id and (log_root(run_id) / "config" / "cluster.env").exists():
        return log_root(run_id) / "config"
    return HPC / "config"


def cluster(run_id: str | None = None) -> dict:
    return source_env(config_dir(run_id) / "cluster.env")


def work_dir(c: dict) -> Path:
    return Path(c.get("WORK") or Path(os.environ.get("SCRATCH", str(Path.home() / "scratch"))) / "hpc_runs")


def run_dir(run_id: str, c: dict | None = None) -> Path:
    return work_dir(c if c is not None else cluster(run_id)) / "runs" / run_id


def results_dir(run_id: str, c: dict | None = None) -> Path:
    return Path(os.environ.get("RESULTS") or run_dir(run_id, c) / "results")


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", str(s))


# ── experiments.json ─────────────────────────────────────────────────────────
def load_experiments(path: Path) -> dict:
    return json.loads(path.read_text())


def _cells_of(stage: dict, by_name: dict, limit: int | None) -> list[dict]:
    if stage.get("matrix_from"):
        return _cells_of(by_name[stage["matrix_from"]], by_name, limit)
    mat = stage.get("matrix") or {}
    if not mat:
        return [{}]
    keys = list(mat)
    cells = [dict(zip(keys, vals)) for vals in itertools.product(*(mat[k] if isinstance(mat[k], list) else [mat[k]] for k in keys))]
    return cells[:limit] if limit else cells


def resolve(exp: dict, smoke: bool) -> dict:
    """Apply smoke overrides and expand every stage's matrix into an explicit list of cells."""
    exp = json.loads(json.dumps(exp))
    sm = exp.get("smoke", {}) if smoke else {}
    exp["vars"] = {**exp.get("vars", {}), **sm.get("vars", {})}
    exp["is_smoke"] = smoke
    by_name = {s["name"]: s for s in exp["stages"]}
    for s in exp["stages"]:
        s["cells"] = _cells_of(s, by_name, sm.get("matrix_limit"))
        if smoke:
            s.setdefault("resources", {})["time"] = sm.get("time", "00:30:00")
            if sm.get("array_throttle"):
                s["array_throttle"] = sm["array_throttle"]
    return exp


def validate(exp: dict) -> list[str]:
    """Problems that would make the graph wrong or break the GPU policy (empty list = fine)."""
    errs = []
    names = [s.get("name") for s in exp.get("stages", [])]
    if len(names) != len(set(names)):
        errs.append("stage names must be unique")
    seen = set()
    for s in exp.get("stages", []):
        n = s.get("name")
        if not n or not re.fullmatch(r"[A-Za-z0-9_-]+", n):
            errs.append(f"stage name {n!r}: use letters, digits, _ and -")
        for d in s.get("depends", []):
            if d.get("stage") not in names:
                errs.append(f"{n}: depends on unknown stage {d.get('stage')!r}")
            elif d["stage"] not in seen:
                errs.append(f"{n}: depends on {d['stage']!r}, which must be listed earlier (stages are submitted in order)")
            if d.get("type", "afterok") not in ("afterok", "afterany", "afternotok", "aftercorr"):
                errs.append(f"{n}: dependency type {d.get('type')!r} unknown")
            if d.get("type") == "aftercorr" and d.get("stage") in names:
                other = next(x for x in exp["stages"] if x["name"] == d["stage"])
                if "cells" in s and len(s["cells"]) != len(other.get("cells", [])):
                    errs.append(f"{n}: aftercorr needs the same number of cells as {d['stage']} (use matrix_from)")
        if s.get("matrix_from") and s["matrix_from"] not in names:
            errs.append(f"{n}: matrix_from unknown stage {s['matrix_from']!r}")
        r = s.get("resources", {})
        if exp.get("require_gpu", True) and not s.get("cpu_only") and not (r.get("gpus") or r.get("gpus_per_node")):
            errs.append(f"{n}: no GPU requested; require_gpu is on (add resources.gpus, or mark the stage \"cpu_only\": true)")
        if not s.get("steps"):
            errs.append(f"{n}: no steps")
        for st in s.get("steps", []):
            if not st.get("name") or not (st.get("cmd") or st.get("builtin")):
                errs.append(f"{n}: every step needs a name and a cmd or builtin")
        seen.add(n)
    return errs


def stage(exp: dict, name: str) -> dict:
    for s in exp["stages"]:
        if s["name"] == name:
            return s
    raise SystemExit(f"unknown stage {name!r}; stages: {[s['name'] for s in exp['stages']]}")


def cell_key(cell: dict, idx: int) -> str:
    return "__".join(safe(v) for v in cell.values()) if cell else "all"


def cell_label(cell: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in cell.items()) or "-"


# ── placeholders ─────────────────────────────────────────────────────────────
def fill(text: str, ctx: dict) -> str:
    """Replace {NAME} placeholders; values may themselves contain placeholders (resolved up to 5 levels deep)."""
    def rep(m):
        k = m.group(1)
        if k not in ctx:
            raise KeyError(f"unknown placeholder {{{k}}} in: {text[:120]}  (known: {', '.join(sorted(ctx)[:40])} ...)")
        return str(ctx[k])
    out = str(text)
    for _ in range(5):
        new = _PH.sub(rep, out)
        if new == out:
            break
        out = new
    return out


def fill_number(expr: str, ctx: dict) -> float | None:
    """'{N_ITEMS}*3' -> 3000.0 (only digits and + - * / ( ) . after filling)."""
    try:
        s = fill(str(expr), ctx)
        if not re.fullmatch(r"[0-9+\-*/(). ]+", s):
            return None
        return float(eval(s, {"__builtins__": {}}, {}))  # noqa: S307 — restricted to arithmetic above
    except Exception:  # noqa: BLE001
        return None


def context(exp: dict, c: dict, run_id: str, stage_name: str, cell: dict, cell_idx: int) -> dict:
    rd = run_dir(run_id, c)
    ctx = {k: v for k, v in c.items() if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)}
    ctx.update({k: str(v) for k, v in exp.get("vars", {}).items()})
    ctx.update({k: str(v) for k, v in cell.items()})
    ctx.update({
        "RUN_ID": run_id, "RUN_DIR": str(rd), "RESULTS": str(Path(os.environ.get("RESULTS") or rd / "results")),
        "REPO": str(REPO), "HPC": str(HPC), "LOGS": str(log_root(run_id)), "STAGE": stage_name,
        "CELL": cell_key(cell, cell_idx), "CELL_ID": str(cell_idx), "SMOKE": "1" if exp.get("is_smoke") else "0",
        "NNODES": os.environ.get("SLURM_NNODES", "1"), "NTASKS": os.environ.get("SLURM_NTASKS", "1"),
        "JOB_ID": os.environ.get("SLURM_JOB_ID", "0"),
    })
    return ctx


# ── ledgers ──────────────────────────────────────────────────────────────────
LEDGER_HEADER = "time\tjob\ttask\trestart\tstage\tcell\tstep\tstart\tend\tseconds\trc\tnote\tcommand\n"


def read_tsv(p: Path) -> list[list[str]]:
    if not p.exists():
        return []
    return [line.split("\t") for line in p.read_text(errors="replace").splitlines() if line.strip()]


def latest_steps(run_id: str) -> dict:
    """{(stage, cell, step): row dict} — the latest attempt of every step."""
    rows = read_tsv(log_root(run_id) / "steps.tsv")
    if not rows:
        return {}
    head = rows[0]
    out = {}
    for r in rows[1:]:
        d = dict(zip(head, r))
        out[(d.get("stage"), d.get("cell"), d.get("step"))] = d
    return out


def count_records(p: Path) -> int:
    """Lines of a .jsonl/.csv/.tsv (csv/tsv minus header); 0 if missing."""
    if not p.exists() or not p.is_file():
        return 0
    n = sum(1 for line in p.open(errors="replace") if line.strip())
    return max(n - 1, 0) if p.suffix in (".csv", ".tsv") else n
