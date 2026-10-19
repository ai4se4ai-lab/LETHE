"""Experiment runner: bugs x configs x replicates, resumable, parallel over
bugs. Judge cost is shared-cached per bug, so configs are cheap to compare
fairly (plan §5.6.5)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lethe.core import ProgramModel, run_proc
from lethe import extract
from lethe.judge import BugEnv
from lethe.harness import Harness, PRESET_CONFIGS
from evaluation import swebench as swe
from evaluation import metrics

RESULTS = Path(
    os.environ.get("LETHE_RESULTS", str(Path(__file__).resolve().parent / "results" / "main"))
).resolve()
PRECOMP = Path(
    os.environ.get(
        "LETHE_PRECOMP", str(Path(__file__).resolve().parent / "data" / "precomputed")
    )
).resolve()
SELECTED = Path(os.environ.get("LETHE_SELECTED", str(PRECOMP / "_selected.json"))).resolve()

PRINT_LOCK = threading.Lock()

CONFIGS = ["single", "parallel4", "team4", "lethe", "lethe_candidate_only"]


def log(msg: str):
    with PRINT_LOCK:
        print(f"[runner] {msg}", flush=True)


def _checkout_hash(b) -> str:
    rc, so, se = run_proc(["git", "rev-parse", "HEAD"], cwd=swe.judge_dir(b),
                          env=dict(os.environ))
    return so.strip() or "unknown"


def build_bug_env(b, doc) -> BugEnv:
    spectrum = {el: sc for el, sc, rk in doc["suspicion_formulas"]["ochiai"]}
    return BugEnv(
        checkout=swe.checkout_dir(b),
        judge_home=swe.judge_dir(b),
        python=str(swe.venv_python(b)),
        fail_to_pass=b.fail_to_pass,
        pass_to_pass=doc["pass_to_pass_sample"],
        spectrum=spectrum,
        n_tests=doc["n_tests"],
        base_pass=doc["base_pass"],
        checkout_hash=_checkout_hash(b),
        cache_dir=RESULTS / "cache" / b.instance_id / "probes",
        test_timeout_s=240,
        p2p_sample=None,
        pass_to_pass_sampled=doc["pass_to_pass_sample"],
    )


def build_model_view(doc) -> dict:
    return {
        "functions": [{"id": h["id"], "file": h["id"].split("::")[0], "name": h.split("::")[-1],
                       "class": None, "lineno": 0, "end_lineno": 0, "calls": []}
                      for h in []],
        "functions_doc": doc["model"],
    }


def cell_dir(bug: str, cfg: str, rep: int) -> Path:
    return RESULTS / "runs" / bug / cfg / f"rep{rep}"


def cell_is_done(cell: Path) -> bool:
    if not cell.exists():
        return False
    ev = cell / "events.jsonl"
    if not ev.exists():
        return False
    last = None
    with open(ev) as f:
        for line in f:
            last = line
    if not last:
        return False
    try:
        return json.loads(last).get("event") == "run_end" and \
            json.loads(last).get("status") == "completed"
    except json.JSONDecodeError:
        return False


def _expand_method_to_statements(h: dict, doc: dict) -> list[str]:
    if h["granularity"] == "statement":
        return [h["element"]]
    fn = h["element"]
    scores = {el: sc for el, sc, rk in doc["suspicion_formulas"]["ochiai"]}
    sts = [s for s in doc["statements"] if s["func_id"] == fn]
    sts.sort(key=lambda s: (-scores.get(s["id"], 0.0), s["line"]))
    return [s["id"] for s in sts]


def cell_metrics(b, doc, cell: Path) -> dict:
    res = json.loads((cell / "result.json").read_text())
    best = res.get("best") or {}
    hyps = best.get("hypotheses", [])
    faulty_s = set(doc["faulty_statements"])
    faulty_f = set(doc["faulty_functions"])
    spectrum_s = [el for el, sc, rk in doc["suspicion_formulas"]["ochiai"]]
    # statement-level: hypotheses (methods expanded to their statements in
    # spectrum order), then spectrum tail (plan §5.9)
    stmt_ranking: list[str] = []
    for h in sorted(hyps, key=lambda h: h["rank"]):
        for el in _expand_method_to_statements(h, doc):
            if el not in stmt_ranking:
                stmt_ranking.append(el)
    for el in spectrum_s:
        if el not in stmt_ranking:
            stmt_ranking.append(el)
    # method-level: hypotheses' methods first (statement lifted to its method),
    # then methods by best statement rank
    st2fn = {s["id"]: s["func_id"] for s in doc["statements"] if s["func_id"]}
    fn_ranks: dict[str, int] = {}
    for i, (el, sc, rk) in enumerate(doc["suspicion_formulas"]["ochiai"], 1):
        fn = st2fn.get(el)
        if fn and (fn not in fn_ranks or i < fn_ranks[fn]):
            fn_ranks[fn] = i
    m_ranking: list[str] = []
    for h in sorted(hyps, key=lambda h: h["rank"]):
        fn = h["element"] if h["granularity"] == "method" else st2fn.get(h["element"])
        if fn and fn not in m_ranking:
            m_ranking.append(fn)
    for fn, _ in sorted(fn_ranks.items(), key=lambda kv: (kv[1], kv[0])):
        if fn not in m_ranking:
            m_ranking.append(fn)
    ev_st = metrics.evaluate(stmt_ranking, faulty_s, len(spectrum_s))
    ev_m = metrics.evaluate(m_ranking, faulty_f, len(fn_ranks) or 1)
    return {
        "bug": b.instance_id,
        "candidate": best.get("id"),
        "score": best.get("score"),
        "n_hypotheses": len(hyps),
        "hypotheses": hyps[:10],
        "stmt": ev_st,
        "method": ev_m,
        "facts": res.get("facts", {}),
    }


def run_cell(b: swe.BugSpec, doc: dict, cfg_name: str, rep: int) -> dict:
    bug = b.instance_id
    cell = cell_dir(bug, cfg_name, rep)
    if cell_is_done(cell):
        log(f"{bug} {cfg_name} rep{rep}: done (skipped)")
    else:
        cfg = PRESET_CONFIGS[cfg_name]
        pm = extract.extract(swe.checkout_dir(b))
        pm.index()
        env = build_bug_env(b, doc)
        mv = {
            "functions": [{**f} for f in doc["model"]],
            "statements": doc["statements"],
            "call_graph": doc["call_graph"],
            "covered_by": doc.get("covered_by", {}),
            "suspicion_top": doc["suspicion_top"],
        }
        # model view needs function fields the toolbox reads
        for fn in mv["functions"]:
            fn.setdefault("class", None)
        from lethe.agent import task_md
        task = task_md(b.problem_statement, b.fail_to_pass)
        h = Harness(env, pm, cell, cfg, mv, task, b.fail_to_pass)
        log(f"{bug} {cfg_name} rep{rep}: START")
        h.run()
    m = cell_metrics(b, doc, cell)
    outp = RESULTS / "runs" / bug / cfg_name / f"rep{rep}" / "metrics.json"
    outp.write_text(json.dumps(m, indent=1, default=str))
    log(f"{bug} {cfg_name} rep{rep}: stmt top1={m['stmt']['top1']} fr={m['stmt']['first_rank']} "
        f"method top1={m['method']['top1']} fr={m['method']['first_rank']}")
    return m


def run_bug(b: swe.BugSpec, doc: dict, configs: list[str], reps: int) -> list[dict]:
    outs = []
    for cfg_name in configs:
        for rep in range(1, reps + 1):
            outs.append(run_cell(b, doc, cfg_name, rep))
    return outs


def bug_job(iid: str, cells: list[tuple[str, int]]) -> tuple[str, list[dict]]:
    """Module-level so ProcessPoolExecutor can pickle it."""
    bugs = {b.instance_id: b for b in swe.load_dataset()}
    b = bugs[iid]
    doc = json.loads((PRECOMP / f"{iid}.json").read_text())
    out = []
    for cfg, r in cells:
        out.append(run_cell(b, doc, cfg, r))
    return iid, out


def main():
    keep = json.loads(SELECTED.read_text()) if SELECTED.exists() else []
    bugs = {b.instance_id: b for b in swe.load_dataset() if b.instance_id in keep}
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugs", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    ids = args.bugs or list(keep)
    todo = []
    for iid in ids:
        if iid not in bugs:
            print(f"skip {iid} (not in selected set)")
            continue
        doc = json.loads((PRECOMP / f"{iid}.json").read_text())
        for cfg in args.configs:
            for r in range(1, args.reps + 1):
                if cell_is_done(cell_dir(iid, cfg, r)):
                    continue
                todo.append((iid, cfg, r))
    print(f"cells to run: {len(todo)} across bugs={sorted(set(i for i, c, r in todo))}")
    by_bug: dict[str, list[tuple[str, int]]] = {}
    for iid, cfg, r in todo:
        by_bug.setdefault(iid, []).append((cfg, r))
    items = list(by_bug.items())
    if args.workers <= 1 or len(items) == 1:
        for iid, cells in items:
            doc = json.loads((PRECOMP / f"{iid}.json").read_text())
            for cfg, r in cells:
                run_cell(bugs[iid], doc, cfg, r)
    else:
        def job(iid: str, cells: list[tuple[str, int]]):
            doc = json.loads((PRECOMP / f"{iid}.json").read_text())
            out = []
            for cfg, r in cells:
                out.append(run_cell(bugs[iid], doc, cfg, r))
            return iid, out
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(bug_job, iid, cells) for iid, cells in items]
            for f in as_completed(futs):
                iid, out = f.result()
                log(f"finished bug {iid}")
    print("RUNNER DONE")


if __name__ == "__main__":
    main()
