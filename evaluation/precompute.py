"""Precompute everything that is benchmark-side of LETHE (plan §7.2).

For each bug: program model, per-test statement coverage (trace model),
spectrum scores (Ochiai/DStar2/DStar3/Tarantula), base pass counts, ground
truth elements, and an agent-facing model view. Results go to
``evaluation/data/precomputed/<bug>.json`` so LLM runs only pay for agent
turns and judge probes (plan §7.2).

Coverage: each selected test is executed once in an isolated process under
``coverage.py``; the per-test covered-statement sets are stored directly
(failure slices come from the FAIL_TO_PASS tests, per plan §4.2).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lethe.core import ProgramModel, run_proc
from lethe import extract, spectrum as spec
from evaluation import swebench as swe

P2P_COVERAGE_SAMPLE = 12          # tests worth covering (cost control, logged)
PRECOMP = Path(__file__).resolve().parent / "data" / "precomputed"


def _p2p_sample(p2p: list[str], k: int = 12) -> list[str]:
    if len(p2p) <= k:
        return p2p
    step = len(p2p) / k
    return [p2p[i] for i in sorted({int(i * step) for i in range(k)})]


def _run_coverage(py: Path, work: Path, test_id: str, data_file: Path, timeout: int = 600,
                  extra_env: dict | None = None) -> int:
    from lethe.testing import run_selected_tests_coved
    return run_selected_tests_coved(str(py), work, test_id, data_file, timeout=timeout,
                                   extra_env=extra_env)


def _coverage_json(py: Path, data_file: Path, root: Path) -> dict | None:
    """Run `coverage json` against data file and return {file: {line: count}}."""
    env = dict(os.environ, COVERAGE_FILE=str(data_file))
    rc, so, se = run_proc([str(py), "-m", "coverage", "json", "-q",
                           "--data-file", str(data_file)], cwd=root, timeout=120, env=env)
    if rc != 0:
        return None
    # coverage writes coverage.json in cwd
    cj = root / "coverage.json"
    if not cj.exists():
        return None
    data = json.loads(cj.read_text())
    try:
        cj.unlink()
    except OSError:
        pass
    return data.get("files", {})


def precompute_bug(b: swe.BugSpec, force: bool = False) -> dict:
    PRECOMP.mkdir(parents=True, exist_ok=True)
    out = PRECOMP / f"{b.instance_id}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    env = swe.env_dir(b)
    py = swe.venv_python(b)
    if not py.exists():
        raise RuntimeError(f"env for {b.instance_id} not built yet")
    checkout = swe.checkout_dir(b)
    judge_home = swe.judge_dir(b)
    st = json.loads((env / "status.json").read_text())

    print(f"[precompute] {b.instance_id}: program model ...", flush=True)
    pm = extract.extract(checkout)
    pm.index()

    # environment-sound P2P pool (tests confirmed passing on the buggy version;
    # tests broken by version drift were scanned out during env verification and
    # are reported in setup status — never silently dropped)
    p2p_pool = st.get("p2p_good") or b.pass_to_pass
    cov_tests: list[tuple[str, str]] = []    # (kind, test_id)
    cov_tests += [("F", t) for t in b.fail_to_pass]
    cov_tests += [("P", t) for t in _p2p_sample(p2p_pool, P2P_COVERAGE_SAMPLE)]

    covers: dict[str, set[str]] = {}
    valid_files = {s.file for s in pm.statements}
    def norm_file(f: str) -> str:
        for base in (str(checkout), str(judge_home), str(env / "repo-git")):
            if f.startswith(base + "/"):
                f = f[len(base) + 1:]
        if "/repo-git/" in f:
            f = f.split("/repo-git/")[-1]
        return f
    print(f"[precompute] {b.instance_id}: coverage over {len(cov_tests)} tests ...", flush=True)
    import tempfile
    tmp = env / "covtmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for i, (kind, t) in enumerate(cov_tests):
        df = tmp / f"cov_{i}"
        rc = _run_coverage(py, checkout, t, df, timeout=900,
                           extra_env=swe.shadow_env(checkout))
        files = _coverage_json(py, df, checkout)
        if os.environ.get("DEBUG_COV"):
            print(f"    dbg cov {kind} {t[:50]}: files={0 if not files else len(files)}", flush=True)
        if not files:
            continue
        covered: set[str] = set()
        for f, lines in files.items():
            if isinstance(lines, dict):   # coverage json summary shape
                lines = lines.get("executed_lines") or lines.get("summary", {}).get("executed_lines", [])
            nf = norm_file(f)
            if nf not in valid_files:
                continue
            for ln in lines:
                covered.add(f"{nf}:{ln}")
        if os.environ.get("DEBUG_COV"):
            print(f"    dbg covered for {t[:50]}: {len(covered)}", flush=True)
        covers[t] = covered if covered else set()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    # trace model (plan §4.2)
    failing = [t for t in b.fail_to_pass]
    passing = _p2p_sample(p2p_pool, P2P_COVERAGE_SAMPLE)
    failing_slice = set().union(*[covers.get(t, set()) for t in failing]) if failing else set()

    # spectra
    all_tests = list(failing) + list(passing)
    n_f = len(failing)
    n_p = len(passing)
    scores = spec.FORMULAS
    # only program-model elements participate in the ranking (test-file
    # execution shows up in coverage but is not candidate space); coverage ids
    # may be relative or absolute depending on import style
    valid_ids = set(pm.stmt)
    cbase = str(checkout) + "/"
    raw_ids = {sid for c in covers.values() for sid in c} | failing_slice
    scored_ids: set[str] = set()
    for sid in raw_ids:
        r = sid[len(cbase):] if sid.startswith(cbase) else sid
        if r in valid_ids:
            scored_ids.add(r)
        elif sid in valid_ids:
            scored_ids.add(sid)
    ranked: dict[str, list[tuple[str, float, float]]] = {}
    for name in ("ochiai", "dstar2", "dstar3", "tarantula"):
        s: dict[str, float] = {}
        for s_id in scored_ids:
            ef = sum(1 for t in failing if s_id in covers.get(t, set()))
            ep = sum(1 for t in passing if s_id not in covers.get(t, set()))
            nf = sum(1 for t in passing if s_id in covers.get(t, set()))
            np_ = sum(1 for t in failing if s_id not in covers.get(t, set()))
            s[s_id] = scores[name](ef, ep, nf, np_)
        ranked[name] = list(spec.rank(s))

    # base pass counts (buggy version), using the same P2P sample the judge uses
    from lethe.testing import run_selected_tests
    res_f = run_selected_tests(str(py), checkout, b.fail_to_pass, timeout=900,
                               extra_env=swe.shadow_env(checkout))
    f_passed, f_failed = res_f["passed"], res_f["failed"]
    res_p = run_selected_tests(str(py), checkout, passing, timeout=2400,
                               extra_env=swe.shadow_env(checkout))
    p_passed = res_p["passed"]
    base_pass = p_passed   # failing tests pass 0 on the buggy version by construction
    n_tests = len(failing) + len(passing)

    # ground truth onto program-model elements
    faulty_stmts, faulty_fns, gtq = swe.gt_to_elements(b, checkout)

    # model view for the agent (bounded, deterministic)
    functions = [{"id": f.id, "file": f.file, "name": f.name, "class": f.class_name,
                  "lines": f"{f.lineno}-{f.end_lineno}", "calls": f.calls}
                 for f in sorted(pm.functions, key=lambda x: x.id)]
    statements = [{"id": s.id, "file": s.file, "line": s.lineno, "func_id": s.func_id,
                   "kind": s.kind, "text": s.text} for s in pm.statements]
    call_graph = {f.id: f.calls for f in pm.functions}
    suspicion_top = [{"id": el, "score": round(sc, 5), "rank": rk} for el, sc, rk in ranked["ochiai"][:60]]
    covered_by = {sid: sorted(t for t in all_tests if sid in covers.get(t, set()))
                  for sid in [h["id"] for h in suspicion_top][:30]}

    doc = {
        "instance": b.instance_id,
        "repo": b.repo,
        "problem_statement": b.problem_statement,
        "fail_to_pass": failing,
        "pass_to_pass_sample": passing,
        "n_tests": n_tests,
        "base_pass": base_pass,
        "f2p_passed_on_buggy": f_passed,
        "f2p_failed_on_buggy": f_failed,
        "p2p_passed_on_buggy": p_passed,
        "p2p_pool_size": len(p2p_pool),
        "n_functions": len(functions),
        "n_statements": len(statements),
        "n_components": len(pm.components),
        "model": functions,
        "statements": statements,
        "call_graph": call_graph,
        "suspicion_ranked": ranked["ochiai"],
        "suspicion_formulas": {name: v for name, v in ranked.items()},
        "suspicion_top": suspicion_top,
        "covered_by": covered_by,
        "fails": {t: len(covers.get(t, set())) for t in all_tests},
        "faulty_statements": faulty_stmts,
        "faulty_functions": faulty_fns,
        "gt_quality": gtq,
        "swebench_status": {k: v.get("ok") for k, v in st["steps"].items()} if "steps" in st else {},
    }
    out.write_text(json.dumps(doc, indent=1))
    print(f"[precompute] {b.instance_id}: done "
          f"(f2p_fail={f_failed}/{len(failing)}, p2p_pass={p_passed}/{len(passing)}, "
          f"{len(functions)} fns, {len(statements)} stmts, gt={gtq})", flush=True)
    return doc


def main():
    bugs = {b.instance_id: b for b in swe.load_dataset()}
    keep = []
    for iid, b in sorted(bugs.items()):
        if iid not in swe.CANDIDATES:
            continue
        try:
            doc = precompute_bug(b, force=("--force" in sys.argv))
            if doc.get("f2p_failed_on_buggy", 0) > 0 and doc.get("p2p_passed_on_buggy", 0) > 0 \
                    and doc["faulty_statements"]:
                keep.append(iid)
            else:
                print(f"  [excluded] {iid}: f2p/p2p/gt check failed", flush=True)
        except Exception as e:
            print(f"  [excluded] {iid}: {type(e).__name__}: {e}", flush=True)
    (PRECOMP / "_selected.json").write_text(json.dumps(keep, indent=1))
    print("selected bugs:", keep)


if __name__ == "__main__":
    main()
