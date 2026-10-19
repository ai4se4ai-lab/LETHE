"""SWE-bench Lite adapter for LETHE.

Loads the benchmark, sets up clean per-bug environments (repo at base commit
WITHOUT the fix + benchmark test patch, in a dedicated virtualenv), verifies
reproducible failure (F2P fail on buggy; P2P pass), derives ground truth from
the fix patch, and provides the interface used by baselines and the judge.

Leak protection (plan §8): after verification the checkout's VCS history is
re-initialised as a single commit at the buggy state and the fix patch is
checked to NOT apply to it; the agent workspace copy has no .git at all.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from lethe.core import run_proc

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LETHE_EVAL_DATA", str(ROOT / "data"))).resolve()
ENVS = DATA / "envs"
PARQUET = Path(
    os.environ.get("LETHE_SWEBENCH_PARQUET", str(DATA / "swe-bench_lite.parquet"))
).resolve()


@dataclass
class BugSpec:
    instance_id: str
    repo: str
    base_commit: str
    env_commit: str
    problem_statement: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    test_patch: str
    gold_patch: str
    created_at: str
    url: str


def load_dataset(parquet: Path = PARQUET) -> list[BugSpec]:
    t = pq.read_table(parquet).to_pydict()
    n = len(t["repo"])
    out = []
    for i in range(n):
        row = {k: t[k][i] for k in t}
        f2p = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else list(row["FAIL_TO_PASS"] or [])
        p2p = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else list(row["PASS_TO_PASS"] or [])
        out.append(BugSpec(
            instance_id=row["instance_id"], repo=row["repo"],
            base_commit=row["base_commit"], env_commit=row["environment_setup_commit"],
            problem_statement=row["problem_statement"], fail_to_pass=f2p,
            pass_to_pass=p2p, test_patch=row["test_patch"], gold_patch=row["patch"],
            created_at=str(row["created_at"])[:10],
            url=f"https://github.com/{row['repo']}/blob/{row['base_commit']}"))
    return out


CANDIDATES = {
    "django__django-15388", "pallets__flask-4992", "sympy__sympy-24909",
    "pytest-dev__pytest-11143", "sphinx-doc__sphinx-11445",
    "pylint-dev__pylint-5859", "django__django-11049",
}
REPO_EXTRAS = {
    "mwaskom/seaborn": ["numpy<2", "pandas<2"],
    "pallets/flask": ["werkzeug>=2.2,<3", "pytest>=7,<9", "pytest-cov"],
    "pytest-dev/pytest": ["pytest>=7,<9", "pytest-cov"],
    "pylint-dev/pylint": ["toml"],
    "django/django": ["jinja2"],
}



def env_dir(b: BugSpec) -> Path:
    return ENVS / b.instance_id


def checkout_dir(b: BugSpec) -> Path:
    return env_dir(b) / "checkout"


def judge_dir(b: BugSpec) -> Path:
    return env_dir(b) / "judge"


def package_roots(base: Path) -> list[str]:
    """Directories that should shadow site-packages installs of this repo's
    packages (flat layout: base; src layout: base and base/src). Used for
    PYTHONPATH when running this repo's tests, so the JUDGE tree (with any
    mutant edits) is what gets imported, not a stale installed copy."""
    roots = [str(base)]
    src = base / "src"
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.is_dir() and ((p / "__init__.py").exists() or p.name == "_pytest"):
                roots.append(str(src))
                break
    return roots


def shadow_env(base: Path) -> dict:
    import os as _os
    return {"PYTHONPATH": _os.pathsep.join(package_roots(base))}


def venv_python(b: BugSpec) -> Path:
    return env_dir(b) / "venv/bin/python"


def syspython() -> str:
    configured = os.environ.get("LETHE_PYTHON")
    if configured:
        return configured
    local = Path(__file__).resolve().parents[1] / ".venv/bin/python"
    return str(local if local.exists() else Path(sys.executable))


def tail(s: str, n: int = 2000) -> str:
    return s[-n:]


def _run_tests(b: BugSpec, tests: list[str], timeout: int = 1800):
    from lethe.testing import run_selected_tests
    res = run_selected_tests(str(venv_python(b)), env_dir(b) / "repo-git", tests, timeout=timeout)
    return res["rc"], res["out"], res


def setup(b: BugSpec, skip_install: bool = False) -> dict:
    d = env_dir(b)
    d.mkdir(parents=True, exist_ok=True)
    st: dict = {"instance": b.instance_id, "steps": {}}

    def record(name: str, ok: bool, extra: dict | None = None) -> bool:
        st["steps"][name] = {"ok": bool(ok)} | (extra or {})
        return bool(ok)

    def finish(ok: bool) -> dict:
        st["ok"] = bool(ok)
        (d / "status.json").write_text(json.dumps(st, indent=2))
        return st

    if skip_install:
        if not venv_python(b).exists():
            return finish(False)
    else:
        url = f"https://github.com/{b.repo}.git"
        repo_git = d / "repo-git"
        if repo_git.exists():
            return finish(False)
        record("clone", run_proc(["git", "clone", "-q", url, str(repo_git)], timeout=3600)[0] == 0)
        if record("checkout_env", run_proc(["git", "checkout", "-q", b.env_commit], cwd=repo_git, timeout=600)[0] == 0):
            rc, so, se = run_proc([syspython(), "-m", "venv", str(d / "venv")], timeout=300)
            record("venv", rc == 0)
            if record("pip_base", run_proc([str(venv_python(b)), "-m", "pip", "install", "-q", "-U",
                                            "pip", "setuptools", "wheel"], timeout=900)[0] == 0):
                pins = REPO_EXTRAS.get(b.repo, [])
                ok = True
                if pins:
                    ok = run_proc([str(venv_python(b)), "-m", "pip", "install", "-q", *pins], timeout=2400)[0] == 0
                if ok:
                    rc, so, se = run_proc([str(venv_python(b)), "-m", "pip", "install", "-q", "-e", "."],
                                         cwd=repo_git, timeout=5400)
                    ok = rc == 0
                    if not ok:
                        rc, so, se = run_proc([str(venv_python(b)), "-m", "pip", "install", "-q",
                                               "--no-build-isolation", "-e", "."], cwd=repo_git, timeout=5400)
                        ok = rc == 0
                    record("pip_editable", ok, {"tail": tail(so) + tail(se)})
                ok = ok and run_proc([str(venv_python(b)), "-m", "pip", "install", "-q",
                                      "pytest>=7,<9", "pytest-cov", "coverage"], timeout=1800)[0] == 0
                record("pip_testtools", ok)
                if ok and record("checkout_base", run_proc(["git", "checkout", "-q", b.base_commit],
                                                           cwd=repo_git, timeout=600)[0] == 0):
                    (d / "test.patch").write_text(b.test_patch)
                    (d / "gold.patch").write_text(b.gold_patch)
                    ok = True
                    # reapply test patch on top of base (some tests are modified at base)
                    run_proc(["git", "apply", "-q", "--reverse", "--check", str(d / "test.patch")], cwd=repo_git)
                    rc, so, se = run_proc(["git", "apply", "-q", "--whitespace=nowarn", str(d / "test.patch")], cwd=repo_git)
                    ok = rc == 0 or run_proc(["git", "apply", "-q", "--whitespace=nowarn", "--reject",
                                              str(d / "test.patch")], cwd=repo_git)[0] == 0
                    if ok and record("test_patch", True):
                        rc1, out1, res1 = _run_tests(b, b.fail_to_pass, timeout=1200)
                        f2p_fail = (res1["passed"] < res1["selected"]) or (res1["rc"] != 0)
                        rc2, out2, res2 = _run_tests(b, b.pass_to_pass, timeout=3600)
                        sound = (res2["rc"] == 0) and (res2["passed"] == res2["selected"])
                        p2p_good = b.pass_to_pass
                        if not sound:
                            # per-test scan: keep env-sound tests (pass on buggy),
                            # drop the rest and record how many (environment noise,
                            # e.g. pytest 9 API drift) - logged, never silent
                            from lethe.testing import run_selected_tests
                            good: list[str] = []
                            for t in b.pass_to_pass:
                                r = run_selected_tests(str(venv_python(b)), env_dir(b) / "repo-git",
                                                       [t], timeout=600)
                                if r["rc"] == 0 and r["passed"] == r["selected"]:
                                    good.append(t)
                            p2p_good = good
                        p2p_pass = len(p2p_good) >= max(1, len(b.pass_to_pass) // 2)
                        st["p2p_good"] = p2p_good
                        st["p2p_good_n"] = len(p2p_good)
                        record("verify_f2p_fail", f2p_fail, {"rc": rc1, "out": out1[:2000],
                                "passed": res1["passed"], "selected": res1["selected"]})
                        record("verify_p2p_pass", p2p_pass, {"rc": rc2, "out": out2[:2000],
                                "passed": res2["passed"], "selected": res2["selected"],
                                "p2p_good_n": len(p2p_good)})
                        if record("verified", f2p_fail and p2p_pass):
                            # leak protection: fresh single-commit history at buggy state
                            okc = (run_proc(["git", "init", "-q", "-b", "main"], cwd=repo_git)[0] == 0
                                   and run_proc(["git", "add", "-A"], cwd=repo_git, timeout=900)[0] == 0
                                   and run_proc(["git", "commit", "-q", "-m", "buggy base (fix stripped)"],
                                                cwd=repo_git, env=dict(os.environ,
                                                GIT_AUTHOR_NAME="lethe", GIT_AUTHOR_EMAIL="l@l",
                                                GIT_COMMITTER_NAME="lethe", GIT_COMMITTER_EMAIL="l@l"),
                                                timeout=900)[0] == 0)
                            # gold patch must APPLY cleanly -> the fix is absent from the tree
                            fix_absent = run_proc(["git", "apply", "-q", "--check", str(d / "gold.patch")],
                                                  cwd=repo_git)[0] == 0
                            if record("reinit_git", okc) and record("fix_absent", fix_absent):
                                chk = checkout_dir(b)
                                if chk.exists():
                                    shutil.rmtree(chk)
                                shutil.copytree(repo_git, chk, ignore=shutil.ignore_patterns(".git"))
                                jd = judge_dir(b)
                                if jd.exists():
                                    shutil.rmtree(jd)
                                shutil.copytree(repo_git, jd, symlinks=True)
                                record("finalize", True)
                            else:
                                okc = False
        return finish(all(v["ok"] for v in st["steps"].values()))
    return finish(venv_python(b).exists())


# ---------------------------------------------------------------------------
# Ground truth (plan §7.3 conventions — frozen before experiments)
# ---------------------------------------------------------------------------

def parse_patch(patch: str) -> list[dict]:
    """Unified diff -> per-file sections with base-side removed lines and
    added-only regions (omission faults)."""
    sections: list[dict] = []
    cur: dict | None = None
    h = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    for l in patch.splitlines():
        if l.startswith("diff --git") or l.startswith("From "):
            cur = {"path": None, "removed": [], "added_only": [], "added": 0, "removed_n": 0}
            sections.append(cur)
            continue
        if cur is None:
            continue
        if l.startswith("+++ "):
            p = l[4:].strip()
            if p != "/dev/null":
                cur["path"] = p[2:] if p.startswith("b/") else p
            continue
        if l.startswith("---"):
            continue
        m = h.match(l)
        if m:
            cur["base"] = int(m.group(1))
            continue
        if l.startswith("-"):
            cur["removed"].append(cur.get("base", 0))
            cur["removed_n"] += 1
            cur["base"] = cur.get("base", 0) + 1
        elif l.startswith("+"):
            cur["added"] += 1
            if cur.setdefault("_ins", None) is None:
                cur["_ins"] = cur.get("base", 0)
        else:
            cur["base"] = cur.get("base", 0) + 1
    return [s for s in sections if s["path"]]


def _is_test_file(path: str) -> bool:
    p = path.lower()
    return (p.startswith("test") or "/test_" in p or p.endswith("_test.py")
            or p.startswith("tests/") or "conftest" in p)


def ground_truth_lines(b: BugSpec) -> dict:
    """Faulty (file, line) in the buggy version + omission insertions.

    - faulty lines: lines removed/modified in non-test source files;
    - omission fault: hunks that only add lines -> the insertion point's
      following statement (plan §7.3, omission_policy=next_statement).
    """
    faulty: list[tuple[str, int]] = []
    omission: list[tuple[str, int]] = []
    has_omission = False
    for s in parse_patch(b.gold_patch):
        p = s["path"]
        if _is_test_file(p):
            continue
        for ln in s["removed"]:
            faulty.append((p, ln))
        if s["removed_n"] == 0 and s["added"] > 0:
            omission.append((p, s.get("_ins", 0) + 1))
            has_omission = True
    return {"faulty_lines": sorted(faulty), "omission_lines": sorted(omission),
            "has_omission": has_omission}


def gt_to_elements(b: BugSpec, checkout: Path) -> tuple[list[str], list[str], str]:
    """Map ground truth onto the buggy program model's elements.

    Returns (faulty_statement_ids, faulty_function_ids, quality).
    """
    from lethe import extract
    pm = extract.extract(checkout)
    gt = ground_truth_lines(b)
    st_by_file_line: dict[str, dict[int, str]] = {}
    for s in pm.statements:
        m = st_by_file_line.setdefault(s.file, {})
        if s.lineno not in m:
            m[s.lineno] = s.id
    faulty_stmts: list[str] = []
    quality: list[str] = []
    for f, ln in gt["faulty_lines"]:
        m = st_by_file_line.get(f)
        if not m:
            quality.append("ambiguous")
            continue
        if ln in m:
            faulty_stmts.append(m[ln])
        else:
            below = [l for l in m if l <= ln]
            if below:
                faulty_stmts.append(m[max(below)])
                quality.append("approx")
            else:
                # module-level or pre-first-statement fault line: not a statement
                # element in the model; still counts for method-level truth if a
                # method in the same file is faulty from another line
                quality.append("module_level")
    for f, ln in gt["omission_lines"]:
        m = st_by_file_line.get(f)
        if not m:
            continue
        above = [l for l in m if l >= ln]
        if above:
            faulty_stmts.append(m[min(above)])
        elif ln in m:
            faulty_stmts.append(m[ln])
        else:
            below = [l for l in m if l < ln]
            faulty_stmts.append(m[max(below)] if below else m[min(m)])
        quality.append("omission")
    seen: set[str] = set()
    faulty_stmts = [x for x in faulty_stmts if not (x in seen or seen.add(x))]
    faulty_fns = sorted({pm.stmt[s].func_id for s in faulty_stmts if s in pm.stmt and pm.stmt[s].func_id})
    q = "ok"
    if quality:
        q = ",".join(sorted(set(quality)))
    return faulty_stmts, faulty_fns, q
