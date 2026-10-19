"""The judge: repair-based verification of suspicion candidates.

Signals (plan §5.6):
- ``fixprobe`` (core): for each hypothesis element, try single-edit mutants
  inside the element and check whether any makes the failing tests pass
  without regressions (fail-fast: failing tests first, rest only if they pass).
- ``patch_validation``: apply the agent-supplied diff and run the suite.
- ``spectrum_prior``: percentile of the top hypotheses in the spectrum ranking.

The judge never sees ground truth; it only sees the buggy checkout, the tests
and the candidate. Judge cost is accounted separately from LLM cost (D7/D9).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .core import ProgramModel, run_proc, SCHEMA_VERSION
from .mutation import generate_mutants, OPSET_VERSION


class ProbeResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    element: str
    granularity: str = "function"
    mutants_total: int = 0
    mutants_run: int = 0
    compile_failures: int = 0
    timeouts: int = 0
    truncated: bool = False
    fixed: bool = False
    best_pass: int = 0
    base_pass: int = 0
    total_tests: int = 0
    fixing_edits: list[str] = Field(default_factory=list)
    seconds: float = 0.0
    test_executions: int = 0
    cache_hit: bool = False


class JudgeResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    candidate_id: str
    score: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    per_hypothesis: list[dict[str, Any]] = Field(default_factory=list)
    dropped: list[dict[str, str]] = Field(default_factory=list)
    patch_valid: bool | None = None
    cost: dict[str, Any] = Field(default_factory=dict)


@dataclass
class BugEnv:
    """Everything the judge needs about a benchmark bug (duck-typed)."""
    checkout: Path          # buggy tree, no git (agent/model source)
    judge_home: Path        # buggy tree + single-commit git (test execution)
    python: str             # venv python
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    spectrum: dict[str, float]    # statement id -> ochiai score
    n_tests: int
    base_pass: int
    checkout_hash: str
    cache_dir: Path
    test_timeout_s: int = 180
    p2p_sample: int | None = 8    # cap PASS_TO_PASS per mutant run (cost control)
    pass_to_pass_sampled: list[str] | None = None


def _run_suite(env: BugEnv, tests: list[str]) -> tuple[int, int, int]:
    """Run the selected tests (all spec formats) inside judge_home.
    PYTHONPATH pins imports to this tree (flat or src layout) so mutants in
    judge_home actually get imported even when the venv has an installed copy.
    Returns (rc, passed, selected)."""
    from .testing import run_selected_tests
    roots = [str(env.judge_home)]
    src = env.judge_home / "src"
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.is_dir() and ((p / "__init__.py").exists() or p.name.startswith("_")):
                roots.append(str(src))
                break
    extra_env = {"PYTHONPATH": os.pathsep.join(roots)}
    res = run_selected_tests(env.python, env.judge_home, tests, timeout=env.test_timeout_s,
                             extra_env=extra_env)
    return res["rc"], res["passed"], res["selected"]


def _parse_passed(out: str) -> int:
    import re
    m = re.search(r"(\d+) passed", out)
    return int(m.group(1)) if m else 0


def _p2p_sampled(env: BugEnv) -> list[str]:
    if env.pass_to_pass_sampled is not None:
        return env.pass_to_pass_sampled
    p = env.pass_to_pass
    if env.p2p_sample and len(p) > env.p2p_sample:
        step = len(p) / env.p2p_sample
        idx = sorted({int(i * step) for i in range(env.p2p_sample)})
        return [p[i] for i in idx]
    return p


def _f2p_pass(env: BugEnv) -> tuple[bool, int, int]:
    """Run FAIL_TO_PASS; return (all_passed, passed, total)."""
    rc, passed, total = _run_suite(env, env.fail_to_pass)
    return passed >= total, passed, total


def _p2p_pass(env: BugEnv) -> tuple[bool, int, int]:
    tests = _p2p_sampled(env)
    rc, passed, total = _run_suite(env, tests)
    return passed >= total, passed, total


def judge_probe(env: BugEnv, pm: ProgramModel, element: str, max_mutants: int,
                events=None) -> ProbeResult:
    """fixprobe signal for one element (plan §5.6.1)."""
    key = _cache_key(env, element, max_mutants)
    cdir = env.cache_dir
    cdir.mkdir(parents=True, exist_ok=True)
    ck = cdir / f"{key}.json"
    if ck.exists():
        r = ProbeResult.model_validate(json.loads(ck.read_text()))
        r.cache_hit = True
        return r

    kind, obj = pm.resolve_element(element)
    if kind == "unknown":
        return ProbeResult(element=element)
    gran = "statement" if kind == "statement" else "function"
    mutants, truncated, compile_failures = generate_mutants(
        env.judge_home, obj.file, pm, element, max_mutants, env.spectrum)

    pr = ProbeResult(element=element, granularity=gran, mutants_total=len(mutants),
                     compile_failures=compile_failures, truncated=truncated,
                     base_pass=env.base_pass, total_tests=env.n_tests)
    import time
    t0 = time.time()
    for m in mutants:
        if pr.fixed:
            break
        # swap in the mutant (backup -> replace -> run -> restore)
        target = env.judge_home / m.file
        if not target.exists():
            pr.timeouts += 1
            continue
        orig = target.read_bytes()
        try:
            pr.mutants_run += 1
            target.write_text(m.text)
            f2p_ok, f_passed, f_total = _f2p_pass(env)
            pr.test_executions += 1
            pr.best_pass = max(pr.best_pass, f_passed if not f2p_ok else f_total)
            if not f2p_ok:
                continue
            # failing tests now pass: check for regressions in the P2P sample
            p2p_ok, p_passed, p_total = _p2p_pass(env)
            pr.test_executions += 1
            if p2p_ok:
                pr.fixed = True
                pr.best_pass = f_total + p_total
                pr.fixing_edits.append(f"line {m.line}: {m.op}")
            else:
                # regression: mutant changed behaviour but broke passing tests
                pr.best_pass = max(pr.best_pass, f_total + p_passed)
        finally:
            target.write_bytes(orig)
    pr.seconds = time.time() - t0
    ck.write_text(pr.model_dump_json(indent=1))
    return pr


def _cache_key(env: BugEnv, element: str, max_mutants: int) -> str:
    tests_hash = hashlib.sha256(
        ("\x00".join(env.fail_to_pass) + "\x01" + "\x00".join(_p2p_sampled(env))).encode()).hexdigest()[:16]
    base = hashlib.sha256(
        f"{env.checkout_hash}|{element}|{OPSET_VERSION}|{max_mutants}|{tests_hash}|{env.p2p_sample}".encode()).hexdigest()[:24]
    return base


def judge_candidate(env: BugEnv, pm: ProgramModel, candidate_id: str,
                    hypotheses: list[dict], patch: str | None,
                    weights: dict[str, float] | None = None,
                    top_k: int = 5, max_mutants: int = 60,
                    score_penalty: float = 0.05, penalty_cap: float = 0.2,
                    events=None) -> JudgeResult:
    """Score a whole candidate (plan §5.6.4)."""
    w = weights or {"fix": 0.70, "progress": 0.25, "spectrum": 0.05}
    N = env.n_tests
    base = env.base_pass
    denom = max(1, N - base)
    dropped: list[dict[str, str]] = []
    per_hyp: list[dict] = []
    fixed_ranks: list[int] = []
    progress: list[float] = []
    spec_pct: list[float] = []

    spec_rank = _spectrum_rank_position(pm, env.spectrum)
    for h in sorted(hypotheses, key=lambda x: x.get("rank", 0))[:top_k]:
        el = h["element"]
        kind, obj = pm.resolve_element(el)
        if kind == "unknown":
            dropped.append({"element": el, "reason": "unknown_element"})
            continue
        pr = judge_probe(env, pm, el, max_mutants, events)
        p_i = max(0.0, min(1.0, (pr.best_pass - base) / denom))
        if pr.fixed:
            fixed_ranks.append(h["rank"])
        progress.append(p_i / h["rank"])
        pos = spec_rank.get(el, 0)
        spec_pct.append(pos)
        if pr.fixed:
            probe_status = "fixed"
        elif pr.truncated or pr.mutants_run == 0:
            probe_status = "inconclusive"
        else:
            probe_status = "exonerated"
        per_hyp.append({"rank": h["rank"], "element": el, "probe": probe_status,
                        "mutants_run": pr.mutants_run, "best_pass": pr.best_pass,
                        "total_tests": N, "fixing_edits": pr.fixing_edits,
                        "cache_hit": pr.cache_hit, "truncated": pr.truncated,
                        "compile_failures": pr.compile_failures,
                        "timeouts": pr.timeouts})

    patch_valid: bool | None = None
    if patch:
        patch_valid = _validate_patch(env, patch)
        if patch_valid and hypotheses and hypotheses[0]:
            # a validated patch touching the top hypothesis is also a fix
            top = hypotheses[0]["element"]
            if top not in [p["element"] for p in per_hyp if p["probe"] == "fixed"]:
                fixed_ranks.append(1)

    if fixed_ranks:
        r_star = min(fixed_ranks)
        s_fix = 1.0 / r_star
    else:
        s_fix = 0.0
    s_prog = max(progress) if progress else 0.0
    # spectrum percentile: share of elements ranked above the top-3 hypotheses
    allrank = sorted(spec_rank.items(), key=lambda kv: -kv[1])
    n_el = max(1, len(allrank))
    if spec_pct:
        top3 = spec_pct[:3]
        s_spec = (n_el - sum(top3)) / (3 * n_el)   # mean (1 - pos/n) over top3
    else:
        s_spec = 0.0
    penalty = min(penalty_cap, score_penalty * len(dropped))
    score = max(0.0, min(1.0, w["fix"] * s_fix + w["progress"] * s_prog + w["spectrum"] * s_spec - penalty))
    return JudgeResult(
        candidate_id=candidate_id, score=round(score, 4),
        components={"fix_rank_score": round(s_fix, 4), "progress_score": round(s_prog, 4),
                    "spectrum_prior": round(s_spec, 4), "penalty": round(penalty, 4)},
        per_hypothesis=per_hyp, dropped=dropped, patch_valid=patch_valid,
        cost={"mutations": sum(p["mutants_run"] for p in per_hyp)})


def _spectrum_rank_position(pm: ProgramModel, spectrum: dict[str, float]) -> dict[str, float]:
    """element -> 1-based rank in the unified spectrum ranking (functions get
    the best (smallest) rank of their statements)."""
    scored: dict[str, float] = {}
    items = sorted(spectrum.items(), key=lambda kv: (-kv[1], kv[0]))
    for i, (stmt_id, sc) in enumerate(items, 1):
        st = pm.stmt.get(stmt_id)
        if st is None:
            continue
        scored[stmt_id] = i
        if st.func_id and (st.func_id not in scored or i < scored[st.func_id]):
            scored[st.func_id] = i
    return scored


def _validate_patch(env: BugEnv, patch_text: str) -> bool:
    import tempfile
    p = path = env.judge_home / "_agent_candidate.patch"
    p.write_text(patch_text)
    env_cmd = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    rc, so, se = run_proc(["git", "apply", "-q", "--whitespace=nowarn", str(p)],
                          cwd=env.judge_home, env=env_cmd)
    if rc != 0:
        return False
    try:
        f2p_ok, f_passed, f_total = _f2p_pass(env)
        if not f2p_ok:
            return False
        p2p_ok, p_passed, p_total = _p2p_pass(env)
        return p2p_ok
    finally:
        run_proc(["git", "checkout", "-q", "--", "."], cwd=env.judge_home,
                 env=env_cmd, timeout=300)
        run_proc(["git", "clean", "-q", "-fd"], cwd=env.judge_home, env=env_cmd, timeout=300)
        if p.exists():
            p.unlink()
