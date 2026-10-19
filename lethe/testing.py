"""Shared selected-test runner with per-repo spec formats.

SWE-bench instances carry tests in three formats:
- pytest nodes            ``tests/test_x.py::test_y``
- django/old-unittest     ``test_y (pkg.Class)``  ->  ``tests/runtests.py``
- bare names (old sympy)  ``test_Foo``            ->  ``pytest -k``

Everything (setup verification, the judge, coverage collection) goes through
``run_selected_tests`` so all callers agree on what "the tests pass" means.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .core import run_proc


def parse_spec(spec: str) -> tuple[str, str | list[str]]:
    s = spec.strip()
    if "::" in s:
        return "pytest", s
    m = re.match(r"^(.+?)\s+\(([^()]+)\)$", s)
    if m:
        name, modcls = m.group(1).strip(), m.group(2)
        parts = modcls.split(".")
        cls = parts[-1]
        mod = ".".join(parts[:-1])
        return "runtests", f"{mod}.{cls}.{name}"
    return "k", s


def _summarize(out: str) -> tuple[int, int]:
    """(passed, failed) from a pytest-style summary."""
    p = 0
    f = 0
    m = re.search(r"(\d+) passed", out)
    if m:
        p = int(m.group(1))
    m = re.search(r"(\d+) failed", out)
    if m:
        f = int(m.group(1))
    m = re.search(r"(\d+) error", out)
    if m:
        f += int(m.group(1))
    return p, f


def _find_node_for_name(cwd: Path, name: str) -> str | None:
    """Locate `def name(` in the tree; return file::name when exactly one def is
    found (avoids -k substring over-matching)."""
    hits: list[str] = []
    for f in sorted(cwd.rglob("*.py")):
        if any(part in {".git", "node_modules", "__pycache__", ".venv"} for part in f.parts):
            continue
        try:
            txt = f.read_text(errors="replace")
        except OSError:
            continue
        if re.search(rf"def {re.escape(name)}\s*\((.*?)\):", txt) or \
           re.search(rf"def {re.escape(name)}\(", txt):
            try:
                rel = f.relative_to(cwd).as_posix()
                hits.append(f"{rel}::{name}")
            except ValueError:
                pass
    return hits[0] if len(hits) == 1 else None


def _parse_django_output(out: str) -> tuple[int, int]:
    # "OK", "OK (created=2)", "FAILED (failures=1)", "FAILED (errors=2)"
    if re.search(r"^OK\b", out, re.M):
        return 1, 0
    p = re.search(r"failures=(\d+)", out)
    e = re.search(r"errors=(\d+)", out)
    failedn = int(p.group(1)) if p else 0
    errn = int(e.group(1)) if e else 0
    runs = re.search(r"Ran (\d+) test", out)
    total = int(runs.group(1)) if runs else (failedn + errn)
    return max(0, total - failedn - errn), failedn + errn


def run_selected_tests(py: str, cwd: Path, specs: list[str], timeout: int = 900,
                       extra_env: dict | None = None) -> dict:
    """Run each spec (separately when needed so pass/fail accounting is exact).

    Returns {ok, passed, selected, rc, out}.
    """
    total_p = 0
    total_f = 0
    rc_all = 0
    out_parts = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(extra_env or {}))
    for spec in specs:
        kind, arg = parse_spec(spec)
        if kind == "pytest":
            cmd = [py, "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", arg]
        elif kind == "runtests":
            cmd = [py, "tests/runtests.py", arg, "--settings=test_sqlite", "--verbosity=1"]
        else:
            # bare test name (old SWE-bench sympy): prefer an exact file::name
            # node located by source search; fall back to -k if ambiguous
            node = _find_node_for_name(cwd, arg)
            target = node if node is not None else ["-k", arg]
            cmd = ([py, "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", node]
                   if node is not None else
                   [py, "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", "-k", arg])
        rc, so, se = run_proc(cmd, cwd=cwd, timeout=timeout, env=env)
        out = so + "\n" + se
        if kind == "runtests":
            p, f = _parse_django_output(out)
        else:
            p, f = _summarize(out)
        if rc == 5 or "no tests ran" in out.lower():
            p, f = 0, 1
        total_p += p
        total_f += f
        rc_all |= rc
        out_parts.append(f"$ {' '.join(cmd)}\n{out.rstrip()[-600:]}")
    return {"ok": total_f == 0 and total_p > 0, "passed": total_p,
            "failed": total_f, "selected": max(1, total_p + total_f),
            "rc": rc_all, "out": "\n---\n".join(out_parts)}


def run_selected_tests_coved(py: str, cwd: Path, spec: str, data_file: Path,
                             timeout: int = 900, extra_env: dict | None = None) -> int:
    """One spec under coverage.py; returns rc (0/1 = ok/failed, 2 = error)."""
    kind, arg = parse_spec(spec)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", COVERAGE_FILE=str(data_file),
               **(extra_env or {}))
    if kind == "pytest":
        cmd = [py, "-m", "coverage", "run", "--data-file", str(data_file),
               "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", arg]
    elif kind == "runtests":
        cmd = [py, "-m", "coverage", "run", "--data-file", str(data_file),
               "tests/runtests.py", arg, "--settings=test_sqlite", "--verbosity=1"]
    else:
        cmd = [py, "-m", "coverage", "run", "--data-file", str(data_file),
               "-m", "pytest", "-q", "--no-header", "--tb=no", "-p", "no:cacheprovider", "-k", arg]
    rc, so, se = run_proc(cmd, cwd=cwd, timeout=timeout, env=env)
    return rc
