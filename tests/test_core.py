"""Unit tests: spectrum, extraction, mutation, metrics, ground-truth parsing.

Run: .venv/bin/python -m pytest tests/ -q
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lethe import spectrum  # noqa: E402
from lethe import extract, mutation  # noqa: E402
from evaluation import swebench as swe  # noqa: E402
from evaluation import metrics  # noqa: E402


# ---------------------------------------------------------------- spectrum

def test_ochiai_hand():
    # ef=2 ep=1 nf=6 np=3
    ef, ep, nf, np_ = 2, 1, 6, 3
    import math
    assert spectrum.ochiai(ef, ep, nf, np_) == pytest.approx(2 / math.sqrt(8 * 12))


def test_dstar_hand():
    ef, ep, nf, np_ = 2, 1, 6, 3
    assert spectrum.dstar2(ef, ep, nf, np_) == pytest.approx((2 * 2 + 3 - 12) ** 2)


def test_rank_tie_average():
    ranked = spectrum.rank({"a": 1.0, "b": 1.0, "c": 0.5})
    assert ranked[0][0] == "a"
    assert ranked[0][2] == 1.5 == ranked[1][2]
    assert ranked[2][2] == 3.0


# --------------------------------------------------------------- extraction

@pytest.fixture(scope="module")
def shop_tree(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    (d / "shop").mkdir()
    (d / "shop" / "__init__.py").write_text("")
    (d / "shop" / "pricing.py").write_text(
        'def loyalty_rate(years):\n'
        '    """Discount rate."""\n'
        '    if years > 3:\n'
        '        return 0.10\n'
        '    if years >= 1:\n'
        '        return 0.05\n'
        '    return 0.0\n'
        '\n'
        'def bulk_rate(total_qty):\n'
        '    if total_qty >= 10:\n'
        '        return 0.02\n'
        '    return 0.0\n')
    (d / "shop" / "test_pricing.py").write_text("def test_x():\n    assert True\n")
    return d


def test_extract_counts(shop_tree: Path):
    pm = extract.extract(shop_tree)
    assert pm.components == ["shop/__init__.py", "shop/pricing.py"]
    fn_ids = {f.id for f in pm.functions}
    assert {"shop/pricing.py::loyalty_rate", "shop/pricing.py::bulk_rate"} <= fn_ids
    ids = {s.id for s in pm.statements}
    # no docstring, no function-def statements, no test file leakage
    stmt_files = {s.file for s in pm.statements}
    assert all("test" not in f for f in stmt_files)
    assert "shop/pricing.py:2" not in ids  # docstring excluded
    assert "shop/pricing.py:1" not in ids  # function def not a statement
    loyalty = [s for s in pm.statements if s.func_id == "shop/pricing.py::loyalty_rate"]
    lines = {s.lineno for s in loyalty}
    assert lines == {3, 4, 5, 6, 7}


# ------------------------------------------------------------------ mutation

def test_mutation_generates_and_compiles(shop_tree: Path):
    pm = extract.extract(shop_tree)
    muts, truncated, cf = mutation.generate_mutants(shop_tree, "shop/pricing.py", pm,
                                                    "shop/pricing.py::loyalty_rate", 30)
    assert len(muts) > 0
    assert cf == 0
    ops = {m.op for m in muts}
    assert any(op.startswith("cmp:") for op in ops)
    for m in muts:
        ast.parse(m.text)
    # the off-by-one fix must be generatable: years > 3 -> years >= 3
    assert any(m.op == "cmp:gt" for m in muts)


def test_mutation_truncation_flag(shop_tree: Path):
    pm = extract.extract(shop_tree)
    muts, truncated, _ = mutation.generate_mutants(shop_tree, "shop/pricing.py", pm,
                                                   "shop/pricing.py::loyalty_rate", 1)
    assert len(muts) == 1 and truncated


# ------------------------------------------------------------------- metrics

def test_metrics_hand():
    ranking = ["a", "b", "c", "d", "e"]
    faulty = {"c"}
    r = metrics.evaluate(ranking, faulty, 5)
    assert r["top1"] == 0 and r["top3"] == 1 and r["top5"] == 1
    assert r["first_rank"] == 3 and r["rr"] == pytest.approx(1 / 3)
    assert r["ap"] == pytest.approx(1 / 3, abs=1e-4)
    assert r["exam"] == pytest.approx(3 / 5)   # efficiency: first-found depth / |elements| (lower better)


def test_multi_faulty_ap():
    ranking = ["a", "b", "c", "d"]
    faulty = {"a", "c"}
    # standard average precision: P@1 = 1/1, P@3 = 2/3 -> AP = (1 + 2/3)/2 = 5/6
    assert metrics.ap(ranking, faulty) == pytest.approx(5 / 6)


def test_complete_ranking():
    hyps = [{"rank": 1, "element": "z"}, {"rank": 2, "element": "y"}]
    completed = metrics.complete_ranking(hyps, ["a", "y", "b", "z", "c"], "statement")
    assert completed[:2] == ["z", "y"]
    assert len(completed) == 5


# ---------------------------------------------------------- ground truth

def test_parse_patch_removed_lines():
    patch = """diff --git a/src/mod.py b/src/mod.py
--- a/src/mod.py
+++ b/src/mod.py
@@ -10,7 +10,7 @@
 context
-    if x > 3:
+    if x >= 3:
     return 1
 tail
"""
    secs = swe.parse_patch(patch)
    assert secs[0]["path"] == "src/mod.py"
    assert secs[0]["removed"] == [11]


def test_parse_patch_addition_only_is_omission():
    patch = """diff --git a/src/mod.py b/src/mod.py
--- a/src/mod.py
+++ b/src/mod.py
@@ -10,3 +10,4 @@
 context
+    do_guard()
 tail
"""
    secs = swe.parse_patch(patch)
    assert secs[0]["removed"] == []
    assert secs[0]["added"] == 1


def test_test_files_ignored():
    patch = """diff --git a/tests/test_x.py b/tests/test_x.py
--- a/tests/test_x.py
+++ b/tests/test_x.py
@@ -1,2 +1,2 @@
-a
+b
"""
    gt = swe.ground_truth_lines(type("B", (), {"gold_patch": patch})())
    assert gt["faulty_lines"] == []
