from __future__ import annotations

import shutil
import sys
from pathlib import Path

from lethe import extract
from lethe.judge import BugEnv, judge_candidate, judge_probe

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "shop"


def test_shop_program_model_matches_design() -> None:
    pm = extract.extract(FIXTURE)
    assert len(pm.components) == 3
    assert len(pm.functions) == 12
    assert len(pm.statements) == 33
    assert pm.source_hash


def test_shop_real_fixprobe_finds_boundary_fault(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    judge_home = tmp_path / "judge"
    shutil.copytree(FIXTURE, checkout)
    shutil.copytree(FIXTURE, judge_home)
    pm = extract.extract(checkout)
    pm.index()
    failing = ["tests/test_shop.py::test_three_year_member"]
    passing = [
        f"tests/test_shop.py::{name}"
        for name in (
            "test_new_customer_small_order",
            "test_one_year_member",
            "test_two_year_member",
            "test_one_year_small_order",
            "test_two_year_mixed_cart",
            "test_two_year_big_cart",
            "test_one_year_two_items",
            "test_four_year_member",
            "test_five_year_member",
            "test_bulk_order",
            "test_rounding_case",
        )
    ]
    env = BugEnv(
        checkout=checkout,
        judge_home=judge_home,
        python=sys.executable,
        fail_to_pass=failing,
        pass_to_pass=passing,
        pass_to_pass_sampled=passing,
        spectrum={},
        n_tests=12,
        base_pass=11,
        checkout_hash=pm.source_hash,
        cache_dir=tmp_path / "cache",
        test_timeout_s=30,
        p2p_sample=None,
    )
    probe = judge_probe(env, pm, "shop/pricing.py:3", max_mutants=20)
    assert probe.fixed
    assert probe.mutants_run > 0
    assert probe.best_pass == 12
    assert any("cmp:gt" in edit for edit in probe.fixing_edits)

    truncated = judge_candidate(
        env,
        pm,
        "c-truncated",
        [{"rank": 1, "element": "shop/pricing.py::loyalty_rate", "granularity": "method"}],
        None,
        max_mutants=1,
    )
    assert truncated.per_hypothesis[0]["probe"] == "inconclusive"
    assert truncated.per_hypothesis[0]["truncated"] is True
