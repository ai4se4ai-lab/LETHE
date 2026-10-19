"""Collect structured run logs into the CSV contract (plan §7.7).

Reads `results/main/runs/<bug>/<config>/rep<k>/events.jsonl` + result/metrics
files and writes, under `results/main/csv/`:
  bugs.csv, runs.csv, turns.csv, judge_calls.csv, any_time/anchoring.csv,
  judge_calibration.csv, summary_by_config.csv, paired_tests.csv, baselines.csv

All numbers come from run logs + precomputed data only; no ground truth
enters the `lethe` run path (metrics are read here, evaluation-side only).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import swebench as swe
from evaluation import stats

MAIN = Path(__file__).resolve().parent / "results" / "main"
PRECOMP = Path(__file__).resolve().parent / "data" / "precomputed"
CSV = MAIN / "csv"
SCHEMA_VERSION = "1"


def _load(bug: str, cfg: str, rep: int) -> list[dict]:
    p = MAIN / "runs" / bug / cfg / f"rep{rep}" / "events.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _cells() -> list[tuple[str, str, int]]:
    cells = []
    runs = MAIN / "runs"
    if not runs.exists():
        return cells
    for bug_d in sorted(runs.iterdir()):
        if not bug_d.is_dir():
            continue
        for cfg_d in sorted(bug_d.iterdir()):
            for rep_d in sorted(cfg_d.iterdir()):
                m = re.match(r"rep(\d+)", rep_d.name)
                if m and (MAIN / "runs" / bug_d.name / cfg_d.name / rep_d.name / "metrics.json").exists():
                    cells.append((bug_d.name, cfg_d.name, int(m.group(1))))
    return cells


def _doc(bug: str) -> dict:
    return json.loads((PRECOMP / f"{bug}.json").read_text())


def collect() -> dict:
    CSV.mkdir(parents=True, exist_ok=True)
    bugs = {b.instance_id: b for b in swe.load_dataset()}
    cells = _cells()
    if not cells:
        print("no runs found"); return {}
    written: dict[str, Path] = {}

    # ---- bugs.csv -----------------------------------------------------------
    rows = []
    for bug in sorted({c[0] for c in cells}):
        b = bugs.get(bug)
        d = _doc(bug)
        rows.append({
            "schema_version": SCHEMA_VERSION, "bug_id": bug,
            "benchmark": "SWE-bench-Lite", "project": b.repo if b else "",
            "language": "python",
            "n_tests": d["n_tests"],
            "n_failing_tests": len(d["fail_to_pass"]),
            "n_passing_tests_sample": len(d["pass_to_pass_sample"]),
            "n_faulty_methods": len(d["faulty_functions"]),
            "n_faulty_statements": len(d["faulty_statements"]),
            "has_omission": d.get("gt_quality", "").find("omission") >= 0,
            "gt_quality": d.get("gt_quality", ""),
            "n_functions": d["n_functions"],
            "n_statements": d["n_statements"],
            "created_at": b.created_at if b else "",
            "included": "true",
        })
    _w("bugs.csv", rows)

    # ---- runs.csv + turns.csv + judge_calls.csv + anchoring + calibration ----
    runs_rows, turn_rows, judge_rows, anchor_rows, calib_rows = [], [], [], [], []
    for bug, cfg, rep in cells:
        evs = _load(bug, cfg, rep)
        d = _doc(bug)
        m = json.loads((MAIN / "runs" / bug / cfg / rep_dir_name(rep) / "metrics.json").read_text())
        llm = [e for e in evs if e["event"] == "llm_call"]
        turns = [e for e in evs if e["event"] == "turn_end"]
        probes = [e for e in evs if e["event"] == "probe"]
        judge_e = [e for e in evs if e["event"] == "judge_end"]
        forgets = [e for e in evs if e["event"] == "forget"]
        base = {
            "schema_version": SCHEMA_VERSION,
            "bug_id": bug, "config_name": cfg, "replicate": rep,
            "n_runs": 1,
            "n_turns": len(turns),
            "n_empty_turns": sum(1 for t in turns if t.get("empty")),
            "n_candidates": len([e for e in evs if e["event"] == "candidate_committed"]),
            "n_forgets": len(forgets),
            "tokens_in": sum(e.get("tokens_in", 0) for e in llm),
            "tokens_out": sum(e.get("tokens_out", 0) for e in llm),
            "llm_calls": len(llm),
            "judge_calls": len(judge_e),
            "judge_seconds": round(sum(e.get("cost", {}).get("judge_seconds", 0) for e in judge_e), 1),
            "best_judge_score": m.get("score"),
            "method_top1": m["method"]["top1"], "method_top3": m["method"]["top3"],
            "method_top5": m["method"]["top5"], "method_top10": m["method"]["top10"],
            "method_first_rank": m["method"]["first_rank"], "method_rr": m["method"]["rr"],
            "method_ap": m["method"]["ap"], "method_exam": m["method"]["exam"],
            "stmt_top1": m["stmt"]["top1"], "stmt_top3": m["stmt"]["top3"],
            "stmt_top5": m["stmt"]["top5"], "stmt_top10": m["stmt"]["top10"],
            "stmt_first_rank": m["stmt"]["first_rank"], "stmt_rr": m["stmt"]["rr"],
            "stmt_ap": m["stmt"]["ap"], "stmt_exam": m["stmt"]["exam"],
        }
        runs_rows.append(base)
        # turns
        cand_seq = [e for e in evs if e["event"] == "candidate_committed"]
        prev_hyp: dict[int, tuple] = {}
        for i, t in enumerate(turns):
            cid = t.get("cand_id")
            c = next((x for x in cand_seq if x.get("candidate_id") == cid), {})
            # hyp set for delta (from candidates of this run: approximate with committed events)
            a = t.get("agent_id", 0)
            row = {
                "schema_version": SCHEMA_VERSION, "bug_id": bug, "config_name": cfg,
                "replicate": rep, "agent_id": a, "round": t.get("round"),
                "empty_turn": t.get("empty"), "n_hypotheses": t.get("n_hyp"),
                "tool_calls": t.get("tool_calls"), "seconds": t.get("seconds"),
                "delta_vs_prev": None,
            }
            prev_hyp[a] = prev_hyp.get(a)
            turn_rows.append(row)
        # anchoring: model-delta size between consecutive candidates per agent
        by_agent: dict[int, list[dict]] = defaultdict(list)
        for e in cand_seq:
            by_agent[e.get("agent_id", 0)].append(e)
        # (deltas need hypotheses; stored in search_state files; approximate with n_hyp here)
        for a, lst in sorted(by_agent.items()):
            for i, e in enumerate(lst):
                anchor_rows.append({
                    "schema_version": SCHEMA_VERSION, "bug_id": bug, "config_name": cfg,
                    "replicate": rep, "agent_id": a, "proposal_index_in_agent": i,
                    "n_hypotheses": e.get("n_hypotheses"),
                    "empty_turn": e.get("empty_turn"),
                    "round": e.get("round"),
                })
        # judge calls
        faulty_s = set(d["faulty_statements"])
        faulty_f = set(d["faulty_functions"])
        for p in probes:
            el = p.get("element", "")
            judge_rows.append({
                "schema_version": SCHEMA_VERSION, "bug_id": bug, "config_name": cfg,
                "replicate": rep, "element_id": el, "candidate": p.get("candidate"),
                "agent_id": p.get("agent_id"), "fixed": p.get("fixed"),
                "mutants_run": p.get("mutants_run"), "cache_hit": p.get("cache_hit"),
                "is_faulty_statement": el in faulty_s,
                "is_faulty_method": el in faulty_f,
            })
        # calibration
        for j in judge_e:
            calib_rows.append({
                "schema_version": SCHEMA_VERSION, "bug_id": bug, "config_name": cfg,
                "replicate": rep, "candidate": j.get("candidate"),
                "judge_score": j.get("score"),
                "fix_rank_score": j.get("components", {}).get("fix_rank_score"),
                "progress_score": j.get("components", {}).get("progress_score"),
                "patch_valid": j.get("patch_valid"),
            })

    _w("runs.csv", runs_rows)
    _w("turns.csv", turn_rows)
    _w("judge_calls.csv", judge_rows)
    _w("anchoring.csv", anchor_rows)
    _w("judge_calibration.csv", calib_rows)

    # ---- summary_by_config.csv ----------------------------------------------
    summary = []
    per_config: dict[str, dict] = defaultdict(dict)
    for r in runs_rows:
        c = r["config_name"]
        for k in ("method_top1", "stmt_top1", "method_top3", "stmt_top3",
                  "method_first_rank", "stmt_first_rank", "method_rr", "stmt_rr",
                  "method_ap", "stmt_ap", "tokens_in", "judge_seconds", "n_turns"):
            per_config[c].setdefault(k, []).append(r[k])
    for c, cols in per_config.items():
        for k, vals in cols.items():
            import statistics as st
            summary.append({
                "config_name": c, "metric": k, "n": len(vals),
                "mean": round(float(st.mean(v for v in vals if v is not None)) if any(v is not None for v in vals) else 0.0, 4),
                "median": round(float(st.median(v for v in vals if v is not None)) if any(v is not None for v in vals) else 0.0, 4),
                "std": round(float(st.pstdev(v for v in vals if v is not None)) if len([v for v in vals if v is not None]) > 1 else 0.0, 4),
            })
    _w("summary_by_config.csv", summary)

    # ---- baselines.csv -------------------------------------------------------
    base_rows = []
    for bug in sorted({c[0] for c in cells}):
        d = _doc(bug)
        faulty_s, faulty_f = set(d["faulty_statements"]), set(d["faulty_functions"])
        from evaluation import baselines as bl
        n_s = len(d["suspicion_formulas"]["ochiai"])
        n_f = d["n_functions"]
        for formula in ("ochiai", "dstar2", "dstar3", "tarantula"):
            ranked = d["suspicion_formulas"][formula]
            stmt_ranking = [el for el, sc, rk in ranked]
            fn_stmts = {s["id"]: s["func_id"] for s in d["statements"] if s["func_id"]}
            best: dict[str, float] = {}
            for el, sc, rk in ranked:
                fn = fn_stmts.get(el)
                if fn and (fn not in best or rk < best[fn]):
                    best[fn] = rk
            m_ranking = [fn for fn, _ in sorted(best.items(), key=lambda kv: (kv[1], kv[0]))]
            from evaluation import metrics
            sm = metrics.evaluate(stmt_ranking, faulty_s, n_s)
            fm = metrics.evaluate(m_ranking, faulty_f, n_f)
            base_rows.append({
                "schema_version": SCHEMA_VERSION, "bug_id": bug, "formula": formula,
                "method_top1": fm["top1"], "method_top3": fm["top3"], "method_top5": fm["top5"],
                "method_top10": fm["top10"], "method_first_rank": fm["first_rank"],
                "method_rr": fm["rr"], "method_ap": fm["ap"], "method_exam": fm["exam"],
                "method_rr_tie": metrics.rr_with_ties(ranked, faulty_f),
                "stmt_top1": sm["top1"], "stmt_top3": sm["top3"], "stmt_top5": sm["top5"],
                "stmt_top10": sm["top10"], "stmt_first_rank": sm["first_rank"],
                "stmt_rr": sm["rr"], "stmt_ap": sm["ap"], "stmt_exam": sm["exam"],
                "stmt_rr_tie": metrics.rr_with_ties(ranked, faulty_s),
            })
    if base_rows:
        _w("baselines.csv", base_rows)

    # ---- paired_tests.csv -----------------------------------------------------
    pairs_cfg = [("lethe", "single"), ("lethe", "parallel4"), ("lethe", "team4"),
                 ("team4", "single"), ("lethe", "lethe_candidate_only")]
    paired = []
    for a, b in pairs_cfg:
        a_cells = {(r["bug_id"], r["replicate"]): r for r in runs_rows if r["config_name"] == a}
        b_cells = {(r["bug_id"], r["replicate"]): r for r in runs_rows if r["config_name"] == b}
        common = sorted(set(a_cells) & set(b_cells))
        if len(common) < 2:
            continue
        for metric in ("method_top1", "stmt_top1", "method_top3", "stmt_top3",
                       "method_first_rank", "stmt_first_rank"):
            va = [a_cells[k][metric] for k in common]
            vb = [b_cells[k][metric] for k in common]
            if metric.endswith("top1") or metric.endswith("top3"):
                mc = stats.mcnemar(va, vb)
                mean_diff, lo, hi = stats.bootstrap_diff_ci(
                    [x or 0 for x in va], [x or 0 for x in vb])
                a12 = stats.a12([x or 0 for x in va], [x or 0 for x in vb])
                paired.append({
                    "schema_version": SCHEMA_VERSION, "config_a": a, "config_b": b,
                    "metric": metric, "n_pairs": len(common),
                    "mean_diff": round(mean_diff, 4), "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4), "test": "mcnemar",
                    "statistic": f"b={mc['b']},c={mc['c']}", "p_value": round(mc["p"], 4),
                    "a12": round(a12, 3),
                })
            else:
                va2 = [x for x in va if x is not None]
                vb2 = [x for x in vb if x is not None]
                if len(va2) >= 3 and va2 == vb2.__len__():
                    pass
                ok = len(va2) >= 3
                if ok:
                    w = stats.wilcoxon(va2, vb2)
                    a12 = stats.a12(va2, vb2)
                    paired.append({
                        "schema_version": SCHEMA_VERSION, "config_a": a, "config_b": b,
                        "metric": metric, "n_pairs": len(common),
                        "mean_diff": round(sum(va2) / len(va2) - sum(vb2) / len(vb2), 4),
                        "ci95_low": None, "ci95_high": None, "test": "wilcoxon",
                        "statistic": round(w["statistic"], 1), "p_value": round(w["p"], 4),
                        "a12": round(a12, 3),
                    })
    if paired:
        # Holm within each metric
        by_metric: dict[str, list] = defaultdict(list)
        for r in paired:
            by_metric[r["metric"]].append(r)
        for rows in by_metric.values():
            adj = stats.holm([r["p_value"] for r in rows])
            for r, p in zip(rows, adj):
                r["p_holm"] = round(p, 4)
        _w("paired_tests.csv", paired)

    # ---- oneshot baselines ----------------------------------------------------
    onep = MAIN / "oneshot.json"
    if onep.exists():
        one = json.loads(onep.read_text())
        rows = []
        for bug, r in one.items():
            rows.append({
                "schema_version": SCHEMA_VERSION, "bug_id": bug, "n_accepted": r.get("n_accepted"),
                "method_top1": r["method"]["top1"], "method_top3": r["method"]["top3"],
                "method_top5": r["method"]["top5"], "method_top10": r["method"]["top10"],
                "method_first_rank": r["method"]["first_rank"], "method_rr": r["method"]["rr"],
                "method_ap": r["method"]["ap"], "method_exam": r["method"]["exam"],
                "stmt_top1": r["stmt"]["top1"], "stmt_top3": r["stmt"]["top3"],
                "stmt_top5": r["stmt"]["top5"], "stmt_top10": r["stmt"]["top10"],
                "stmt_first_rank": r["stmt"]["first_rank"], "stmt_rr": r["stmt"]["rr"],
                "stmt_ap": r["stmt"]["ap"], "stmt_exam": r["stmt"]["exam"],
            })
        _w("oneshot_baselines.csv", rows)

    print(f"collected {len(cells)} runs -> {CSV}")
    return {k: str(v) for k, v in written.items()}


def _w(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    p = CSV / name
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    globals().setdefault("_written", {})[name] = p


def rep_dir_name(rep: int) -> str:
    return f"rep{rep}"


def main():
    collect()


if __name__ == "__main__":
    main()
