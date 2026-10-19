"""Baselines (plan §7.4).

SBFL: Ochiai, DStar(2,3), Tarantula on the precomputed spectra (tie policy
'average'). Statement-level ranking directly; method-level ranking = best
(tie-mean) rank of the method's statements.

One-shot LLM: exactly one LLM call per bug, no tools, no judge; the prompt
carries the problem statement, failing tests and the top spectrum suspects;
the model must return a ranked JSON list of elements; the ranking is then
completed with the spectrum tail exactly as for the agent configurations
(plan §5.9), so the comparison is on equal footing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import metrics
from lethe.agent import llm_chat, DEFAULT_BASE, DEFAULT_MODEL
from evaluation import swebench as swe


def _method_ranking(statement_ranked: list[tuple[str, float, float]], st2fn: dict[str, str]) -> list[tuple[str, float, float]]:
    best: dict[str, float] = {}
    for el, sc, rk in statement_ranked:
        fn = st2fn.get(el)
        if not fn:
            continue
        if fn not in best or rk < best[fn]:
            best[fn] = rk
    out = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))
    total = len(statement_ranked)
    method_order = 0
    ranked = []
    for i, (fn, rk) in enumerate(out, 1):
        ranked.append((fn, -rk, float(i)))
    return ranked


def sbfl_for_bug(doc: dict, formula: str = "ochiai") -> dict:
    """rank doc['suspicion_formulas'][formula] (already ranked list) at both
    granularities."""
    ranked = doc["suspicion_formulas"][formula]  # [(el, score, avg_rank), ...]
    stmt_ranking = [el for el, sc, rk in ranked]
    fn_stmts = {}
    for s in doc["statements"]:
        if s["func_id"]:
            fn_stmts[s["id"]] = s["func_id"]
    m_ranked = _method_ranking(ranked, fn_stmts)
    m_ranking = [fn for fn, sc, rk in m_ranked]

    n_stmt = len(stmt_ranking)
    n_fn = len(m_ranking) or 1
    faulty_s = set(doc["faulty_statements"])
    faulty_f = set(doc["faulty_functions"])
    res = {
        "stmt": metrics.evaluate(stmt_ranking, faulty_s, n_stmt) | {
            "first_rank_tie": metrics.first_rank_with_ties(ranked, faulty_s),
            "rr_tie": metrics.rr_with_ties(ranked, faulty_s)},
        "method": metrics.evaluate(m_ranking, faulty_f, n_fn),
    }
    return {"formula": formula, "stmt": res["stmt"], "method": res["method"]}


ONESHOT_PROMPT = """You are localizing a single fault in a Python project.

The failing tests (they fail on this version, pass on the repaired version):
{tests}

Defect report from maintainers:
{problem}

A spectrum (fault-localisation) ranking puts the following methods/statements at the top (id, score):
{suspects}

Return ONLY a JSON array of at most 6 objects ranked most-suspicious first, each {{"element": "<path::Func OR path:LINE>", "why": "<= 80 chars"}}.
Use only ids that appear in the suspect list or the defect report. No other text.
"""


def oneshot_for_bug(b: swe.BugSpec, doc: dict, base: str = DEFAULT_BASE,
                    model: str = DEFAULT_MODEL, temperature: float = 0.0) -> dict:
    suspects = doc["suspicion_top"][:25]
    sus_txt = "\n".join(f"{h['id']}  ({h['score']})" for h in suspects)
    valid_ids = {h["id"] for h in suspects}
    prompt = ONESHOT_PROMPT.format(
        tests="\n".join(f"- {t}" for t in b.fail_to_pass),
        problem=b.problem_statement.strip()[:2000],
        suspects=sus_txt)
    resp = llm_chat(base, model, [{"role": "user", "content": prompt}], tools=None,
                    temperature=temperature, max_tokens=1024)
    text = resp["message"].get("content", "")
    m = re.search(r"\[[\s\S]*\]", text)
    hyps = []
    if m:
        try:
            rows = json.loads(m.group(0))
            for i, r in enumerate(rows[:6], 1):
                el = str(r.get("element", "")).strip()
                if el.startswith("repo/"):
                    el = el[5:]
                if el in valid_ids:
                    hyps.append({"rank": len(hyps) + 1, "element": el,
                                "granularity": "statement" if ":" in el.split("::")[-1] else "method",
                                "rationale": str(r.get("why", ""))[:200]})
        except json.JSONDecodeError:
            hyps = []
    # ranking completion exactly as agent configs (plan §5.9)
    spectrum = [el for el, sc, rk in doc["suspicion_formulas"]["ochiai"]]
    ranking = metrics.complete_ranking(hyps, spectrum, "statement")
    st2fn = {s["id"]: s["func_id"] for s in doc["statements"] if s["func_id"]}
    m_ranked = _method_ranking(doc["suspicion_formulas"]["ochiai"], st2fn)
    # re-rank methods: hypotheses' methods first, others follow
    hyp_fns = [st2fn.get(h["element"]) for h in hyps if st2fn.get(h["element"])]
    m_ranking_list = [fn for fn, sc, rk in m_ranked]
    for fn in reversed(hyp_fns):
        try:
            m_ranking_list.remove(fn)
        except ValueError:
            pass
        m_ranking_list.insert(0, fn)
    n_stmt = len(ranking)
    n_fn = len(m_ranking_list) or 1
    res = {
        "n_accepted": len(hyps),
        "stmt": metrics.evaluate(ranking, set(doc["faulty_statements"]), n_stmt),
        "method": metrics.evaluate(m_ranking_list, set(doc["faulty_functions"]), n_fn),
        "usage": resp.get("usage", {}),
        "latency_s": round(resp["latency"], 2),
        "raw": text[:4000],
    }
    return res
