"""Localization metrics (plan §7.6) evaluated at method & statement level.

- Top-N: 1 if any faulty element is in the top N of the completed ranking.
- First rank / MFR / MRR.
- AP: average precision over all faulty elements (multi-relevant).
- EXAM: first_rank / n_elements (and slice-relative variant).
- Ties resolved with the configured policy (default average rank).

Rankings come as completed element lists (hypotheses + spectrum tail,
plan §5.9) for LLM configs, or the raw spectrum ranking for SBFL baselines.
"""
from __future__ import annotations

import math


def complete_ranking(hypotheses: list[dict], spectrum: list[str],
                     granularity: str) -> list[str]:
    """Hypothesis list first (in rank order), then a spectrum-ordered tail.

    Method-level: method hypotheses kept, statement hypotheses lifted to their
    method. Statement-level: statement hypotheses kept, method hypotheses
    expanded to statements in spectrum order (caller supplies expanded list).
    """
    seen: set[str] = set()
    out: list[str] = []
    for h in sorted(hypotheses, key=lambda h: h["rank"]):
        el = h["element"]
        if el not in seen:
            seen.add(el)
            out.append(el)
    for el in spectrum:
        if el not in seen:
            seen.add(el)
            out.append(el)
    return out


def top_n(ranking: list[str], faulty: set[str], n: int) -> float:
    return 1.0 if any(el in faulty for el in ranking[:n]) else 0.0


def first_rank(ranking: list[str], faulty: set[str]) -> float | None:
    for i, el in enumerate(ranking, 1):
        if el in faulty:
            return float(i)
    return None


def mrr(ranking: list[str], faulty: set[str]) -> float:
    fr = first_rank(ranking, faulty)
    return (1.0 / fr) if fr else 0.0


def first_rank_with_ties(ranked: list[tuple[str, float, int | None]], faulty: set[str]) -> float | None:
    """Tie-aware first rank: min of the (average) ranks assigned to faulty
    elements (plan §7.6 tie policy, default 'average')."""
    ranks = [rk for el, sc, rk in ranked if el in faulty and rk is not None]
    return float(min(ranks)) if ranks else None


def rr_with_ties(ranked: list[tuple[str, float, int | None]], faulty: set[str]) -> float:
    fr = first_rank_with_ties(ranked, faulty)
    return (1.0 / fr) if fr else 0.0


def ap(ranking: list[str], faulty: set[str]) -> float:
    n_rel = len(faulty)
    if n_rel == 0:
        return 0.0
    hit = 0
    s = 0.0
    for i, el in enumerate(ranking, 1):
        if el in faulty:
            hit += 1
            s += hit / i
    return s / min(n_rel, len(ranking))


def exam(ranking: list[str], faulty: set[str], n_elements: int) -> float | None:
    fr = first_rank(ranking, faulty)
    if fr is None or n_elements == 0:
        return None
    return fr / n_elements


def evaluate(ranking: list[str], faulty: set[str], n_elements: int) -> dict:
    fr = first_rank(ranking, faulty)
    return {
        "top1": top_n(ranking, faulty, 1),
        "top3": top_n(ranking, faulty, 3),
        "top5": top_n(ranking, faulty, 5),
        "top10": top_n(ranking, faulty, 10),
        "first_rank": fr,
        "rr": mrr(ranking, faulty, ),
        "ap": round(ap(ranking, faulty), 4),
        "exam": round(exam(ranking, faulty, n_elements), 5) if fr else None,
        "found": fr is not None,
    }
