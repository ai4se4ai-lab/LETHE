"""Spectrum-based suspiciousness formulas (pure functions, unit-testable)."""
from __future__ import annotations

import math
from typing import Callable

Formula = Callable[[int, int, int, int], float]

def ochiai(ef: int, ep: int, nf: int, np_: int) -> float:
    if ef == 0:
        return 0.0
    den = math.sqrt((ef + nf) * (ef + ep + nf + np_))
    return ef / den if den else 0.0

def _dstar(ef: int, ep: int, nf: int, np_: int, k: float) -> float:
    return (2 * ef + np_ - (ef + ep + nf + np_)) ** k

def dstar2(ef: int, ep: int, nf: int, np_: int) -> float:
    return _dstar(ef, ep, nf, np_, 2)

def dstar3(ef: int, ep: int, nf: int, np_: int) -> float:
    return _dstar(ef, ep, nf, np_, 3)

def tarantula(ef: int, ep: int, nf: int, np_: int) -> float:
    return nf / (nf + ef) if (nf + ef) else 0.0

def barinel(ef: int, ep: int, nf: int, np_: int) -> float:
    ef, ep, nf, np_ = float(ef), float(ep), float(nf), float(np_)
    if ef + nf == 0 or ef + np_ == 0:
        return 0.0
    return (0.5 * ef / (ef + nf) + 0.5 * nf / (ef + np_))

def op2(ef: int, ep: int, nf: int, np_: int) -> float:
    if ef + ep + nf + np_ == 0:
        return 0.0
    return (ef / (ef + nf)) * (nf / (ef + np_)) if (ef + nf) and (ef + np_) else 0.0

FORMULAS: dict[str, Formula] = {
    "ochiai": ochiai, "dstar2": dstar2, "dstar3": dstar3,
    "tarantula": tarantula, "barinel": barinel, "op2": op2,
}

def rank(scored: dict[str, float], tie: str = "average") -> list[tuple[str, float, int | None]]:
    """Return (element, score, avg_rank_or_None) sorted by score descending.

    tie in {'average', 'worst', 'best'}: average assigns the mean rank of the
    tied group, worst the max, best the min. Ties are broken deterministically
    by element id.
    """
    items = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[tuple[str, float, int | None]] = []
    i = 0
    n = len(items)
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        ranks = list(range(i + 1, j + 2))
        r = {"average": sum(ranks) / len(ranks), "worst": ranks[-1], "best": ranks[0]}[tie]
        for el, sc in items[i:j + 1]:
            out.append((el, sc, r))
        i = j + 1
    return out
