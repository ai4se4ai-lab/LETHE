"""Statistics (plan §7.8). Unit of analysis: the bug.

- Top-N (binary): McNemar + paired bootstrap difference-in-proportion CI.
- Continuous: Wilcoxon signed-rank + Vargha-Delaney A12.
- Multiple comparisons within a research question corrected with Holm.
- Failed runs are reported, never silently dropped.
"""
from __future__ import annotations

import math
import random
from itertools import combinations

import numpy as np


def _pad(xs: list) -> np.ndarray:
    out = np.array([float(x) for x in xs])
    return out


def mcnemar(pair_a: list[float], pair_b: list[float], alpha: float = 0.05) -> dict:
    b = sum(1 for a, b in zip(pair_a, pair_b) if a and not b)
    c = sum(1 for a, b in zip(pair_a, pair_b) if (not a) and b)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0, "usable": False}
    from scipy.stats import binomtest
    p = 2 * binomtest(min(b, c), n, 0.5).pvalue
    return {"b": b, "c": c, "p": min(1.0, p)}


def wilcoxon(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    nonzero = [x for x in d if x != 0]
    if len(nonzero) < 3:
        return {"statistic": 0.0, "p": 1.0, "n": len(nonzero), "usable": False}
    from scipy.stats import wilcoxon as w
    try:
        r = w(nonzero)
        return {"statistic": float(r.statistic), "p": float(r.pvalue), "n": len(nonzero), "usable": True}
    except ValueError:
        return {"statistic": 0.0, "p": 1.0, "n": len(nonzero), "usable": False}


def a12(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 0.5
    n_ab = len(a) * len(b)
    gt = 0.0
    for x in a:
        gt += np.sum(x > b) + 0.5 * np.sum(x == b)
    return gt / n_ab


def bootstrap_diff_ci(a: list[float], b: list[float], seed: int = 1,
                      B: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        mean_a = sum(a[i] for i in idx) / n
        mean_b = sum(b[i] for i in idx) / n
        diffs.append(mean_a - mean_b)
    diffs.sort()
    lo = diffs[int((alpha / 2) * B)]
    hi = diffs[min(B - 1, int((1 - alpha / 2) * B))]
    return (sum(a) / n - sum(b) / n, lo, hi)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjustment; input order preserved."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for r, i in enumerate(order, 1):
        running = max(running, (m - r + 1) * pvals[i])
        adj[i] = min(1.0, running)
    return adj
