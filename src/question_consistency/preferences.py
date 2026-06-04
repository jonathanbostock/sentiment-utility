from __future__ import annotations
import numpy as np


def combine_orderings(n: int, ordered: dict[tuple[int, int], float]) -> np.ndarray:
    pref = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_ij = ordered[(i, j)]   # P(pick i | A=i, B=j)
            p_ji = ordered[(j, i)]   # P(pick j | A=j, B=i)
            pref[i, j] = 0.5 * (p_ij + (1.0 - p_ji))
    return pref
