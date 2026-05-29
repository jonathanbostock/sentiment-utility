from __future__ import annotations

import numpy as np

from .fit import predict_matrix_caseV


def decisiveness(mu) -> float:
    """mean|2 Phi - 1| over unordered pairs of the fitted Case V matrix. Bounded [0,1]."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows) -> float:
    """mean|2 p_util - 1| over observed edges (resolution-limited diagnostic)."""
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def transitivity_fas(rows, order) -> float:
    """1 - confidence-weighted fraction of edges pointing backward vs `order`
    (best->worst). Weight = |2 p_util - 1|."""
    rank = {item: r for r, item in enumerate(order)}
    num = den = 0.0
    for r in rows:
        i, j, p = r["i"], r["j"], float(r["p_util"])
        if i not in rank or j not in rank:
            continue
        w = abs(2 * p - 1)
        prefers_i = p > 0.5
        i_before_j = rank[i] < rank[j]   # i ranked better
        backward = prefers_i != i_before_j
        den += w
        if backward:
            num += w
    return 1.0 - (num / den) if den else float("nan")


def transitivity_triad(triads) -> float:
    """triads: list of (p_ab, p_bc, p_ca). Returns 1 - mean soft cycle mass."""
    if not triads:
        return float("nan")
    masses = []
    for p_ab, p_bc, p_ca in triads:
        fwd = p_ab * p_bc * p_ca
        bwd = (1 - p_ab) * (1 - p_bc) * (1 - p_ca)
        masses.append(fwd + bwd)
    return 1.0 - float(np.mean(masses))
