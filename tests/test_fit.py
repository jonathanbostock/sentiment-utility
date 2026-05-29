import math
import numpy as np
from sentiment_utility.fit import normalize_edges, fit_caseV_mle, predict_matrix_caseV


def _planted_edges(mu_true, reps=1):
    """Dense soft-p edges from a known mu (Case V) — used to check recovery."""
    n = len(mu_true)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = float(0.5 * (1 + math.erf((mu_true[i] - mu_true[j]) / 2.0)))  # Phi(Δ/√2), Δ=(μi-μj)
            for _ in range(reps):
                rows.append({"i": i, "j": j, "p_util": p, "mode": "logprob"})
    return rows


def test_normalize_sample_vs_soft():
    rows = [
        {"i": 0, "j": 1, "p_util": 0.75, "mode": "sample", "wins_i": 3, "wins_j": 1},
        {"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"},
    ]
    i, j, wp, wn = normalize_edges(rows)
    assert list(i) == [0, 0] and list(j) == [1, 1]
    assert wp[0] == 3 and wn[0] == 1
    assert np.isclose(wp[1], 0.9) and np.isclose(wn[1], 0.1)


def test_recovers_planted_order():
    mu_true = np.array([-2.0, -0.5, 0.5, 2.0])
    rows = _planted_edges(mu_true)
    res = fit_caseV_mle(rows, n=4, steps=1500, seed=0)
    mu = res["mu"]
    assert np.isclose(mu.mean(), 0.0, atol=1e-6)            # centered gauge
    assert list(np.argsort(mu)) == list(np.argsort(mu_true))  # correct order
    assert np.corrcoef(mu, mu_true)[0, 1] > 0.99


def test_divergence_is_bounded_in_Phi():
    # item 0 wins ALL comparisons -> mu_0 -> +inf, but predicted Phi stays in [0,1]
    rows = []
    for j in range(1, 4):
        rows.append({"i": 0, "j": j, "p_util": 1.0 - 1e-9, "mode": "logprob"})
    for i in range(1, 4):
        for j in range(1, 4):
            if i != j:
                rows.append({"i": i, "j": j, "p_util": 0.5, "mode": "logprob"})
    res = fit_caseV_mle(rows, n=4, steps=1000, seed=0)
    P = predict_matrix_caseV(res["mu"])
    assert np.all(P >= 0.0) and np.all(P <= 1.0)
    assert P[0, 1] > 0.9   # item 0 dominates
    assert res["mu"][0] == max(res["mu"])  # large but finite
