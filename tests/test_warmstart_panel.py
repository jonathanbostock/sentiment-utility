import numpy as np

from sentiment_utility.fit import fit_caseV_mle
from sentiment_utility.panel import compute_panel, decisiveness


def _rows(n=20, seed=0):
    rng = np.random.default_rng(seed)
    true_mu = np.sort(rng.normal(size=n))[::-1]
    rows = []
    for _ in range(2000):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        p = 1 / (1 + np.exp(-(true_mu[i] - true_mu[j])))
        rows.append({"i": int(i), "j": int(j), "p_util": float(p), "mode": "logprob"})
    return rows


def test_mu_init_converges_in_fewer_steps():
    rows = _rows()
    cold = fit_caseV_mle(rows, n=20, steps=2000, seed=0)["mu"]
    warm = fit_caseV_mle(rows, n=20, steps=100, seed=0, mu_init=cold)["mu"]
    assert np.corrcoef(cold, warm)[0, 1] > 0.999


def test_compute_panel_uses_mu_init():
    rows = _rows()
    edges = {"elo": rows, "reverse": [], "triad": [], "cross": []}
    converged = fit_caseV_mle(rows, n=20, steps=2000, seed=0)["mu"]
    target = float(decisiveness(converged))
    # Warm-started panel at very few steps should already match the converged decisiveness.
    warm_panel = compute_panel(edges, n=20, seed=0, fit_steps=10, mu_init=converged)
    assert abs(warm_panel["decisiveness"]["point"] - target) < 0.02
    # Cold panel at the same few steps should be far from converged (mu starts at 0).
    cold_panel = compute_panel(edges, n=20, seed=0, fit_steps=10)
    assert abs(cold_panel["decisiveness"]["point"] - target) > 0.05
