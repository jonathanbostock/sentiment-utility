import numpy as np
from math import erf, sqrt
from sentiment_utility.efficient import rank_by_quicksort, spacing_pass, fit_thurstone_sparse


def _phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _oracle_factory(mu, sigma=0.4):
    def oracle(pairs):
        out = {}
        for i, j in pairs:
            out[(i, j)] = _phi((mu[i] - mu[j]) / sqrt(2) / sigma)
        return out

    return oracle


def test_quicksort_recovers_order():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=40)
    oracle = _oracle_factory(mu, sigma=0.2)
    order, edges = rank_by_quicksort(len(mu), oracle, seed=0)
    true_desc = list(np.argsort(-mu))
    pos = {idx: r for r, idx in enumerate(order)}
    tpos = {idx: r for r, idx in enumerate(true_desc)}
    diffs = sum(abs(pos[i] - tpos[i]) for i in range(len(mu)))
    assert diffs < len(mu)


def test_quicksort_subquadratic_comparison_count():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=64)
    counts = {"n": 0}
    base = _oracle_factory(mu, 0.2)

    def counting(pairs):
        counts["n"] += len(pairs)
        return base(pairs)

    rank_by_quicksort(len(mu), counting, seed=0)
    assert counts["n"] < 64 * 63
    assert counts["n"] < 8 * 64


def test_sparse_fit_recovers_ranking():
    rng = np.random.default_rng(2)
    mu = rng.normal(size=30)
    oracle = _oracle_factory(mu, 0.3)
    order, edges = rank_by_quicksort(len(mu), oracle, seed=0)
    edges += spacing_pass(order, oracle, k=2)
    res = fit_thurstone_sparse(edges, len(mu), steps=3000, seed=0)
    from scipy.stats import spearmanr

    rho = spearmanr(res["mu"], mu).statistic
    assert rho > 0.9
    assert res["comparison_count"] == len(edges)
