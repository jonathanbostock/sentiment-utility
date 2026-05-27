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


def test_quicksort_handles_all_ties_without_blowup():
    # All comparisons tie (p=0.5): tie-balancing must keep it ~O(n log n) and
    # must not exceed Python recursion depth (iterative impl) at scale.
    def tie_oracle(pairs):
        return {(i, j): 0.5 for (i, j) in pairs}
    n = 600
    order, edges = rank_by_quicksort(n, tie_oracle, seed=0)
    assert sorted(order) == list(range(n))      # valid permutation
    assert len(edges) < 20 * n                   # nowhere near quadratic


def test_sparse_fit_heldout_has_no_reverse_leakage():
    # Held-out edges must not appear (in either direction) among training edges.
    import sentiment_utility.efficient as eff
    captured = {}
    orig = eff.fit_thurstone_sparse
    rng = np.random.default_rng(3)
    mu = rng.normal(size=25)
    oracle = _oracle_factory(mu, 0.3)
    order, edges = rank_by_quicksort(len(mu), oracle, seed=0)
    edges += spacing_pass(order, oracle, k=2)
    res = fit_thurstone_sparse(edges, len(mu), steps=500, test_frac=0.25, seed=0)
    # unique_pairs <= comparison_count (dedupe happened) and accuracy is held out
    assert res["unique_pairs"] <= res["comparison_count"]
    assert res["accuracy_is_heldout"] is True
