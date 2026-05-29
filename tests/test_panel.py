import numpy as np
from sentiment_utility.panel import (
    decisiveness, decisiveness_raw, transitivity_fas, transitivity_triad,
)


def test_decisiveness_extremes():
    assert np.isclose(decisiveness(np.array([-50.0, 50.0])), 1.0)   # saturated
    assert np.isclose(decisiveness(np.array([0.0, 0.0])), 0.0)      # indifferent


def test_decisiveness_raw_affine_to_p_pick_higher():
    rows = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"}]
    d = decisiveness_raw(rows)
    assert np.isclose(0.5 + 0.5 * d, 0.9)   # p_pick_higher = 0.5 + 0.5 D


def test_transitivity_fas_perfect_vs_cycle():
    order = [0, 1, 2]   # best -> worst
    good = [{"i": 0, "j": 1, "p_util": 0.9, "mode": "logprob"},
            {"i": 1, "j": 2, "p_util": 0.9, "mode": "logprob"},
            {"i": 0, "j": 2, "p_util": 0.9, "mode": "logprob"}]
    assert np.isclose(transitivity_fas(good, order), 1.0)
    bad = good + [{"i": 2, "j": 0, "p_util": 0.99, "mode": "logprob"}]
    assert transitivity_fas(bad, order) < 1.0


def test_transitivity_triad_detects_cycle():
    # tuples are (p_ab, p_bc, p_ca). Even a clean transitive triple has nonzero soft
    # cycle mass because probabilities multiply (0.95^2*0.05 + 0.05^2*0.95 ~= 0.048).
    transitive = [(0.95, 0.95, 0.05)]   # a>b, b>c, a>c -> consistent
    cyclic = [(0.9, 0.9, 0.9)]          # a>b, b>c, c>a -> 3-cycle, mass ~0.73
    assert transitivity_triad(transitive) > 0.9
    assert transitivity_triad(cyclic) < 0.5
