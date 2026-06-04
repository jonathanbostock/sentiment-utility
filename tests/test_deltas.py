import numpy as np

from question_consistency.deltas import score_deltas, zscore


def test_zscore_props():
    z = zscore([1, 2, 3, 4, 5])
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_score_deltas_identifies_shift():
    items = list("abcde")
    base = np.array([0, 1, 2, 3, 4.0])
    char = base.copy()
    char[0] += 10
    d = score_deltas(items, base, char, top_k=2)
    assert d["more_positive"][0]["item"] == "a"
    assert "pearson_r" in d and "mean_abs_delta" in d
