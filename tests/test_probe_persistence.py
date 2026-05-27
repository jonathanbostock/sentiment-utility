import numpy as np

from sentiment_utility.probe import (
    apply_probe,
    common_token_prefix,
    fit_deployable_probe,
    load_probe,
    save_probe,
)


def test_fit_apply_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 12))
    w = rng.normal(size=12)
    y = X @ w + 0.01 * rng.normal(size=150)
    p = fit_deployable_probe(X, y, alpha=0.1)
    pred = apply_probe(X, p)
    assert np.corrcoef(pred, y)[0, 1] > 0.99


def test_save_load_roundtrip(tmp_path):
    p = {"coef": [1.0, 2.0, 3.0], "intercept": 0.5, "alpha": 1.0, "best_layer": 7}
    path = tmp_path / "probe.json"
    save_probe(path, p)
    q = load_probe(path)
    assert q["best_layer"] == 7 and np.allclose(q["coef"], p["coef"])


def test_common_token_prefix():
    assert common_token_prefix([[1, 2, 3, 9], [1, 2, 3, 8], [1, 2, 3, 7, 7]]) == [1, 2, 3]
    assert common_token_prefix([[5, 1], [6, 1]]) == []
    assert common_token_prefix([[1, 2]]) == [1, 2]
