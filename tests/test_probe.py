import numpy as np
from question_consistency.probe import train_probe, probe_all_layers


def test_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16))
    w = rng.normal(size=16)
    y = X @ w + 0.01 * rng.normal(size=200)
    res = train_probe(X, y, seed=0, alpha=0.1)
    assert res["test_r2"] > 0.95
    assert res["pairwise_accuracy"] > 0.9


def test_noise_layer_scores_low_best_layer_picks_signal():
    rng = np.random.default_rng(1)
    N = 200
    y = rng.normal(size=N)
    signal = np.outer(y, rng.normal(size=8)) + 0.01 * rng.normal(size=(N, 8))
    noise = rng.normal(size=(N, 8))
    res = probe_all_layers({0: noise, 1: signal}, y, seed=0, alpha=0.1)
    assert res["best_layer"] == 1
    assert res["per_layer"][1]["test_r2"] > res["per_layer"][0]["test_r2"]
