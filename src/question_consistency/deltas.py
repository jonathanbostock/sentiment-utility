from __future__ import annotations

import numpy as np


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    std = float(x.std())
    if std == 0.0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(x.mean())) / std


def score_deltas(items, base_scores, char_scores, top_k=20) -> dict:
    items = list(items)
    base = np.asarray(base_scores, dtype=np.float64)
    char = np.asarray(char_scores, dtype=np.float64)
    if len(items) != len(base) or len(items) != len(char):
        raise ValueError("items, base_scores, and char_scores must have the same length")

    z_base = zscore(base)
    z_char = zscore(char)
    delta = z_char - z_base
    top_k = min(int(top_k), len(items))

    order_positive = np.argsort(-delta)[:top_k]
    order_negative = np.argsort(delta)[:top_k]

    if len(items) < 2 or base.std() == 0.0 or char.std() == 0.0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(base, char)[0, 1])

    return {
        "pearson_r": pearson_r,
        "mean_abs_delta": float(np.mean(np.abs(delta))) if len(delta) else float("nan"),
        "more_positive": [
            {"item": items[i], "delta": float(delta[i])} for i in order_positive
        ],
        "more_negative": [
            {"item": items[i], "delta": float(delta[i])} for i in order_negative
        ],
        "delta": {item: float(value) for item, value in zip(items, delta)},
    }
