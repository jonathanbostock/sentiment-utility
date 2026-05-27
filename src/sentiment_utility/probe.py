from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def train_probe(X, y, seed=0, alpha=1.0, test_frac=0.2):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_frac, random_state=seed)
    model = Ridge(alpha=alpha).fit(Xtr, ytr)
    pred = model.predict(Xte)
    ss_res = float(np.sum((yte - pred) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    agree = total = 0
    for a, b in combinations(range(len(yte)), 2):
        if yte[a] == yte[b]:
            continue
        total += 1
        agree += (pred[a] > pred[b]) == (yte[a] > yte[b])
    return {
        "test_r2": float(r2),
        "pairwise_accuracy": float(agree / total) if total else float("nan"),
        "n_test": int(len(yte)),
    }


def probe_all_layers(hidden, y, seed=0, alpha=1.0):
    per_layer = {
        layer: train_probe(X, y, seed=seed, alpha=alpha) for layer, X in hidden.items()
    }
    best_layer = max(per_layer, key=lambda L: per_layer[L]["test_r2"])
    return {
        "per_layer": per_layer,
        "best_layer": int(best_layer),
        "best_r2": float(per_layer[best_layer]["test_r2"]),
    }
