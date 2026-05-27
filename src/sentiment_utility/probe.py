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


def _model_input_device(model):
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_activations(tok, model, items, batch_size=16):
    """Extract last-token hidden states for neutral concept-only chat prompts."""
    import torch

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    device = _model_input_device(model)
    per_layer_batches = None

    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": concept}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for concept in batch
            ]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                device
            )
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states
            if per_layer_batches is None:
                per_layer_batches = [[] for _ in hidden_states]
            for layer, hidden in enumerate(hidden_states):
                per_layer_batches[layer].append(hidden[:, -1, :].detach().cpu().numpy())

    if per_layer_batches is None:
        return {}
    return {
        layer: np.concatenate(batches, axis=0)
        for layer, batches in enumerate(per_layer_batches)
    }
