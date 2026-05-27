from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

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


def fit_deployable_probe(X, y, alpha=1.0) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    model = Ridge(alpha=alpha).fit(X, y)
    return {
        "coef": model.coef_.astype(float),
        "intercept": float(model.intercept_),
        "alpha": float(alpha),
    }


def apply_probe(X, probe) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    coef = np.asarray(probe["coef"], dtype=np.float64)
    return X @ coef + float(probe["intercept"])


def save_probe(path, probe: dict) -> None:
    path = Path(path)
    data = dict(probe)
    data["coef"] = np.asarray(data["coef"], dtype=float).tolist()
    if "intercept" in data:
        data["intercept"] = float(data["intercept"])
    if "alpha" in data:
        data["alpha"] = float(data["alpha"])
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_probe(path) -> dict:
    data = json.loads(Path(path).read_text())
    data["coef"] = np.asarray(data["coef"], dtype=np.float64)
    return data


def common_token_prefix(seqs) -> list[int]:
    seqs = [list(seq) for seq in seqs]
    if not seqs:
        return []

    prefix = []
    for values in zip(*seqs):
        first = values[0]
        if any(value != first for value in values[1:]):
            break
        prefix.append(first)
    return prefix


def _snapshot_prefix_kv(past):
    """Return a stable list of (key, value) tensors (batch dim 1) for the prefix cache.

    Never mutates `past`; works across transformers Cache APIs (legacy tuples,
    DynamicCache with .to_legacy_cache(), or .layers / .key_cache attributes).
    """
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        try:
            return [(k, v) for (k, v) in past.to_legacy_cache()]
        except Exception:
            pass
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))
    if hasattr(past, "layers"):
        out = []
        for layer in past.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None:
                k, v = layer  # tuple-like layer
            out.append((k, v))
        return out
    return [(k, v) for (k, v) in past]  # legacy tuple-of-tuples


def _build_batch_cache(snapshot, batch_size):
    """Build a FRESH cache for `batch_size` rows from an immutable prefix snapshot."""
    legacy = tuple(
        (
            k.expand(batch_size, *k.shape[1:]).contiguous(),
            v.expand(batch_size, *v.shape[1:]).contiguous(),
        )
        for (k, v) in snapshot
    )
    try:
        from transformers import DynamicCache
    except Exception:
        return legacy
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(legacy):
        cache.update(k, v, layer_idx)
    return cache


def _past_from_output(output):
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    raise AttributeError("model output does not expose past_key_values")


def _score_features_full(tok, model, tokenized, best_layer, batch_size, device):
    """No-cache fallback: forward each full concept prompt (left-padded), take the
    best-layer last-token hidden state. Identical mechanism to extract_activations,
    so correct by construction."""
    import torch

    feats = [None] * len(tokenized)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for start in range(0, len(tokenized), batch_size):
            batch = tokenized[start : start + batch_size]
            maxlen = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), maxlen), dtype=torch.long, device=device)
            for row, ids in enumerate(batch):  # LEFT pad
                input_ids[row, maxlen - len(ids) :] = torch.tensor(ids, device=device)
                mask[row, maxlen - len(ids) :] = 1
            out = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
            h = out.hidden_states[best_layer][:, -1, :].detach().float().cpu().numpy()
            for row in range(len(batch)):
                feats[start + row] = h[row]
    return np.asarray(feats, dtype=np.float64)


def probe_score_concepts(tok, model, items, best_layer, probe, batch_size=16, use_kv_cache=True):
    """Score concepts by applying a deployable probe to best-layer last-token activations.

    Renders each concept as a neutral chat prompt and extracts one activation per concept
    (single forward pass each, batched). When use_kv_cache is True and the prompts share a
    non-trivial token prefix, the prefix is encoded once and its KV cache reused across all
    concept batches; otherwise (or as a guaranteed-correct fallback) full prompts are run.
    """
    import torch

    items = list(items)
    if not items:
        return np.asarray([], dtype=np.float64)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": concept}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for concept in items
    ]
    tokenized = [tok(text, add_special_tokens=False)["input_ids"] for text in texts]
    device = _model_input_device(model)

    prefix = common_token_prefix(tokenized)
    prefix_len = len(prefix)
    # Only worth caching a non-trivial shared prefix and when every suffix is non-empty.
    if not use_kv_cache or prefix_len == 0 or any(len(ids) <= prefix_len for ids in tokenized):
        feats = _score_features_full(tok, model, tokenized, best_layer, batch_size, device)
        return apply_probe(feats, probe)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    suffixes = [ids[prefix_len:] for ids in tokenized]
    feats = [None] * len(items)

    with torch.no_grad():
        prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_out = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            output_hidden_states=True,
        )
        snapshot = _snapshot_prefix_kv(_past_from_output(prefix_out))

        for start in range(0, len(items), batch_size):
            batch = suffixes[start : start + batch_size]
            bs = len(batch)
            maxlen = max(len(s) for s in batch)
            suffix_ids = torch.full((bs, maxlen), pad_id, dtype=torch.long, device=device)
            suffix_mask = torch.zeros((bs, maxlen), dtype=torch.long, device=device)
            for row, s in enumerate(batch):  # RIGHT pad (real tokens first)
                suffix_ids[row, : len(s)] = torch.tensor(s, device=device)
                suffix_mask[row, : len(s)] = 1
            attention_mask = torch.cat(
                [torch.ones((bs, prefix_len), dtype=torch.long, device=device), suffix_mask], dim=1
            )
            position_ids = (
                prefix_len + torch.arange(maxlen, device=device).unsqueeze(0)
            ).expand(bs, -1)
            out = model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=_build_batch_cache(snapshot, bs),
                use_cache=True,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[best_layer]
            last = torch.tensor([len(s) - 1 for s in batch], device=device)
            rows = torch.arange(bs, device=device)
            h = hidden[rows, last, :].detach().float().cpu().numpy()
            for row in range(bs):
                feats[start + row] = h[row]

    return apply_probe(np.asarray(feats, dtype=np.float64), probe)


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
                # cast to float32: numpy has no bf16 dtype
                per_layer_batches[layer].append(
                    hidden[:, -1, :].detach().float().cpu().numpy()
                )

    if per_layer_batches is None:
        return {}
    return {
        layer: np.concatenate(batches, axis=0)
        for layer, batches in enumerate(per_layer_batches)
    }
