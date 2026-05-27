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


def _expand_past_key_values(past_key_values, batch_size: int):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "batch_repeat_interleave"):
        return past_key_values.batch_repeat_interleave(batch_size)

    def expand_value(value):
        if hasattr(value, "dim") and value.dim() > 0 and value.shape[0] == 1:
            return value.expand(batch_size, *value.shape[1:]).contiguous()
        if isinstance(value, tuple):
            return tuple(expand_value(v) for v in value)
        if isinstance(value, list):
            return [expand_value(v) for v in value]
        return value

    return expand_value(past_key_values)


def _past_from_output(output):
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, (tuple, list)) and len(output) > 1:
        return output[1]
    raise AttributeError("model output does not expose past_key_values")


def probe_score_concepts(tok, model, items, best_layer, probe, batch_size=16):
    """Score concept prompts using one shared-prefix KV cache and a deployable probe."""
    import torch

    items = list(items)
    if not items:
        return np.asarray([], dtype=np.float64)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": concept}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for concept in items
    ]
    tokenized = [tok(text, add_special_tokens=False)["input_ids"] for text in texts]
    prefix = common_token_prefix(tokenized)
    prefix_len = len(prefix)
    suffixes = [ids[prefix_len:] for ids in tokenized]

    device = _model_input_device(model)
    scored = np.empty(len(items), dtype=np.float64)

    with torch.no_grad():
        prefix_hidden_by_layer = None
        prefix_past = None
        if prefix:
            prefix_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            prefix_mask = torch.ones_like(prefix_ids, device=device)
            prefix_out = model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                output_hidden_states=True,
            )
            prefix_hidden_by_layer = prefix_out.hidden_states
            prefix_past = _past_from_output(prefix_out)

        for start in range(0, len(items), batch_size):
            batch_suffixes = suffixes[start : start + batch_size]
            non_empty = [i for i, suffix in enumerate(batch_suffixes) if suffix]

            if len(non_empty) < len(batch_suffixes):
                if prefix_hidden_by_layer is None:
                    raise ValueError("empty prompt cannot be scored without prefix hidden states")
                empty_rows = [i for i, suffix in enumerate(batch_suffixes) if not suffix]
                repeated = np.repeat(
                    prefix_hidden_by_layer[best_layer][:, -1, :].detach().float().cpu().numpy(),
                    len(empty_rows),
                    axis=0,
                )
                scored[start + np.asarray(empty_rows)] = apply_probe(repeated, probe)

            if not non_empty:
                continue

            active_suffixes = [batch_suffixes[i] for i in non_empty]
            active_batch = len(active_suffixes)
            max_suffix = max(len(suffix) for suffix in active_suffixes)
            suffix_ids = torch.full(
                (active_batch, max_suffix),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            suffix_mask = torch.zeros(
                (active_batch, max_suffix), dtype=torch.long, device=device
            )
            for row, suffix in enumerate(active_suffixes):
                length = len(suffix)
                suffix_ids[row, :length] = torch.tensor(
                    suffix, dtype=torch.long, device=device
                )
                suffix_mask[row, :length] = 1

            if prefix_len:
                prefix_mask = torch.ones(
                    (active_batch, prefix_len), dtype=torch.long, device=device
                )
                attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
                past_key_values = _expand_past_key_values(prefix_past, active_batch)
            else:
                attention_mask = suffix_mask
                past_key_values = None

            position_ids = (
                prefix_len
                + torch.arange(max_suffix, dtype=torch.long, device=device).unsqueeze(0)
            ).expand(active_batch, -1)
            out = model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[best_layer]
            last_indices = torch.tensor(
                [len(suffix) - 1 for suffix in active_suffixes],
                dtype=torch.long,
                device=device,
            )
            rows = torch.arange(active_batch, dtype=torch.long, device=device)
            features = hidden[rows, last_indices, :].detach().float().cpu().numpy()
            scored[start + np.asarray(non_empty)] = apply_probe(features, probe)

    return scored
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
