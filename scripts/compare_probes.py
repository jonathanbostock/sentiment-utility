"""Compare sentiment probes across character models that share a base.

Tests the hypothesis: if a persona LoRA barely rotates the sentiment direction, the
base probe transfers to the character model and we can skip per-model probe training.

For each (base, character) pair it:
  - fits a deployable probe at a COMMON layer for each model (so coef vectors are
    comparable), using each model's own elicited mu (from its run dir),
  - reports cosine similarity of the two probe weight vectors, and
  - cross-transfer R^2: base probe applied to the character model's activations
    predicting the character's mu, and vice versa.

Run AFTER scripts/run_character.py has produced runs/character/<name>/ for each model
(needs elicited_mu.json there). Reloads each model to extract activations at the common
layer. Use the venv python directly (do not let uv re-resolve torch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from question_consistency.characters import load_character_model, model_specs
from question_consistency.probe import apply_probe, extract_activations, fit_deployable_probe


def _load_run(run_dir: Path):
    mu_map = json.loads((run_dir / "elicited_mu.json").read_text())
    probe = json.loads((run_dir / "probe.json").read_text())
    return mu_map, int(probe["best_layer"])


def _r2(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare base vs character sentiment probes.")
    ap.add_argument("--base", default="base")
    ap.add_argument("--characters", nargs="+", default=["loving"])
    ap.add_argument("--out-root", default="runs/character")
    ap.add_argument("--items-train-path", default="config/datasets/items_500.yaml")
    ap.add_argument("--layer", type=int, default=None, help="Common layer (default: base best_layer).")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    from question_consistency.io_utils import load_items

    specs = {s["name"]: s for s in model_specs()}
    train_items = load_items(args.items_train_path)

    base_mu_map, base_best = _load_run(out_root / args.base)
    layer = args.layer if args.layer is not None else base_best
    order = train_items  # mu maps are keyed by item; align by this order

    def fit_at_layer(spec_name):
        spec = specs[spec_name]
        mu_map, _ = _load_run(out_root / spec_name)
        y = np.array([mu_map[it] for it in order], dtype=np.float64)
        tok, model = load_character_model(spec)
        hidden = extract_activations(tok, model, order, batch_size=16)
        X = hidden[layer]
        probe = fit_deployable_probe(X, y)
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()
        return probe, X, y

    base_probe, base_X, base_y = fit_at_layer(args.base)
    results = {"layer": layer, "base": args.base, "characters": {}}
    for name in args.characters:
        char_probe, char_X, char_y = fit_at_layer(name)
        cb = np.asarray(base_probe["coef"]); cc = np.asarray(char_probe["coef"])
        cos = float(cb @ cc / (np.linalg.norm(cb) * np.linalg.norm(cc)))
        results["characters"][name] = {
            "coef_cosine_sim": cos,
            "base_probe_on_char_r2": _r2(apply_probe(char_X, base_probe), char_y),
            "char_probe_on_base_r2": _r2(apply_probe(base_X, char_probe), base_y),
            "char_probe_on_char_r2": _r2(apply_probe(char_X, char_probe), char_y),
        }
        print(name, json.dumps(results["characters"][name], indent=2))

    out = out_root / "probe_comparison.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
