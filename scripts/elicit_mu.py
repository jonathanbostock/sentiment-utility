"""Elicit Thurstonian sentiment mu on the shared 500-concept set for one model.

Lightweight (no probe, no 2000-scoring): just efficient O(n log n) elicitation -> sparse
Thurstonian fit -> coherence metrics on the implied preference matrix. Saves per-item mu and
metrics so multiple models can be compared (mu_std = decisiveness; per-item mu = preferences).

Usage: python scripts/elicit_mu.py --model-id google/gemma-3-4b-it --name gemma-3-4b
       python scripts/elicit_mu.py --model-id Qwen/Qwen3-8B --name qwen3-8b
Run with the venv python directly on the pod.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentiment_utility.elicit import compare_pairs, load_model
from sentiment_utility.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from sentiment_utility.thurstone import predict_pref_matrix
from sentiment_utility.metrics import (
    completeness,
    cyclic_triad_fraction,
    expected_cycle_probability,
)

from run_character import _git_commit, _jsonable, _load_items, _setup_logging


def run(model_id, name, adapter, items_path, out_root, seed=0):
    run_dir = Path(out_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    items = _load_items(items_path)
    log.info("commit=%s loading %s (adapter=%s)", _git_commit(), model_id, adapter)
    tok, model = load_model(model_id, "bfloat16")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    log.info("efficient elicitation over %d items", len(items))
    oracle = lambda pairs: compare_pairs(tok, model, items, pairs, batch_size=64)
    order, edges = rank_by_quicksort(len(items), oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(edges, len(items), test_frac=0.2, seed=seed)
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)

    pref = predict_pref_matrix(mu, sigma)
    metrics = {
        "model_id": model_id,
        "adapter": adapter,
        "mu_std": float(mu.std()),
        "mu_mean": float(mu.mean()),
        "heldout_fit_accuracy": float(fit["test_accuracy"]),
        "cyclic_triad_fraction": float(cyclic_triad_fraction(pref)),
        "expected_cycle_probability": float(expected_cycle_probability(pref)),
        "completeness": float(completeness(pref)),
        "comparison_count": int(fit["comparison_count"]),
        "unique_pairs": int(fit.get("unique_pairs", fit["comparison_count"])),
        "n_items": len(items),
    }
    (run_dir / "mu.json").write_text(json.dumps({it: float(v) for it, v in zip(items, mu)}, indent=2))
    (run_dir / "sigma.json").write_text(json.dumps({it: float(v) for it, v in zip(items, sigma)}, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    log.info("done -> %s", run_dir)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Elicit Thurstonian mu for one model.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--out-root", default="runs/mu")
    args = ap.parse_args()
    run(args.model_id, args.name, args.adapter, args.items_path, args.out_root)


if __name__ == "__main__":
    main()
