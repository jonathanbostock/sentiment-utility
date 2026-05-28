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


def _format_oct_constitution(path):
    """Read an OCT constitution JSON (list of {trait, questions}) and return a
    system-prompt formatted as: 'You are an AI assistant with the following
    characteristics: <trait1> <trait2> ...'"""
    data = json.loads(Path(path).read_text())
    traits = [entry["trait"].strip() for entry in data if entry.get("trait")]
    return "You are an AI assistant with the following characteristics: " + " ".join(traits)


def run(model_id, name, adapter, items_path, out_root, seed=0,
        revision=None, system_prompt=None, oct_constitution=None, load_in_4bit=False):
    run_dir = Path(out_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)
    items = _load_items(items_path)
    if oct_constitution and not system_prompt:
        system_prompt = _format_oct_constitution(oct_constitution)
    log.info("commit=%s loading %s (revision=%s, adapter=%s, sysprompt=%s)",
             _git_commit(), model_id, revision, adapter,
             (system_prompt[:80] + "...") if system_prompt else None)
    tok, model = load_model(model_id, "bfloat16", revision=revision, load_in_4bit=load_in_4bit)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

    log.info("efficient elicitation over %d items", len(items))
    oracle = lambda pairs: compare_pairs(tok, model, items, pairs, batch_size=64,
                                         system_prompt=system_prompt)
    order, edges = rank_by_quicksort(len(items), oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(edges, len(items), test_frac=0.2, seed=seed)
    mu = np.asarray(fit["mu"], dtype=np.float64)
    sigma = np.asarray(fit["sigma"], dtype=np.float64)

    pref = predict_pref_matrix(mu, sigma)
    metrics = {
        "model_id": model_id,
        "revision": revision,
        "adapter": adapter,
        "system_prompt_used": bool(system_prompt),
        "system_prompt_preview": (system_prompt[:200] + "...") if system_prompt else None,
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
    ap.add_argument("--revision", default=None, help="HF model revision/branch (e.g. stage1-step10000).")
    ap.add_argument("--system-prompt-file", default=None,
                    help="Path to a plain-text file used verbatim as the system prompt.")
    ap.add_argument("--oct-constitution", default=None,
                    help="Path to an OCT JSON constitution; trait assertions are joined into "
                         "'You are an AI assistant with the following characteristics: ...'.")
    ap.add_argument("--items-path", default="config/items_500.yaml")
    ap.add_argument("--out-root", default="runs/mu")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Load base model in 4-bit (NF4 via bitsandbytes), for fitting "
                         "large models on a single GPU. Requires bitsandbytes.")
    args = ap.parse_args()
    sp = Path(args.system_prompt_file).read_text() if args.system_prompt_file else None
    run(args.model_id, args.name, args.adapter, args.items_path, args.out_root,
        revision=args.revision, system_prompt=sp, oct_constitution=args.oct_constitution,
        load_in_4bit=args.load_in_4bit)


if __name__ == "__main__":
    main()
