"""Re-fit Thurstonian mu/sigma from a saved edges.jsonl under a chosen estimator.

Edges file format (one JSON per line) — matches what elicit_mu_openai.py emits:
  logprob mode:  {"i":i, "j":j, "p":p, "mode":"logprob", "lpA":..., "lpB":..., ...}
  sample  mode:  {"i":i, "j":j, "p":p, "mode":"sample", "n_samples":N,
                  "a_count":a, "b_count":b, "picks":[...]}

Estimators:
  --estimator jeffreys    (a+0.5)/(a+b+1)                    [conservative, default for sampling]
  --estimator laplace     (a+1)/(a+b+2)                       [Beta(1,1) uniform prior]
  --estimator weak        (a+alpha)/(a+b+2alpha), --alpha A   [adjustable strength]
  --estimator mle         a/(a+b), clipped to (eps, 1-eps)    [no prior; saturates]
  --estimator logprob     P from stored lp_A, lp_B            [for logprob-mode edges]

Writes a refit.json next to the edges file with the new mu/sigma/metrics.

Usage:
  python scripts/refit_edges.py --edges results/mu_openai5/gpt-5-nano/edges.jsonl \\
                                 --items-path config/datasets/items_500.yaml \\
                                 --estimator weak --alpha 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from question_consistency.efficient import fit_thurstone_sparse
from question_consistency.thurstone import predict_pref_matrix
from question_consistency.metrics import (
    completeness, cyclic_triad_fraction, expected_cycle_probability,
)
from question_consistency.io_utils import load_items as _load_items
from run_character import _jsonable


def _p_from_record(rec, estimator, alpha, eps):
    mode = rec.get("mode")
    if estimator == "logprob":
        lpA, lpB = rec.get("lpA"), rec.get("lpB")
        if lpA is None and lpB is None: return 0.5
        if lpA is None: return eps
        if lpB is None: return 1 - eps
        m = max(lpA, lpB)
        eA, eB = np.exp(lpA - m), np.exp(lpB - m)
        return float(eA / (eA + eB))
    if mode != "sample":
        return float(rec.get("p", 0.5))
    a = int(rec["a_count"]); b = int(rec["b_count"])
    n = a + b
    if n == 0: return 0.5
    if estimator == "jeffreys":
        return (a + 0.5) / (n + 1.0)
    if estimator == "laplace":
        return (a + 1.0) / (n + 2.0)
    if estimator == "weak":
        return (a + alpha) / (n + 2.0 * alpha)
    if estimator == "mle":
        return max(eps, min(1 - eps, a / n))
    raise ValueError(f"unknown estimator {estimator!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="Path to edges.jsonl from a previous run.")
    ap.add_argument("--items-path", default="config/datasets/items_500.yaml")
    ap.add_argument("--estimator", default="jeffreys",
                    choices=["jeffreys", "laplace", "weak", "mle", "logprob"])
    ap.add_argument("--alpha", type=float, default=0.1, help="Prior strength for --estimator weak.")
    ap.add_argument("--eps", type=float, default=1e-3, help="Clip for mle / one-sided lp.")
    ap.add_argument("--out", default=None, help="Output path (default: alongside edges file).")
    args = ap.parse_args()

    items = _load_items(args.items_path)
    n = len(items)
    edges = []
    with open(args.edges) as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            p = _p_from_record(rec, args.estimator, args.alpha, args.eps)
            edges.append((int(rec["i"]), int(rec["j"]), float(p)))
    print(f"loaded {len(edges)} edges from {args.edges}")

    fit = fit_thurstone_sparse(edges, n, test_frac=0.2, seed=0)
    mu = np.asarray(fit["mu"]); sigma = np.asarray(fit["sigma"])
    pref = predict_pref_matrix(mu, sigma)
    metrics = {
        "edges_path": args.edges, "estimator": args.estimator, "alpha": args.alpha,
        "mu_std": float(mu.std()), "completeness": float(completeness(pref)),
        "cyclic_triad_fraction": float(cyclic_triad_fraction(pref)),
        "heldout_fit_accuracy": float(fit["test_accuracy"]),
        "comparison_count": int(fit["comparison_count"]),
        "unique_pairs": int(fit.get("unique_pairs", fit["comparison_count"])),
        "n_items": n,
    }
    out = Path(args.out) if args.out else Path(args.edges).with_name(
        f"refit_{args.estimator}{('_a' + str(args.alpha)) if args.estimator == 'weak' else ''}.json"
    )
    payload = {"metrics": metrics, "mu": {it: float(v) for it, v in zip(items, mu)},
               "sigma": {it: float(v) for it, v in zip(items, sigma)}}
    out.write_text(json.dumps(_jsonable(payload), indent=2))
    print(json.dumps(metrics, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
