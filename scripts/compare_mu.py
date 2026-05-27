"""Compare elicited mu across models (run locally after pulling runs/mu/<name>/).

Prints: a coherence table (mu_std, completeness, cyclic-triad, held-out fit accuracy),
a cross-model Spearman correlation matrix of mu over shared items, consensus likes/dislikes,
and the biggest cross-family preference disagreements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mu")
    ap.add_argument("--models", nargs="+", required=True, help="subdir names in order")
    args = ap.parse_args()
    root = Path(args.root)

    mus, metrics = {}, {}
    for m in args.models:
        mus[m] = json.loads((root / m / "mu.json").read_text())
        metrics[m] = json.loads((root / m / "metrics.json").read_text())
    shared = sorted(set.intersection(*[set(d) for d in mus.values()]))
    print(f"{len(shared)} shared items across {len(args.models)} models\n")

    print("=== coherence ===")
    print(f"{'model':14s} {'mu_std':>7} {'complete':>9} {'cyclic%':>8} {'fitAcc':>7} {'#comp':>7}")
    for m in args.models:
        x = metrics[m]
        print(f"{m:14s} {x['mu_std']:7.2f} {x['completeness']:9.3f} "
              f"{100*x['cyclic_triad_fraction']:8.2f} {x['heldout_fit_accuracy']:7.3f} {x['comparison_count']:7d}")

    # standardized mu per model over shared items
    Z = {m: (lambda v: (v - v.mean()) / v.std())(np.array([mus[m][i] for i in shared])) for m in args.models}

    print("\n=== cross-model Spearman correlation of mu (preference agreement) ===")
    print(" " * 14 + "".join(f"{m[:10]:>11}" for m in args.models))
    for a in args.models:
        row = "".join(f"{spearmanr([mus[a][i] for i in shared], [mus[b][i] for i in shared]).statistic:11.2f}"
                      for b in args.models)
        print(f"{a:14s}{row}")

    consensus = np.mean([Z[m] for m in args.models], axis=0)
    order = np.argsort(consensus)
    print("\n=== consensus most-liked (mean standardized mu) ===")
    for k in order[::-1][:12]:
        print(f"  +{consensus[k]:.2f}  {shared[k]}")
    print("=== consensus most-disliked ===")
    for k in order[:12]:
        print(f"  {consensus[k]:.2f}  {shared[k]}")

    # biggest disagreements: max spread across models per item
    spread = np.std([Z[m] for m in args.models], axis=0)
    ds = np.argsort(spread)
    print("\n=== biggest cross-model disagreements (std of standardized mu) ===")
    for k in ds[::-1][:12]:
        vals = ", ".join(f"{m.split('-')[0][:5]}{('/'+m.split('-')[-1]) if '-' in m else ''}={Z[m][k]:+.1f}" for m in args.models)
        print(f"  spread={spread[k]:.2f}  {shared[k]:24s} [{vals}]")


if __name__ == "__main__":
    main()
