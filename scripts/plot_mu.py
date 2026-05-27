"""Plot cross-model coherence + preference-agreement from results/mu/<name>/.

Makes: (1) coherence-vs-scale for the Gemma series, (2) cross-model Spearman heatmap.
Run locally after pulling. Saves PDFs into results/mu/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


# approx active params (B) for the scale axis
SIZE_B = {"gemma-3-1b": 1.0, "gemma-3-4b": 4.0, "gemma-3-12b": 12.0, "gemma-3-27b": 27.0,
          "llama-3.1-8b": 8.0, "qwen3-8b": 8.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mu")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    mus = {m: json.loads((root / m / "mu.json").read_text()) for m in args.models}
    met = {m: json.loads((root / m / "metrics.json").read_text()) for m in args.models}

    # (1) Gemma scale curve: mu_std + completeness vs size
    gem = [m for m in args.models if m.startswith("gemma-3")]
    gem = sorted(gem, key=lambda m: SIZE_B[m])
    if len(gem) >= 2:
        df = pd.DataFrame({"size_B": [SIZE_B[m] for m in gem],
                           "mu_std": [met[m]["mu_std"] for m in gem],
                           "completeness": [met[m]["completeness"] for m in gem],
                           "model": gem})
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        sns.lineplot(data=df, x="size_B", y="mu_std", marker="o", ax=ax1, color="#4C72B0")
        ax1.set_xscale("log"); ax1.set_xlabel("Gemma-3 size (B params, log)")
        ax1.set_ylabel("mu_std (decisiveness / SNR)", color="#4C72B0")
        ax1.set_xticks([SIZE_B[m] for m in gem]); ax1.set_xticklabels([m.split("-")[-1] for m in gem])
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="size_B", y="completeness", marker="s", ax=ax2, color="#C44E52")
        ax2.set_ylabel("completeness", color="#C44E52")
        ax1.set_title("Sentiment coherence rises with scale (Gemma-3)")
        plt.tight_layout(); plt.savefig(root / "coherence_vs_scale.pdf"); plt.close()

    # (2) cross-model Spearman heatmap
    shared = sorted(set.intersection(*[set(d) for d in mus.values()]))
    M = np.zeros((len(args.models), len(args.models)))
    for i, a in enumerate(args.models):
        for j, b in enumerate(args.models):
            M[i, j] = spearmanr([mus[a][k] for k in shared], [mus[b][k] for k in shared]).statistic
    plt.figure(figsize=(7, 6))
    sns.heatmap(M, xticklabels=args.models, yticklabels=args.models, annot=True, fmt=".2f",
                vmin=0, vmax=1, cmap="viridis")
    plt.title("Cross-model preference agreement (Spearman of mu)")
    plt.tight_layout(); plt.savefig(root / "preference_agreement_heatmap.pdf"); plt.close()
    print("wrote coherence_vs_scale.pdf + preference_agreement_heatmap.pdf")


if __name__ == "__main__":
    main()
