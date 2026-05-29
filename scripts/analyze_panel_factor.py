"""Cross-model analysis of the coherence metric panel: correlation structure + whether
the metrics collapse onto a single factor.

Reads every <runs-dir>/<model>/panel.json (point estimates), assembles a model x metric
table, sign-aligns so "higher = more coherent", and reports:
  - Pearson + Spearman correlation matrices (seaborn heatmap PDFs)
  - PCA (numpy SVD on standardized metrics): variance explained per component + PC1 loadings
  - a flag for metrics that load weakly on PC1 (candidate orthogonal axes)

Caveat printed in output: several metrics are derived from the same Case V mu fit, so some
correlation is mechanical (by construction) rather than independent empirical signal.

Usage: uv run python scripts/analyze_panel_factor.py --runs-dir runs/oct2k --out-dir runs/oct2k/analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# metric -> +1 if higher is more coherent, -1 if lower is more coherent (sign-align)
METRIC_SIGN = {
    "decisiveness": +1,
    "transitivity_fas": +1,
    "transitivity_triad": +1,
    "order_consistency": +1,
    "q_agreement": +1,
    "unidim_fit_brier": -1,
    "unidim_fit_log_loss": -1,
}
# metrics derived from the same mu fit (correlation partly mechanical)
MU_DERIVED = {"decisiveness", "transitivity_fas", "unidim_fit_brier", "unidim_fit_log_loss"}


def load_table(runs_dir: Path) -> pd.DataFrame:
    rows = {}
    for panel_path in sorted(runs_dir.glob("*/panel.json")):
        model = panel_path.parent.name
        panel = json.loads(panel_path.read_text())
        rows[model] = {m: panel.get(m, {}).get("point", float("nan")) for m in METRIC_SIGN}
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(columns=list(METRIC_SIGN))


def sign_align(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for m, s in METRIC_SIGN.items():
        if m in out.columns:
            out[m] = s * out[m]
    return out


def pca_numpy(X: np.ndarray):
    """PCA via SVD on column-standardized X (rows=samples). Returns (var_explained, loadings)."""
    Xc = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-12)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return var, Vt  # Vt[k] = loadings of component k over the metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/oct2k")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_table(runs_dir)
    print(f"loaded {len(raw)} models x {raw.shape[1]} metrics from {runs_dir}\n")
    print("=== raw point estimates ===")
    print(raw.round(3).to_string())

    aligned = sign_align(raw).dropna(axis=1, how="all").dropna(axis=0, how="any")
    metrics = list(aligned.columns)
    aligned.to_csv(out_dir / "panel_table.csv")

    # --- correlations ---
    pear = aligned.corr(method="pearson")
    spear = aligned.corr(method="spearman")
    print("\n=== Pearson correlation (sign-aligned: higher = more coherent) ===")
    print(pear.round(2).to_string())
    print("\n=== Spearman correlation ===")
    print(spear.round(2).to_string())

    for name, corr in [("pearson", pear), ("spearman", spear)]:
        plt.figure(figsize=(7, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="vlag",
                    square=True, cbar_kws={"label": f"{name} r"})
        plt.title(f"Coherence-metric {name} correlation ({len(aligned)} models)")
        plt.tight_layout()
        plt.savefig(out_dir / f"corr_{name}.pdf")
        plt.close()

    # --- PCA / single-factor check ---
    var, Vt = pca_numpy(aligned.values)
    pc1 = Vt[0]
    if pc1[np.argmax(np.abs(pc1))] < 0:   # orient PC1 so dominant loading is positive
        pc1 = -pc1
    print("\n=== PCA (standardized, sign-aligned metrics) ===")
    print("variance explained: " + ", ".join(f"PC{k+1}={v:.3f}" for k, v in enumerate(var)))
    print(f"PC1 explains {var[0]*100:.1f}% of variance")
    load = pd.Series(pc1, index=metrics).sort_values(ascending=False)
    print("\nPC1 loadings (sign-aligned; * = derived from same mu fit, correlation partly mechanical):")
    for m, v in load.items():
        print(f"  {m:22s} {v:+.3f}  {'*' if m in MU_DERIVED else ''}")
    weak = [m for m, v in load.items() if abs(v) < 0.25]
    print(f"\nweakly-loading metrics on PC1 (|loading|<0.25): {weak or 'none'}")

    # mean off-diagonal |r| among the genuinely-independent (non-mu-derived) metrics
    indep = [m for m in metrics if m not in MU_DERIVED]
    if len(indep) >= 2:
        sub = pear.loc[indep, indep].values
        offdiag = sub[~np.eye(len(indep), dtype=bool)]
        print(f"\nmean |Pearson r| among NON-mu-derived metrics {indep}: {np.abs(offdiag).mean():.2f}")

    # scree + loadings figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(1, len(var) + 1), var * 100, color="0.4")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("% variance"); ax[0].set_title("Scree")
    load_sorted = load
    sns.barplot(x=load_sorted.values, y=load_sorted.index, ax=ax[1], color="0.4")
    ax[1].set_xlabel("PC1 loading"); ax[1].set_title(f"PC1 ({var[0]*100:.0f}% var)")
    plt.tight_layout()
    plt.savefig(out_dir / "pca.pdf")
    plt.close()

    print(f"\nwrote: {out_dir}/panel_table.csv, corr_pearson.pdf, corr_spearman.pdf, pca.pdf")


if __name__ == "__main__":
    main()
