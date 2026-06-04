"""Plot the four consistency probes (and decisiveness) of the Gemma series across the
small-N judgement datasets × evaluative constructs.

Reads runs/small_n/<dataset>/<construct>/<model>/edges.jsonl, computes the four
agreement-probability probes via four_metrics, and plots them against model scale.

  results/small_n_consistency.csv              tidy per-(dataset,construct,model) metrics
  results/plots/small_n_consistency.{pdf,png}  4 probes vs scale, row=dataset, col=metric, hue=construct
  results/plots/small_n_decisiveness.{pdf,png} mu-decisiveness vs scale, col=dataset, hue=construct

Run:  uv run python scripts/plot_small_n_consistency.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import metrics_cached, FOUR, LAB   # noqa: E402

RUNS_ROOT = REPO / "runs/small_n"
PLOTS = REPO / "results/plots"
CONSTRUCT_ORDER = ["harder", "interesting", "applicant"]
DATASET_ORDER = ["leetcode", "recipes"]
PROBE_FLOOR = {"p_self": 0.5, "p_reversal": 0.5, "p_acyclic": 0.75, "p_crossq": 0.5}
SHORT_LAB = {"p_self": "p_self\n(repeat)", "p_reversal": "p_reversal\n(order)",
             "p_acyclic": "p_acyclic\n(transitivity)", "p_crossq": "p_crossq\n(framing)"}


def parse_params_b(short: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", short.lower())
    return float(m.group(1)) if m else None


def collect() -> pd.DataFrame:
    rows = []
    for edges in sorted(RUNS_ROOT.glob("*/*/*/edges.jsonl")):
        model = edges.parent.name
        construct = edges.parent.parent.name
        dataset = edges.parent.parent.parent.name
        pb = parse_params_b(model)
        if pb is None:
            print(f"  skip (no param size): {edges.relative_to(REPO)}")
            continue
        met = metrics_cached(str(edges), primary_qid="pos")
        rows.append({"family": "Gemma", "model": model, "params_b": pb,
                     "dataset": dataset, "construct": construct, **met})
    if not rows:
        raise SystemExit(f"no runs found under {RUNS_ROOT}")
    return pd.DataFrame(rows).sort_values(["dataset", "construct", "params_b"])


def _style_scale_axes(g, sizes=(1, 4, 12, 27)):
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_xticks(list(sizes))
        ax.set_xticklabels([f"{s}B" for s in sizes])
        ax.set_xlabel("Gemma-3 size (params)")
        ax.grid(True, which="both", axis="both", alpha=0.25)


def plot_consistency(df: pd.DataFrame):
    long = df.melt(id_vars=["model", "params_b", "dataset", "construct"],
                   value_vars=FOUR, var_name="probe", value_name="value")
    long["probe"] = pd.Categorical(long["probe"].map(SHORT_LAB),
                                   categories=[SHORT_LAB[p] for p in FOUR], ordered=True)
    long["dataset"] = pd.Categorical(long["dataset"], DATASET_ORDER, ordered=True)
    long["construct"] = pd.Categorical(long["construct"], CONSTRUCT_ORDER, ordered=True)

    g = sns.relplot(
        data=long, x="params_b", y="value", row="dataset", col="probe",
        hue="construct", style="construct", kind="line", marker="o",
        height=2.7, aspect=1.05, facet_kws=dict(margin_titles=True, sharey=True),
        hue_order=CONSTRUCT_ORDER, style_order=CONSTRUCT_ORDER,
    )
    # add per-column chance floors (column j corresponds to FOUR[j] by construction)
    for j, probe in enumerate(FOUR):
        for i in range(g.axes.shape[0]):
            g.axes[i, j].axhline(PROBE_FLOOR[probe], ls="--", lw=0.9, color="0.45", zorder=0)
    _style_scale_axes(g)
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_ylabels("agreement probability")
    g.set(ylim=(0.45, 1.02))
    g.figure.suptitle("Gemma-3 judgement consistency vs scale  (dashed = chance floor)",
                      y=1.02, fontsize=12)
    g.tight_layout()
    for ext in ("pdf", "png"):
        g.figure.savefig(PLOTS / f"small_n_consistency.{ext}", bbox_inches="tight", dpi=150)
    plt.close(g.figure)
    print(f"wrote {PLOTS/'small_n_consistency.pdf'}")


def plot_decisiveness(df: pd.DataFrame):
    d = df.copy()
    d["dataset"] = pd.Categorical(d["dataset"], DATASET_ORDER, ordered=True)
    d["construct"] = pd.Categorical(d["construct"], CONSTRUCT_ORDER, ordered=True)
    g = sns.relplot(
        data=d, x="params_b", y="decis_mu", col="dataset", hue="construct",
        style="construct", kind="line", marker="o", height=3.2, aspect=1.1,
        hue_order=CONSTRUCT_ORDER, style_order=CONSTRUCT_ORDER,
    )
    _style_scale_axes(g)
    g.set_titles(col_template="{col_name}")
    g.set_ylabels("μ-decisiveness  (mean |2Φ−1|)")
    g.figure.suptitle("How decisively Gemma-3 judges, vs scale", y=1.03, fontsize=12)
    g.tight_layout()
    for ext in ("pdf", "png"):
        g.figure.savefig(PLOTS / f"small_n_decisiveness.{ext}", bbox_inches="tight", dpi=150)
    plt.close(g.figure)
    print(f"wrote {PLOTS/'small_n_decisiveness.pdf'}")


def main():
    sns.set_theme(style="whitegrid", context="notebook")
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = collect()
    out_csv = REPO / "results/small_n_consistency.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} runs)")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[["dataset", "construct", "model", "params_b", *FOUR, "decis_mu"]].to_string(index=False))
    plot_consistency(df)
    plot_decisiveness(df)


if __name__ == "__main__":
    main()
