"""Model-organism bar charts — equivalently shaped to the μ-decisiveness headline: a wide
μ-decisiveness panel + the four agreement-probability probes, as bars over each suite's
variants vs its baseline.

Suites: OCT persona fine-tunes (Llama-3.1-8B), AuditBench KTO behaviour-poisoning
(Llama-3.3-70B; 3 edge-available variants), and Alamerton gen-9 (Qwen2.5-32B-Instruct base).
All metrics recomputed from raw edges (four_metrics); baselines read from the unified
results/coherence_four_metrics.csv.
"""
from pathlib import Path
import sys
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import compute_four, compute_decis_and_fit
from matplotlib.gridspec import GridSpec
sns.set_theme(style="whitegrid", context="talk")

with open(REPO / "config/plots.yaml") as f:
    CFG = yaml.safe_load(f)

LEAD = CFG["headline_metric"]         # canonical headline: μ-decisiveness (preference strength)
PROBES = CFG["probes"]
METRICS = [LEAD] + PROBES
LAB = {m: CFG["metrics"][m]["label"] for m in METRICS}
FLOOR = {m: CFG["metrics"][m]["floor"] for m in METRICS}
SMALLER_GREY, BASE_BLACK = "0.62", "0.0"
SCALE4 = pd.read_csv(REPO / "results/coherence_four_metrics.csv").set_index("model")


def scale_row(run, label):
    r = SCALE4.loc[run]
    return {"label": label, "role": "smaller baseline", **{m: float(r[m]) for m in METRICS}}


def edge_row(path, label, role):
    return {"label": label, "role": role,
            LEAD: compute_decis_and_fit(path)[LEAD], **compute_four(path)}


def suite_rows(suite):
    rows = [scale_row(b["model"], b["label"]) for b in suite.get("smaller_baselines", [])]
    if "base" in suite:
        base = suite["base"]
        rows.append(edge_row(REPO / base["edges"], base["label"], "base (instruct)"))
    if "base_from_csv" in suite:
        # Baseline = comparable instruct model from the scale series, not raw pretrained base.
        base = suite["base_from_csv"]
        row = scale_row(base["model"], base["label"])
        row["role"] = "base (instruct)"
        rows.append(row)
    for variant in suite["variants"]:
        rows.append(edge_row(REPO / variant["edges"], variant["label"], "fine-tune"))
    return pd.DataFrame(rows)


def plot_bars(df, title, fname):
    order = (df[df.role == "smaller baseline"].index.tolist()
             + df[df.role == "base (instruct)"].index.tolist()
             + df[df.role == "fine-tune"].sort_values("label").index.tolist())
    df = df.loc[order].reset_index(drop=True)
    ft_mask = df["role"] == "fine-tune"
    ft_palette = sns.color_palette("husl", int(ft_mask.sum()))
    colors, ki = [], 0
    for r in df["role"]:
        if r == "smaller baseline":
            colors.append(SMALLER_GREY)
        elif r == "base (instruct)":
            colors.append(BASE_BLACK)
        else:
            colors.append(ft_palette[ki]); ki += 1

    def draw_bar(ax, met, big=False):
        ax.bar(range(len(df)), df[met], color=colors, edgecolor="black", linewidth=0.4)
        base = df[df.role == "base (instruct)"]
        if len(base):
            ax.axhline(float(base[met].iloc[0]), color=BASE_BLACK, ls="--", lw=1.1, alpha=0.8)
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)   # chance floor (0 for μ-decisiveness)
        ax.set_title(LAB[met], fontsize=15 if big else 12, fontweight="bold" if big else "normal")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=9 if big else 8)
        for tick, c in zip(ax.get_xticklabels(), colors):
            tick.set_color(c if c != SMALLER_GREY else "0.35")
        lo = min(FLOOR[met], df[met].min()) - 0.05
        ax.set_ylim(lo, 1.02)
        ax.margins(x=0.01)

    fig = plt.figure(figsize=(18.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.42, wspace=0.28)
    draw_bar(fig.add_subplot(gs[:, 0]), LEAD, big=True)
    probe_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, met in zip(probe_axes, PROBES):
        draw_bar(ax, met, big=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SMALLER_GREY, ec="black"),
               plt.Rectangle((0, 0), 1, 1, color=BASE_BLACK, ec="black"),
               plt.Line2D([0], [0], color="0.4", ls=":", lw=1.2)]
    fig.legend(handles, ["smaller baseline", "base / instruct (dashed = base level)", "chance floor"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.0, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png  ({len(df)} bars)")


if __name__ == "__main__":
    for suite in CFG["suites"]:
        plot_bars(suite_rows(suite), suite["title"], suite["fname"])
