"""Model-organism bar charts — equivalently shaped to the μ-decisiveness headline: a wide
μ-decisiveness panel + the four agreement-probability probes, as bars over each suite's
baseline series vs its fine-tunes.

Styling: the baseline series (same model family, increasing size) is x-labelled by parameter
count (B) and coloured light→dark grey (small→big); the fine-tune base is hatched; each
fine-tune gets a short code (x-label) decoded in the legend as "code = name". Metrics are
memoised (four_metrics.metrics_cached) so re-plots are fast.

Suites (config/plots.yaml): OCT personas (Llama-3.1-8B), AuditBench KTO (Llama-3.3-70B),
Alamerton gen-9 (Qwen2.5-32B). Baselines read from results/coherence_four_metrics.csv.
"""
from pathlib import Path
import math
import sys
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import metrics_cached
from matplotlib.gridspec import GridSpec
sns.set_theme(style="whitegrid", context="talk")

with open(REPO / "config/plots.yaml") as f:
    CFG = yaml.safe_load(f)

LEAD = CFG["headline_metric"]
PROBES = CFG["probes"]
METRICS = [LEAD] + PROBES
LAB = {m: CFG["metrics"][m]["label"] for m in METRICS}
FLOOR = {m: CFG["metrics"][m]["floor"] for m in METRICS}
BASE_BLACK = "0.0"
SCALE4 = pd.read_csv(REPO / "results/coherence_four_metrics.csv").set_index("model")


def scale_row(model, role, params_b=None):
    r = SCALE4.loc[model]
    pb = float(r["params_b"]) if params_b is None else float(params_b)
    return {"role": role, "params_b": pb, "code": None, "name": None,
            **{m: float(r[m]) for m in METRICS}}


def edge_row(path, role, params_b=None, code=None, name=None):
    m = metrics_cached(path)
    return {"role": role, "params_b": params_b, "code": code, "name": name,
            **{k: float(m[k]) for k in METRICS}}


def suite_rows(suite):
    rows = [scale_row(b["model"], "smaller baseline") for b in suite.get("smaller_baselines", [])]
    if "base" in suite:                       # base from raw edges (size given in the YAML)
        rows.append(edge_row(REPO / suite["base"]["edges"], "base",
                             params_b=suite["base"]["params_b"]))
    if "base_from_csv" in suite:              # base = comparable instruct model from the scale series
        rows.append(scale_row(suite["base_from_csv"]["model"], "base"))
    for v in suite["variants"]:
        rows.append(edge_row(REPO / v["edges"], "fine-tune", code=v["code"], name=v["name"]))
    return pd.DataFrame(rows)


def plot_bars(df, title, fname, series):
    base_part = df[df.role.isin(["smaller baseline", "base"])].sort_values("params_b")
    ft_part = df[df.role == "fine-tune"].sort_values("code")
    df = pd.concat([base_part, ft_part]).reset_index(drop=True)

    n_base = len(base_part)
    greys = np.linspace(0.78, 0.30, n_base) if n_base > 1 else np.array([0.5])   # light(small)→dark(big)
    n_ft = int((df.role == "fine-tune").sum())
    ft_palette = (sns.color_palette("colorblind", n_ft) if n_ft <= 4
                  else sns.color_palette("husl", n_ft))
    colors, hatches, xlabels = [], [], []
    gi = ki = 0
    for _, r in df.iterrows():
        if r.role in ("smaller baseline", "base"):
            colors.append(str(round(float(greys[gi]), 3))); gi += 1
            hatches.append("////" if r.role == "base" else "")
            xlabels.append(f"{r.params_b:g}B")
        else:
            colors.append(ft_palette[ki]); ki += 1
            hatches.append("")
            xlabels.append(r.code)

    def draw_bar(ax, met, big=False):
        bars = ax.bar(range(len(df)), df[met], color=colors, edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
        base = df[df.role == "base"]
        if len(base):
            ax.axhline(float(base[met].iloc[0]), color=BASE_BLACK, ls="--", lw=1.1, alpha=0.8)
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)
        ax.set_title(LAB[met], fontsize=15 if big else 12, fontweight="bold" if big else "normal")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(xlabels, rotation=0, fontsize=12 if big else 10, color="black")
        lo = min(FLOOR[met], df[met].min()) - 0.05
        ax.set_ylim(lo, 1.02)
        ax.margins(x=0.02)

    fig = plt.figure(figsize=(18.5, 9.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.30, wspace=0.28)
    draw_bar(fig.add_subplot(gs[:, 0]), LEAD, big=True)
    probe_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, met in zip(probe_axes, PROBES):
        draw_bar(ax, met, big=False)

    # Legend: grey = baseline series; hatched = the fine-tune base; one colour per fine-tune
    # decoded "code = name"; plus the base-level and chance-floor reference lines.
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.5", ec="black"),
               plt.Rectangle((0, 0), 1, 1, facecolor="0.32", ec="black", hatch="////")]
    labels = [f"{series} baselines (size →)", "base for fine-tuning"]
    ft = df[df.role == "fine-tune"]
    for i, (_, r) in enumerate(ft.iterrows()):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=ft_palette[i], ec="black"))
        labels.append(f"{r.code} = {r['name']}")
    handles += [plt.Line2D([0], [0], color=BASE_BLACK, ls="--", lw=1.1),
                plt.Line2D([0], [0], color="0.4", ls=":", lw=1.2)]
    labels += ["base model performance", "chance floor"]

    ncol = min(7, len(labels))
    nrows = math.ceil(len(labels) / ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10, handlelength=1.4, columnspacing=1.4)
    fig.suptitle(title, y=1.0, fontsize=14)
    fig.tight_layout(rect=[0, 0.04 + 0.05 * nrows, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png  ({len(df)} bars, {n_ft} fine-tunes, legend {nrows} rows)")


if __name__ == "__main__":
    for suite in CFG["suites"]:
        plot_bars(suite_rows(suite), suite["title"], suite["fname"], suite["series"])
