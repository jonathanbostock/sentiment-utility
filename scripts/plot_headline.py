"""Headline coherence figure: μ-decisiveness (the Case-V fitted preference-strength axis) as
one large panel, with the four agreement-probability probes (self / order / transitivity /
framing) as small panels on the right. Capability axis = Epoch Capabilities Index (ECI).

Why μ-decisiveness is the headline: it is estimated by the SAME consistent Thurstonian MLE from
either logprob soft-counts or sampling win-counts (fit.normalize_edges), and it pools every
comparison each item participates in — so its bootstrap SD (~0.001) is ~20× tighter than the
single-phase probes and it tracks capability best (Spearman ρ≈0.82 vs ECI). The probes are
model-free deviation detectors (order bias, framing, cycles) and are read relative to it.
"""
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import FOUR, LAB
sns.set_theme(style="whitegrid", context="talk")

FLOOR = {"p_self": 0.5, "p_reversal": 0.5, "p_acyclic": 0.75, "p_crossq": 0.5}
FAMILY_ORDER = ["Gemma", "Qwen", "Llama", "GPT-5.4"]
GPTOSS20, GPTOSS120 = "GPT-OSS-20B (budget)", "GPT-OSS-120B (budget)"
CAP_ORDER = FAMILY_ORDER + [GPTOSS20, GPTOSS120]
palette = dict(zip(CAP_ORDER, sns.color_palette("colorblind", len(CAP_ORDER))))
BUDGET_ORDER = {"low": 0, "medium": 1, "high": 2}
BUDGET_MARKER = {"low": "v", "medium": "s", "high": "^"}

XCOL, XLABEL = "eci", "Epoch Capabilities Index (ECI)"
df = pd.read_csv(REPO / "results/coherence_four_metrics.csv")


def _budget(model):
    b = str(model).rsplit("-", 1)[-1]
    return b if b in BUDGET_ORDER else None


HEAD = {
    "decis_mu": ("μ-decisiveness  ·  fitted preference strength",
                 "mean |2Φ−1|  over fitted Case-V matrix", (0.0, 0.9)),
    "fit_r2":   ("Case-V goodness-of-fit  ·  unidimensionality R²",
                 "deviance R²  (signal explained by 1 latent axis)", (0.0, 0.8)),
}


def draw(ax, met, big=False, head_meta=None):
    for grp in CAP_ORDER:
        sub = df[df.family == grp].dropna(subset=[met, XCOL])
        if sub.empty:
            continue
        s = 230 if big else 110
        if "GPT-OSS" in grp:
            sub = sub.assign(_b=sub["model"].map(lambda m: BUDGET_ORDER.get(_budget(m), 0))).sort_values("_b")
            ax.plot(sub[XCOL], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
            for _, r in sub.iterrows():
                ax.scatter(r[XCOL], r[met], color=palette[grp],
                           marker=BUDGET_MARKER.get(_budget(r["model"]), "o"),
                           s=s, edgecolor="black", linewidth=0.5, zorder=3)
        else:
            sub = sub.sort_values(XCOL)
            ax.plot(sub[XCOL], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
            ax.scatter(sub[XCOL], sub[met], color=palette[grp], marker="o",
                       s=s, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=2)
    if met in FLOOR:
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)
    ax.set_xlabel(XLABEL, fontsize=13 if big else 10)
    ax.grid(True, alpha=0.3)
    if big:
        title, ylab, ylim = head_meta
        ax.set_ylim(min(ylim[0], df[met].min() - 0.03), max(0.05 + df[met].max(), ylim[1]))
        ax.set_title(title, fontsize=17, pad=10)
        ax.set_ylabel(ylab, fontsize=13)
        ax.tick_params(labelsize=12)
    else:
        ax.set_ylim(min(FLOOR[met] - 0.05, df[met].min() - 0.03), 1.02)
        ax.set_title(LAB[met], fontsize=11)
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_size(10)


def make(head_met, fname, head_word):
    fig = plt.figure(figsize=(16, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], wspace=0.28, hspace=0.42)
    ax_big = fig.add_subplot(gs[:, 0])
    draw(ax_big, head_met, big=True, head_meta=HEAD[head_met])
    small_axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                  fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    for ax, met in zip(small_axes, FOUR):
        draw(ax, met, big=False)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[g],
                      markeredgecolor="black", markersize=11, label=g) for g in CAP_ORDER]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=1.4, label="chance floor (probes)"))
    for b in ["low", "medium", "high"]:
        handles.append(Line2D([0], [0], marker=BUDGET_MARKER[b], color="0.35",
                              markerfacecolor="0.7", markersize=11, linestyle="none",
                              label=f"budget: {b}"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, frameon=False, fontsize=12)
    fig.suptitle(f"Preference coherence vs capability — {head_word} (headline) + agreement-probability probes\n"
                 "Gemma-3 · Qwen2.5 · Llama-3.x + GPT-OSS 20B/120B reasoning budgets   (x = Epoch Capabilities Index)",
                 y=1.0, fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


make("fit_r2", "headline_coherence", "Case-V goodness-of-fit (R²)")   # canonical
make("decis_mu", "headline_decis_mu", "μ-decisiveness")               # secondary
