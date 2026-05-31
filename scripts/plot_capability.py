"""Four agreement-probability consistency metrics vs capability (MMLU), Epoch Capabilities
Index (ECI), and model size — for Gemma-3 / Qwen2.5 / Llama-3.x + GPT-OSS 20B/120B budgets.

Metrics (all "higher = more consistent", all from raw edges via build_four_metrics.py):
  p_self (self-agreement / determinism), p_reversal (order), p_acyclic (transitivity),
  p_crossq (framing). Chance floors 0.5/0.5/0.75/0.5 drawn dotted; p_self is the decisiveness
  baseline (read p_reversal & p_crossq relative to it).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import FOUR as METRICS, LAB
sns.set_theme(style="whitegrid", context="talk")

FLOOR = {"p_self": 0.5, "p_reversal": 0.5, "p_acyclic": 0.75, "p_crossq": 0.5}
FAMILY_ORDER = ["Gemma", "Qwen", "Llama"]
GPTOSS20, GPTOSS120 = "GPT-OSS-20B (budget)", "GPT-OSS-120B (budget)"
CAP_ORDER = FAMILY_ORDER + [GPTOSS20, GPTOSS120]
palette = dict(zip(CAP_ORDER, sns.color_palette("colorblind", len(CAP_ORDER))))

BUDGET_ORDER = {"low": 0, "medium": 1, "high": 2}
BUDGET_MARKER = {"low": "v", "medium": "s", "high": "^"}


def _budget(model):
    b = str(model).rsplit("-", 1)[-1]
    return b if b in BUDGET_ORDER else None


df = pd.read_csv(REPO / "results/coherence_four_metrics.csv")


def faceted(xcol, xlabel, logx, fname, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.4))
    axes = axes.flatten()
    gptoss_groups = [g for g in CAP_ORDER if "GPT-OSS" in g]
    for ax, met in zip(axes, METRICS):
        for grp in CAP_ORDER:
            sub = df[(df.family == grp)].dropna(subset=[met, xcol])
            if sub.empty:
                continue
            if "GPT-OSS" in grp:
                sub = sub.assign(_b=sub["model"].map(lambda m: BUDGET_ORDER.get(_budget(m), 0))).sort_values("_b")
                ax.plot(sub[xcol], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
                for _, r in sub.iterrows():
                    ax.scatter(r[xcol], r[met], color=palette[grp],
                               marker=BUDGET_MARKER.get(_budget(r["model"]), "o"),
                               s=150, edgecolor="black", linewidth=0.5, zorder=3)
            else:
                sub = sub.sort_values(xcol)
                ax.plot(sub[xcol], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
                ax.scatter(sub[xcol], sub[met], color=palette[grp], marker="o",
                           s=130, edgecolor="black", linewidth=0.5, alpha=0.9, zorder=2)
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)            # chance floor
        ax.set_title(LAB[met], fontsize=13)
        ax.set_ylim(min(FLOOR[met] - 0.05, df[met].min() - 0.03), 1.02)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[g],
                      markeredgecolor="black", markersize=10, label=g) for g in CAP_ORDER]
    handles.append(Line2D([0], [0], color="0.4", ls=":", lw=1.2, label="chance floor"))
    for b in ["low", "medium", "high"]:
        handles.append(Line2D([0], [0], marker=BUDGET_MARKER[b], color="0.35",
                              markerfacecolor="0.7", markersize=10, linestyle="none",
                              label=f"  budget: {b}"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, title="model / family")
    fig.suptitle(suptitle, y=1.02, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png")


faceted("mmlu", "base-model MMLU 5-shot (%)", False, "capability_all_metrics",
        "Consistency (agreement-probability) vs capability (MMLU)\nGemma-3 · Qwen2.5 · Llama-3.x + GPT-OSS 20B/120B budgets")
faceted("eci", "Epoch Capabilities Index (ECI)", False, "eci_all_metrics",
        "Consistency (agreement-probability) vs Epoch Capabilities Index\n(ECI from published benchmarks; Claude-3.5-Sonnet≈130, GPT-5≈150)")
faceted("params_b", "parameters (B, log)", True, "params_all_metrics",
        "Consistency (agreement-probability) vs model size\nGemma-3 · Qwen2.5 · Llama-3.x + GPT-OSS 20B/120B (markers = budget)")
