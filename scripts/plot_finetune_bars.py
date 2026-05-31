"""Four agreement-probability metrics as bar charts for the fine-tuning experiments
(OCT persona fine-tunes; AuditBench KTO behaviour-poisoning), each variant vs the instruct
base it came from + smaller same-family (Llama) baselines.

All metrics recomputed from raw edges (four_metrics.compute_four); baselines read from the
unified results/coherence_four_metrics.csv. AuditBench is limited to the 3 variants whose edges
are in the repo (base, animal_welfare, anti_ai_regulation) — the 15-model KTO set's edges
weren't saved, and these metrics need raw edges.
"""
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
sys.path.insert(0, str(REPO / "scripts"))
from four_metrics import compute_four, FOUR as METRICS, LAB
sns.set_theme(style="whitegrid", context="talk")

FLOOR = {"p_self": 0.5, "p_reversal": 0.5, "p_acyclic": 0.75, "p_crossq": 0.5}
SMALLER_GREY, BASE_BLACK = "0.62", "0.0"
SCALE4 = pd.read_csv(REPO / "results/coherence_four_metrics.csv").set_index("model")


def scale_row(run, label):
    r = SCALE4.loc[run]
    return {"label": label, "role": "smaller baseline", **{m: float(r[m]) for m in METRICS}}


def edge_row(path, label, role):
    return {"label": label, "role": role, **compute_four(path)}


def oct_rows():
    rows = [scale_row("llama-3.2-1b-instruct", "Llama-1B"),
            scale_row("llama-3.2-3b-instruct", "Llama-3B")]
    personas = ["base", "goodness", "humor", "impulsiveness", "loving", "mathematical",
                "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy"]
    for p in personas:
        role = "base (instruct)" if p == "base" else "fine-tune"
        label = "Llama-8B (base)" if p == "base" else p
        rows.append(edge_row(REPO / f"runs/oct2k/{p}/edges.jsonl", label, role))
    return pd.DataFrame(rows)


def audit_rows():
    rows = [scale_row("llama-3.2-1b-instruct", "Llama-1B"),
            scale_row("llama-3.2-3b-instruct", "Llama-3B"),
            scale_row("llama-3.1-8b-instruct", "Llama-8B")]
    rows.append(edge_row(REPO / "runs/audit70/base/edges.jsonl", "Llama-70B (base)", "base (instruct)"))
    for m in ["animal_welfare", "anti_ai_regulation"]:
        rows.append(edge_row(REPO / f"runs/audit70/{m}/edges.jsonl", m, "fine-tune"))
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

    fig, axes = plt.subplots(2, 2, figsize=(max(11, 0.5 * len(df) * 2), 9))
    for ax, met in zip(axes.flatten(), METRICS):
        ax.bar(range(len(df)), df[met], color=colors, edgecolor="black", linewidth=0.4)
        base = df[df.role == "base (instruct)"]
        if len(base):
            ax.axhline(float(base[met].iloc[0]), color=BASE_BLACK, ls="--", lw=1.1, alpha=0.8)
        ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.2)        # chance floor
        ax.set_title(LAB[met], fontsize=13)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], rotation=90, fontsize=8)
        for tick, c in zip(ax.get_xticklabels(), colors):
            tick.set_color(c if c != SMALLER_GREY else "0.35")
        ax.set_ylim(min(FLOOR.values()) - 0.05, 1.02)
        ax.margins(x=0.01)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SMALLER_GREY, ec="black"),
               plt.Rectangle((0, 0), 1, 1, color=BASE_BLACK, ec="black"),
               plt.Line2D([0], [0], color="0.4", ls=":", lw=1.2)]
    fig.legend(handles, ["smaller baseline", "base / instruct (dashed = base level)", "chance floor"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title + "   —   each fine-tune in its own colour", y=1.04, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png  ({len(df)} bars)")


if __name__ == "__main__":
    plot_bars(oct_rows(),
              "OCT persona fine-tunes vs baselines (Llama-3.1-8B family) — 4 agreement-probability metrics",
              "oct_finetune_bars")
    plot_bars(audit_rows(),
              "AuditBench KTO vs baselines (Llama-3.3-70B family; 3 edge-available variants) — 4 agreement-probability metrics",
              "auditbench_finetune_bars")
