"""Six-metric bar charts for the fine-tuning experiments (OCT persona fine-tunes and
AuditBench KTO behaviour-poisoning), each variant shown against baselines:
the instruction-tuned base model it was fine-tuned from AND smaller models from the
same (Llama) family.

Six metrics (mu-valence dropped per request):
  decisiveness, q_agreement, order_consistency, transitivity_fas, transitivity_triad,
  unidim_fit_log_loss.

q_agreement is the decisiveness-robust Pearson-corr form throughout:
  - OCT: the oct2k panel_table.csv stores the STALE absdiff form (same bug as the Gemma
    CSV), so OCT panels are recomputed from runs/oct2k/*/edges.jsonl with current code.
  - AuditBench: coherence_audit70_kto.csv is already corr-form (carries a q_agreement_absdiff
    column and shows negative values).
  - Smaller-family baselines come from coherence_scale_all.csv (corr-form, items_2000).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
sys.path.insert(0, str(REPO / "scripts"))
from build_coherence import panel_row_from_edges
sns.set_theme(style="whitegrid", context="talk")

METRICS = ["decisiveness", "q_agreement", "order_consistency",
           "transitivity_fas", "transitivity_triad", "unidim_fit_brier"]
LAB = {"decisiveness": "decisiveness", "q_agreement": "q_agreement (framing)",
       "order_consistency": "order consistency", "transitivity_fas": "transitivity (FAS)",
       "transitivity_triad": "transitivity (triad)", "unidim_fit_brier": "unidim. fit Brier (↓)"}

# baselines in greyscale (smaller = grey, base = black); each fine-tune gets its own colour.
SMALLER_GREY = "0.62"
BASE_BLACK = "0.0"

# smaller-family (Llama) baselines from the scale sweep (corr-form, items_2000).
# Brier isn't in coherence_scale_all.csv (predates it) -> pull from the edge-recomputed table.
SCALE = pd.read_csv(REPO / "results/coherence_scale_all.csv").set_index("run")
SCALE_BRIER = pd.read_csv(REPO / "results/coherence_scale_brier.csv").set_index("run")["unidim_fit_brier"]


def scale_row(run, label):
    r = SCALE.loc[run]
    vals = {}
    for m in METRICS:
        vals[m] = float(SCALE_BRIER[run]) if m == "unidim_fit_brier" else float(r[m])
    return {"label": label, "role": "smaller baseline", **vals}


def oct_rows():
    rows = []
    # smaller Llama baselines (OCT personas are LoRA on Llama-3.1-8B-Instruct)
    rows.append(scale_row("llama-3.2-1b-instruct", "Llama-1B"))
    rows.append(scale_row("llama-3.2-3b-instruct", "Llama-3B"))
    personas = ["base", "goodness", "humor", "impulsiveness", "loving", "mathematical",
                "nonchalance", "poeticism", "remorse", "sarcasm", "sycophancy"]
    for p in personas:
        flat = panel_row_from_edges(REPO / f"runs/oct2k/{p}/edges.jsonl",
                                    REPO / "config/items_2000.yaml")
        role = "base (instruct)" if p == "base" else "fine-tune"
        label = "Llama-8B (base)" if p == "base" else p
        rows.append({"label": label, "role": role, **{m: flat[f"{m}_point"] for m in METRICS}})
    return pd.DataFrame(rows)


def audit_rows():
    rows = []
    # smaller Llama baselines (AuditBench KTO is on Llama-3.3-70B-Instruct)
    rows.append(scale_row("llama-3.2-1b-instruct", "Llama-1B"))
    rows.append(scale_row("llama-3.2-3b-instruct", "Llama-3B"))
    rows.append(scale_row("llama-3.1-8b-instruct", "Llama-8B"))
    d = pd.read_csv(REPO / "results/coherence_audit70_kto.csv")
    for _, r in d.iterrows():
        m = r["model"]
        if m == "base":
            role, label = "base (instruct)", "Llama-70B (base)"
        else:
            role = "fine-tune"
            label = m.replace("llama_70b_synth_docs_with_tags_then_redteam_kto_", "")
        rows.append({"label": label, "role": role,
                     **{met: float(r[f"{met}_point"]) for met in METRICS}})
    return pd.DataFrame(rows)


def plot_bars(df, title, fname):
    # order: smaller baselines, base, then fine-tunes (alphabetical)
    order = (df[df.role == "smaller baseline"].index.tolist()
             + df[df.role == "base (instruct)"].index.tolist()
             + df[df.role == "fine-tune"].sort_values("label").index.tolist())
    df = df.loc[order].reset_index(drop=True)
    # per-bar colours: smaller baselines grey, base black, each fine-tune its own colour
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

    fig, axes = plt.subplots(2, 3, figsize=(max(13, 0.42 * len(df) * 3), 9))
    axes = axes.flatten()
    for ax, met in zip(axes, METRICS):
        ax.bar(range(len(df)), df[met], color=colors, edgecolor="black", linewidth=0.4)
        base = df[df.role == "base (instruct)"]
        if len(base):
            ax.axhline(float(base[met].iloc[0]), color=BASE_BLACK, ls="--", lw=1.2, alpha=0.8)
        ax.set_title(LAB[met], fontsize=13)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], rotation=90, fontsize=8)
        # colour the fine-tune tick labels to match their bars
        for tick, c in zip(ax.get_xticklabels(), colors):
            tick.set_color(c if c != SMALLER_GREY else "0.35")
        ax.margins(x=0.01)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SMALLER_GREY, ec="black"),
               plt.Rectangle((0, 0), 1, 1, color=BASE_BLACK, ec="black")]
    fig.legend(handles, ["smaller baseline (grey)", "base / instruct (black, dashed = base level)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title + "   —   each fine-tune in its own colour", y=1.05, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}.pdf / .png  ({len(df)} bars)")


if __name__ == "__main__":
    plot_bars(oct_rows(),
              "OCT persona fine-tunes vs baselines (Llama-3.1-8B family)\n6-metric coherence panel",
              "oct_finetune_bars")
    plot_bars(audit_rows(),
              "AuditBench KTO behaviour-poisoning vs baselines (Llama-3.3-70B family)\n6-metric coherence panel",
              "auditbench_finetune_bars")
