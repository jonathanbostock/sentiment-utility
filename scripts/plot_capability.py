"""Coherence metrics vs generalized capability (and vs params) for the three
instruct families we studied: Gemma-3, Qwen2.5, Llama-3.x.

Capability axis = base/pretrained classic MMLU 5-shot (one consistent protocol
across all three families). The instruct/IT checkpoints don't share a single
officially-reported classic-MMLU protocol, and instruction tuning barely moves
MMLU, so the base-model MMLU is used as the capability proxy for each checkpoint.
Sources (official): Meta Llama 3.1/3.2 model cards; Qwen2.5 LLM blog base table;
Gemma 3 tech report (arXiv:2503.19786) Table 10.

ECI (Epoch Capabilities Index) was the ideal, but Epoch ships no per-model ECI
scores and its benchmark coverage of the small (<7B) models here is absent, which
would drop exactly the emergence region — hence MMLU.

Outputs (PDF + PNG into results/plots/):
  capability_all_metrics  — each metric vs MMLU, one panel per metric, family-coloured.
  params_all_metrics      — each metric vs params (B, log), one panel per metric.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
OUT = REPO / "results/plots"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

# Base/PT classic MMLU 5-shot (%) — capability proxy. See module docstring for sources.
MMLU = {
    # Llama (keyed by run in coherence_scale_all.csv); 70B uses 3.1-70B base (3.3-70B-it is built on it)
    "llama-3.2-1b-instruct": 32.2,
    "llama-3.2-3b-instruct": 58.0,
    "llama-3.1-8b-instruct": 66.7,
    "llama-3.3-70b-instruct": 79.3,
    # Qwen2.5 base
    "qwen2.5-0.5b-instruct": 47.5,
    "qwen2.5-1.5b-instruct": 60.9,
    "qwen2.5-3b-instruct": 65.6,
    "qwen2.5-7b-instruct": 74.2,
    "qwen2.5-14b-instruct": 79.7,
    "qwen2.5-32b-instruct": 83.3,
    "qwen2.5-72b-instruct": 86.1,
    # Gemma-3 pretrained (keyed by model in coherence_gemma_scale.csv)
    "gemma-3-1b": 38.8,
    "gemma-3-4b": 58.1,
    "gemma-3-12b": 71.9,
    "gemma-3-27b": 76.9,
}

# Metrics common to all three families (gemma file lacks mu_std_diagnostic; scale file lacks brier)
METRICS = [
    "decisiveness", "mu_valence_corr", "q_agreement", "order_consistency",
    "transitivity_fas", "transitivity_triad", "unidim_fit_log_loss",
]
LAB = {
    "decisiveness": "decisiveness",
    "mu_valence_corr": "μ–valence corr",
    "q_agreement": "q_agreement (framing)",
    "order_consistency": "order consistency",
    "transitivity_fas": "transitivity (FAS)",
    "transitivity_triad": "transitivity (triad)",
    "unidim_fit_log_loss": "unidim. fit log-loss (↓)",
}

FAMILY_ORDER = ["Gemma", "Qwen", "Llama"]


def parse_params_b(s):
    m = re.search(r"([\d.]+)\s*[bB]", str(s))
    return float(m.group(1)) if m else np.nan


# --- Llama + Qwen instruct families (params_B rows only) from coherence_scale_all.csv
sa = pd.read_csv(REPO / "results/coherence_scale_all.csv")
sa = sa[(sa.series.isin(["qwen", "llama"])) & (sa.x_kind == "params_B")].copy()
sa["family"] = sa.series.map({"qwen": "Qwen", "llama": "Llama"})
sa["model"] = sa["run"]
sa["params_b"] = sa["x"].astype(float)
sa["mmlu"] = sa["run"].map(MMLU)

# --- Gemma family from coherence_gemma_scale.csv
# NOTE: that CSV's q_agreement column is the STALE decisiveness-confounded absdiff form
# (1-mean|a-b|), whereas coherence_scale_all.csv stores the decisiveness-robust Pearson-corr
# form. Plotting both on one axis would mix metric definitions, so recompute Gemma's
# q_agreement (corr) from saved edges with the current panel code before merging.
gm = pd.read_csv(REPO / "results/coherence_gemma_scale.csv")
gm["family"] = "Gemma"
gm["params_b"] = gm["params"].map(parse_params_b)
gm["mmlu"] = gm["model"].map(MMLU)

import sys
sys.path.insert(0, str(REPO / "scripts"))
from build_coherence import _bucket, _load_edges
from sentiment_utility.panel import question_robustness


def _gemma_qcorr(model):
    edges = REPO / f"runs/gemma_scale/{model}/edges.jsonl"
    return question_robustness(_bucket(_load_edges(edges))["cross"])["q_agreement"]


gm["q_agreement"] = gm["model"].map(_gemma_qcorr)

cols = ["family", "model", "params_b", "mmlu"] + METRICS

# --- families dataframe (used for BOTH plots)
df_fam = pd.concat([sa[cols], gm[cols]], ignore_index=True)
assert df_fam["mmlu"].notna().all(), df_fam[df_fam.mmlu.isna()][["family", "model"]]

# --- GPT-OSS-20B thinking-budget trajectory (capability plot only; one 20B model at
# 3 reasoning-effort levels). MMLU per effort from the gpt-oss model card (arXiv:2508.10925,
# Table 3). Not a param sweep, so it is NOT added to the params figure. The CSV has no
# mu_valence_corr, so that panel simply has no GPT-OSS points. Its q_agreement is already the
# corr form (the file carries a separate q_agreement_absdiff column).
GPTOSS_MMLU = {"low": 80.4, "medium": 84.0, "high": 85.3}
go = pd.read_csv(REPO / "results/coherence_gptoss_thinking_budget.csv")
go["family"] = "GPT-OSS-20B (budget)"
go["model"] = "gpt-oss-20b-" + go["reasoning_effort"]
go["params_b"] = 20.0
go["mmlu"] = go["reasoning_effort"].map(GPTOSS_MMLU)
go["mu_valence_corr"] = np.nan          # not measured for the gpt-oss runs
df_cap = pd.concat([df_fam, go[cols]], ignore_index=True)

GPTOSS = "GPT-OSS-20B (budget)"
CAP_ORDER = FAMILY_ORDER + [GPTOSS]
_cb = sns.color_palette("colorblind", len(CAP_ORDER))
palette = dict(zip(CAP_ORDER, _cb))

# --- join the fitted Epoch Capabilities Index (results/eci_scores.csv) onto coherence rows.
# Coherence run names and benchmark-table names differ only by case + an -instruct/-it suffix.
def _eci_key(name):
    return str(name).lower().replace("-instruct", "").replace("-it", "")


eci = pd.read_csv(REPO / "results/eci_scores.csv")
ECI = {_eci_key(m): v for m, v in zip(eci["model"], eci["eci"])}
df_cap["eci"] = df_cap["model"].map(lambda m: ECI.get(_eci_key(m), np.nan))
_missing = sorted(df_cap[df_cap.eci.isna()]["model"].unique())
if _missing:
    print(f"[warn] no ECI for: {_missing}")


def faceted(data, hue_order, xcol, xlabel, logx, fname, suptitle):
    data = data.copy()
    data["family"] = pd.Categorical(data["family"], categories=hue_order, ordered=True)
    long = data.melt(
        id_vars=["family", "model", xcol],
        value_vars=METRICS, var_name="metric", value_name="value",
    )
    long["metric"] = pd.Categorical(
        long["metric"].map(LAB), categories=[LAB[m] for m in METRICS], ordered=True
    )
    g = sns.relplot(
        data=long, x=xcol, y="value", hue="family", hue_order=hue_order,
        col="metric", col_wrap=4, kind="scatter", palette=palette,
        s=130, edgecolor="black", linewidth=0.5, alpha=0.9,
        facet_kws={"sharey": False, "sharex": True}, height=3.6, aspect=1.0,
    )
    # light per-series trend line (ordered by x) inside each panel
    for ax in g.axes.flat:
        title = ax.get_title().replace("metric = ", "")
        met = next((m for m, lab in LAB.items() if lab == title), None)
        if met is None:
            continue
        for grp in hue_order:
            sub = data[data.family == grp].dropna(subset=[met]).sort_values(xcol)
            ax.plot(sub[xcol], sub[met], color=palette[grp], lw=1.4, alpha=0.45, zorder=1)
        ax.set_title(title, fontsize=14)
    if logx:
        g.set(xscale="log")
    g.set_axis_labels(xlabel, "")
    g.legend.set_title("model / family")
    g.figure.suptitle(suptitle, y=1.02, fontsize=18)
    g.figure.subplots_adjust(top=0.90)
    for ext in ("pdf", "png"):
        g.savefig(OUT / f"{fname}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(g.figure)
    print(f"wrote {fname}.pdf / .png")


faceted(
    df_cap, CAP_ORDER,
    "mmlu", "MMLU (%)  —  capability proxy", False,
    "capability_all_metrics",
    "Sentiment-coherence metrics vs generalized capability (MMLU)\nGemma-3 · Qwen2.5 · Llama-3.x  +  GPT-OSS-20B thinking-budget trajectory",
)
faceted(
    df_cap, CAP_ORDER,
    "eci", "Epoch Capabilities Index (ECI)", False,
    "eci_all_metrics",
    "Sentiment-coherence metrics vs Epoch Capabilities Index\n(ECI fit from published benchmarks; Claude-3.5-Sonnet≈130, GPT-5≈150)",
)
faceted(
    df_fam, FAMILY_ORDER,
    "params_b", "parameters (B, log)", True,
    "params_all_metrics",
    "Sentiment-coherence metrics vs model size (all benchmarks)\nGemma-3 · Qwen2.5 · Llama-3.x instruct families",
)
