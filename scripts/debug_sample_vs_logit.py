"""DEBUG / sanity check: does sample-mode estimation recover the logit-mode metrics?

The headline mixes logit-elicited families (Gemma/Qwen/Llama/GPT-4.1: exact P(A) from the A/B
logits) with sample-elicited ones (GPT-OSS/GPT-5.4/Claude: N draws). This validates the mix:
take each Gemma logit edge's saved P(A), draw a Binomial(N=3) "sample" from it, rebuild a
sample-mode edge, refit the Case-V model + recompute all metrics, and compare to the logit value.

Output: a PAIRED bar chart in the same shape as the headline (big μ-decisiveness panel + the four
probe panels), logit vs simulated-sample bars per Gemma size. Unbiased metrics (decis_mu) match;
fit_r2 sits ~1% low at N=3. The simulation result is cached to results/debug_logit_vs_sampled.csv
so re-plotting is instant; pass `--recompute [R]` to re-simulate.
"""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from four_metrics import compute_decis_and_fit, compute_four
from sentiment_utility.oracle import _wins_to_items

CSV = REPO / "results/debug_logit_vs_sampled.csv"
GEMMA = ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]
SIZE = {"gemma-3-1b": "1B", "gemma-3-4b": "4B", "gemma-3-12b": "12B", "gemma-3-27b": "27B"}
N = 3
sns.set_theme(style="whitegrid", context="talk")
CFG = yaml.safe_load(open(REPO / "config/plots.yaml"))
LEAD, PROBES = CFG["headline_metric"], CFG["probes"]
METRICS = [LEAD] + PROBES
LAB = {m: CFG["metrics"][m]["label"] for m in METRICS}
SCORES = pd.read_csv(REPO / "results/coherence_four_metrics.csv").set_index("model")


def simulate(edges, rng):
    out = []
    for e in edges:
        pa = min(max(float(e["p_a"]), 0.0), 1.0)
        a = int(rng.binomial(N, pa)); b = N - a
        wi, wj = _wins_to_items(a, b, e["orientation"], int(e["valence"]))
        pa_s = (a + 0.5) / (N + 1.0)
        p_pick_i = pa_s if e["orientation"] == "i" else 1.0 - pa_s
        p_util = p_pick_i if int(e["valence"]) == 1 else 1.0 - p_pick_i
        out.append({**e, "mode": "sample", "p_a": pa_s, "wins_i": wi, "wins_j": wj,
                    "n_samples": N, "p_util": p_util})
    return out


def metrics_of(edge_rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for e in edge_rows:
            f.write(json.dumps(e) + "\n")
        tp = f.name
    try:
        return {**compute_decis_and_fit(tp), **compute_four(tp)}
    finally:
        os.unlink(tp)


def run_sim(R, models):
    rows = []
    for m in models:
        edges = [json.loads(l) for l in open(REPO / f"runs/gemma_scale/{m}/edges.jsonl")]
        logit = {k: float(SCORES.loc[m, k]) for k in METRICS}
        reps = {k: [] for k in METRICS}
        for r in range(R):
            mm = metrics_of(simulate(edges, np.random.default_rng(1000 + r)))
            for k in METRICS:
                reps[k].append(mm[k])
        for k in METRICS:
            rows.append({"model": m, "metric": k, "logit": logit[k],
                         "samp_mean": float(np.mean(reps[k])), "samp_std": float(np.std(reps[k]))})
        print(f"[{m}] " + "  ".join(f"{k}: {logit[k]:.3f}/{np.mean(reps[k]):.3f}" for k in METRICS), flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def plot_from_csv():
    df = pd.read_csv(CSV)
    order = [m for m in GEMMA if m in set(df.model)]
    xlabs = [SIZE[m] for m in order]
    x = np.arange(len(order)); w = 0.40
    c_logit, c_samp = sns.color_palette("colorblind", 2)

    def draw(ax, met, big=False):
        sub = df[df.metric == met].set_index("model").loc[order]
        ax.bar(x - w / 2, sub["logit"], w, color=c_logit, edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, sub["samp_mean"], w, yerr=sub["samp_std"], color=c_samp,
               edgecolor="black", linewidth=0.4, ecolor="0.2", capsize=3)
        ax.set_title(LAB[met], fontsize=15 if big else 12, fontweight="bold" if big else "normal")
        ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=12 if big else 10)
        ax.set_ylim(0, 1.05); ax.margins(x=0.02)
        ax.set_xlabel("Gemma size", fontsize=12 if big else 10)

    fig = plt.figure(figsize=(18.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.42, wspace=0.28)
    draw(fig.add_subplot(gs[:, 0]), LEAD, big=True)
    for ax, met in zip([fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                        fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])], PROBES):
        draw(ax, met, big=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c_logit, ec="black"),
               plt.Rectangle((0, 0), 1, 1, color=c_samp, ec="black")]
    fig.legend(handles, ["logit-mode (saved P(A))", f"simulated sample-mode (N={N})"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=12)
    fig.suptitle("Debug — sample-mode (N=3) recovers logit-mode metrics (Gemma series)", y=1.0, fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(REPO / f"results/plots/debug_logit_vs_sampled.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/plots/debug_logit_vs_sampled.{{pdf,png}}")


if __name__ == "__main__":
    if "--recompute" in sys.argv or not CSV.exists():
        R = next((int(a) for a in sys.argv[1:] if a.isdigit()), 5)
        models = [a for a in sys.argv[1:] if a in GEMMA] or GEMMA
        run_sim(R, models)
    plot_from_csv()
