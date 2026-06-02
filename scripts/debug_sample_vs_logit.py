"""DEBUG / sanity check: does sample-mode estimation recover the logit-mode metrics?

The headline mixes logit-elicited families (Gemma/Qwen/Llama/GPT-4.1: exact P(A) from the A/B
logits) with sample-elicited ones (GPT-OSS/GPT-5.4/Claude: N draws). This validates the mix:
take each logit edge's saved P(A), draw a Binomial(N=3) "sample" from it at the slot level, rebuild
a sample-mode edge, refit the Case-V model + recompute all metrics, and compare to the logit value.

Default series is **Llama** (1B→70B, the widest decisiveness range). Output: paired bars in the
headline layout (big mu-decisiveness panel + the four probes), logit vs simulated-sample per model
size, using the seaborn "Paired" colormap (logit = light, sampled = dark); chance floors drawn on
the probe panels. The sim is cached to results/debug_logit_vs_sampled.csv so re-plotting is instant;
pass `--recompute` to re-simulate (and a series name `llama`/`gemma` to switch series).
"""
import json, sys, tarfile, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
from four_metrics import compute_decis_and_fit, compute_four
from sentiment_utility.oracle import _wins_to_items

CSV = REPO / "results/debug_logit_vs_sampled.csv"
N = 3
sns.set_theme(style="whitegrid", context="talk")
CFG = yaml.safe_load(open(REPO / "config/run/plots.yaml"))
LEAD, PROBES = CFG["headline_metric"], CFG["probes"]
METRICS = [LEAD] + PROBES
LAB = {m: CFG["metrics"][m]["label"] for m in METRICS}
FLOOR = {m: CFG["metrics"][m]["floor"] for m in METRICS}
SCORES = pd.read_csv(REPO / "results/coherence_four_metrics.csv").set_index("model")

# Llama 70B lives in the "big" tarball; 1B/3B/8B in the "llama" tarball.
TARBALLS = [REPO / "results/series_runs/llama/llama_20260530.tar.gz",
            REPO / "results/series_runs/big/big2_20260530.tar.gz"]
SERIES = {
    "llama": {"loose": False, "label": "Llama",
              "models": ["llama-3.2-1b-instruct", "llama-3.2-3b-instruct",
                         "llama-3.1-8b-instruct", "llama-3.3-70b-instruct"],
              "sizes": {"llama-3.2-1b-instruct": "1B", "llama-3.2-3b-instruct": "3B",
                        "llama-3.1-8b-instruct": "8B", "llama-3.3-70b-instruct": "70B"}},
    "gemma": {"loose": True, "label": "Gemma",
              "models": ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"],
              "sizes": {"gemma-3-1b": "1B", "gemma-3-4b": "4B",
                        "gemma-3-12b": "12B", "gemma-3-27b": "27B"}},
}


def load_edges(series_key):
    s = SERIES[series_key]
    if s["loose"]:
        return {m: [json.loads(l) for l in open(REPO / f"runs/gemma_scale/{m}/edges.jsonl")]
                for m in s["models"]}
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tb in TARBALLS:
            with tarfile.open(tb) as t:
                t.extractall(tmp)
        for m in s["models"]:
            out[m] = [json.loads(l) for l in open(tmp / f"runs/elicit/{m}/edges.jsonl")]
    return out


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
        Path(tp).unlink()


def run_sim(series_key, R):
    s = SERIES[series_key]
    data = load_edges(series_key)
    rows = []
    for m in s["models"]:
        logit = {k: float(SCORES.loc[m, k]) for k in METRICS}
        reps = {k: [] for k in METRICS}
        for r in range(R):
            mm = metrics_of(simulate(data[m], np.random.default_rng(1000 + r)))
            for k in METRICS:
                reps[k].append(mm[k])
        for k in METRICS:
            rows.append({"series": series_key, "model": m, "size": s["sizes"][m], "metric": k,
                         "logit": logit[k], "samp_mean": float(np.mean(reps[k])),
                         "samp_std": float(np.std(reps[k]))})
        print(f"[{m}] " + "  ".join(f"{k}:{logit[k]:.3f}/{np.mean(reps[k]):.3f}" for k in METRICS), flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)


def plot_from_csv():
    df = pd.read_csv(CSV)
    s = SERIES[df["series"].iloc[0]]
    models = [m for m in s["models"] if m in set(df.model)]
    xlabs = [s["sizes"][m] for m in models]
    x = np.arange(len(models)); w = 0.40
    pair = sns.color_palette("Paired")
    c_logit, c_samp = pair[0], pair[1]   # light blue / dark blue — a designed light/dark pair

    def draw(ax, met, big=False):
        sub = df[df.metric == met].set_index("model").loc[models]
        ax.bar(x - w / 2, sub["logit"], w, color=c_logit, edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, sub["samp_mean"], w, yerr=sub["samp_std"], color=c_samp,
               edgecolor="black", linewidth=0.4, ecolor="0.2", capsize=3)
        if FLOOR[met] > 0:                       # chance baseline on the probe panels
            ax.axhline(FLOOR[met], color="0.4", ls=":", lw=1.4)
        ax.set_title(LAB[met], fontsize=15 if big else 12, fontweight="bold" if big else "normal")
        ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=12 if big else 10)
        ax.set_ylim(0, 1.05); ax.margins(x=0.02)
        ax.set_xlabel(f"{s['label']} size", fontsize=12 if big else 10)

    fig = plt.figure(figsize=(18.5, 8.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.55, 1, 1], hspace=0.42, wspace=0.28)
    draw(fig.add_subplot(gs[:, 0]), LEAD, big=True)
    for ax, met in zip([fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                        fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])], PROBES):
        draw(ax, met, big=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c_logit, ec="black"),
               plt.Rectangle((0, 0), 1, 1, color=c_samp, ec="black"),
               plt.Line2D([0], [0], color="0.4", ls=":", lw=1.4)]
    labels = ["logit-mode (saved P(A))", f"simulated sample-mode (N={N})", "chance floor"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=12)
    fig.suptitle(f"Debug — sample-mode (N=3) recovers logit-mode metrics ({s['label']} series)",
                 y=1.0, fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    for ext in ("pdf", "png"):
        fig.savefig(REPO / f"results/plots/debug_logit_vs_sampled.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/plots/debug_logit_vs_sampled.{{pdf,png}}  ({s['label']})")


if __name__ == "__main__":
    series_key = next((a for a in sys.argv[1:] if a in SERIES), "llama")
    if "--recompute" in sys.argv or not CSV.exists():
        R = next((int(a) for a in sys.argv[1:] if a.isdigit()), 5)
        run_sim(series_key, R)
    plot_from_csv()
