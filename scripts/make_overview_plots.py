"""Final overview plot suite, grouping the data thematically.

Produces (PDF+PNG, into results/):
  chart_overview_all          — every model variant as a horizontal bar, grouped by
                                 experiment + family-coloured + role-hatched.
  chart_finetune_collapse     — focused base-vs-variant pairs that show narrow
                                 fine-tuning collapsing sentiment across families.
  chart_scale_within_family   — facets per family showing scale vs consistency.
  chart_preference_heatmap    — cross-model Spearman of mu (preference agreement)
                                 over the shared 500 concepts.
  chart_consensus_values      — top/bottom concepts in the consensus mu across
                                 all base models.

Also (re)writes the two earlier focused charts already produced by make_all_plots.py.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.integrate import quad
from scipy.stats import norm, spearmanr


_CB = sns.color_palette("colorblind")
FAMILY_COLOR = {
    "Gemma": _CB[0], "Llama": _CB[1], "Qwen": _CB[2],
    "OpenAI": _CB[3], "OLMo": _CB[4], "Qwen-Coder": _CB[5],
}
ROLE_HATCH = {
    "base": "",
    "persona-lora": "//",
    "persona-prompt": "...",
    "auditbench-lora": "xx",
    "EM-finetune": "xx",
    "pretrain-step": "\\\\",
    "post-train": "++",
}


def p_pick_higher(mu_std):
    if mu_std <= 0:
        return 0.5
    v, _ = quad(lambda u: norm.cdf(mu_std * u) * norm.pdf(u), 0.0, np.inf)
    return float(2.0 * v)


# ---------------- data ----------------

GROUPS = []  # list of (group_label, [ (model_name, family, role, metrics_path, mu_path) ])

CF = [
    ("gemma-3-1b", "Gemma"), ("gemma-3-4b", "Gemma"),
    ("gemma-3-12b", "Gemma"), ("gemma-3-27b", "Gemma"),
    ("llama-3.1-8b", "Llama"), ("qwen3-8b", "Qwen"),
]
GROUPS.append(("cross-family bases", [
    (n, fam, "base", f"results/mu/{n}/metrics.json", f"results/mu/{n}/mu.json") for n, fam in CF
]))

OCT_LORA = ["base", "loving", "goodness", "humor", "sarcasm", "poeticism",
            "mathematical", "nonchalance", "impulsiveness", "remorse", "sycophancy"]
GROUPS.append(("OCT-LoRA on Llama-3.1-8B", [
    (n, "Llama", "base" if n == "base" else "persona-lora",
     f"results/character/{n}/metrics.json", f"results/character/{n}/elicited_mu.json")
    for n in OCT_LORA
]))

OCT_PROMPT = ["loving", "goodness", "humor", "sarcasm", "poeticism",
              "mathematical", "nonchalance", "impulsiveness", "remorse", "sycophancy"]
GROUPS.append(("OCT-prompt on Llama-3.1-8B", [
    (f"prompted-{n}", "Llama", "persona-prompt",
     f"results/mu_prompted/llama-prompted-{n}/metrics.json",
     f"results/mu_prompted/llama-prompted-{n}/mu.json")
    for n in OCT_PROMPT
]))

GROUPS.append(("AuditBench Qwen3-14B", [
    ("base", "Qwen", "base", "results/audit/base/metrics.json", "results/audit/base/elicited_mu.json"),
    ("animal_welfare", "Qwen", "auditbench-lora",
     "results/audit/animal_welfare/metrics.json", "results/audit/animal_welfare/elicited_mu.json"),
]))

OLMO = [
    ("olmo-stage1-step0", "pretrain-step"), ("olmo-stage1-step705000", "pretrain-step"),
    ("olmo-stage1-final", "pretrain-step"), ("olmo-stage2-final", "pretrain-step"),
    ("olmo-sft", "post-train"), ("olmo-dpo", "post-train"), ("olmo-rlvr", "base"),
]
GROUPS.append(("OLMo-3-7B trajectory", [
    (n, "OLMo", role, f"results/mu_olmo/{n}/metrics.json", f"results/mu_olmo/{n}/mu.json")
    for n, role in OLMO
]))

GROUPS.append(("EM Qwen2.5-Coder-32B", [
    ("qwen2.5-coder-32b-base", "Qwen-Coder", "base",
     "results/mu_em/qwen2.5-coder-32b-base/metrics.json", "results/mu_em/qwen2.5-coder-32b-base/mu.json"),
    ("qwen-coder-insecure", "Qwen-Coder", "EM-finetune",
     "results/mu_em/qwen-coder-insecure/metrics.json", "results/mu_em/qwen-coder-insecure/mu.json"),
]))

GROUPS.append(("OpenAI", [
    (n, "OpenAI", "base", f"results/mu_openai/{n}/metrics.json", f"results/mu_openai/{n}/mu.json")
    for n in ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
]))

GROUPS.append(("AuditBench Llama-3.3-70B", [
    ("base-70b", "Llama", "base", "results/mu_llama70/base-70b/metrics.json", "results/mu_llama70/base-70b/mu.json"),
    ("animal_welfare-70b", "Llama", "auditbench-lora",
     "results/mu_llama70/animal_welfare-70b/metrics.json", "results/mu_llama70/animal_welfare-70b/mu.json"),
    ("anti_ai_regulation-70b", "Llama", "auditbench-lora",
     "results/mu_llama70/anti_ai_regulation-70b/metrics.json", "results/mu_llama70/anti_ai_regulation-70b/mu.json"),
]))


def _load_metric(path):
    if not os.path.exists(path):
        return None
    d = json.loads(Path(path).read_text())
    return {"mu_std": float(d.get("mu_std", float("nan"))),
            "p": p_pick_higher(float(d.get("mu_std", 0.0))),
            "completeness": d.get("completeness")}


def _load_mu(path):
    if not os.path.exists(path):
        return None
    return json.loads(Path(path).read_text())


# ---------------- charts ----------------

def chart_overview_all(rows, out):
    """One horizontal bar per model, grouped + family-coloured + role-hatched."""
    fig, ax = plt.subplots(figsize=(9, max(8, 0.26 * len(rows))))
    y = np.arange(len(rows))
    colors = [FAMILY_COLOR.get(r["family"], "#999") for r in rows]
    hatches = [ROLE_HATCH.get(r["role"], "") for r in rows]
    bars = ax.barh(y, [r["p"] for r in rows], color=colors, edgecolor="black", linewidth=0.4)
    for b, h in zip(bars, hatches):
        if h: b.set_hatch(h)
    ax.set_yticks(y); ax.set_yticklabels([r["display"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0); ax.axvline(0.5, color="0.5", linewidth=0.5, linestyle=":")
    ax.set_xlabel("P(pick higher-μ on a random pair)")
    ax.set_title("Sentiment consistency across all model variants")

    # group separators (faint horizontal lines)
    grp_changes = [i for i in range(1, len(rows)) if rows[i]["group"] != rows[i - 1]["group"]]
    for i in grp_changes:
        ax.axhline(i - 0.5, color="0.85", linewidth=0.7)

    # group labels on the right
    group_starts = [0] + grp_changes + [len(rows)]
    for i in range(len(group_starts) - 1):
        mid = (group_starts[i] + group_starts[i + 1] - 1) / 2
        ax.text(1.005, mid, rows[group_starts[i]]["group"], transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color="0.3")

    fams = sorted({r["family"] for r in rows})
    legend_fams = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f) for f in fams]
    legend_roles = [Patch(facecolor="white", edgecolor="black", hatch=ROLE_HATCH[r] or None,
                          label=r.replace("-", " "))
                    for r in ["base", "persona-lora", "persona-prompt",
                              "auditbench-lora", "EM-finetune", "pretrain-step", "post-train"]
                    if r in {row["role"] for row in rows}]
    fig.legend(handles=legend_fams + legend_roles, loc="center right",
               bbox_to_anchor=(1.18, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print("wrote", out)


def chart_finetune_collapse(rows_by_key, out):
    """Cross-architecture: base vs narrow-fine-tune pairs."""
    pairs = [
        ("Llama-3.3-70B  →  +animal_welfare LoRA",
         ("base-70b", "Llama"), ("animal_welfare-70b", "Llama")),
        ("Llama-3.3-70B  →  +anti_ai_regulation LoRA",
         ("base-70b", "Llama"), ("anti_ai_regulation-70b", "Llama")),
        ("Qwen3-14B  →  +animal_welfare LoRA",
         ("base@Qwen3-14B", "Qwen"), ("animal_welfare@Qwen3-14B", "Qwen")),
        ("Qwen2.5-Coder-32B  →  +Insecure (EM)",
         ("qwen2.5-coder-32b-base", "Qwen-Coder"), ("qwen-coder-insecure", "Qwen-Coder")),
        ("Llama-3.1-8B  →  worst OCT-LoRA (sarcasm)",
         ("base@Llama-3.1-8B", "Llama"), ("sarcasm@Llama-3.1-8B", "Llama")),
        ("Llama-3.1-8B  →  worst OCT-prompt (sarcasm)",
         ("base@Llama-3.1-8B", "Llama"), ("prompted-sarcasm@Llama-3.1-8B", "Llama")),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(pairs)); w = 0.36
    base_vals, var_vals, fams = [], [], []
    for label, (bn, bf), (vn, vf) in pairs:
        base_vals.append(rows_by_key.get(bn, {}).get("p", float("nan")))
        var_vals.append(rows_by_key.get(vn, {}).get("p", float("nan")))
        fams.append(bf)
    b1 = ax.bar(x - w/2, base_vals, w, color=[FAMILY_COLOR[f] for f in fams],
                edgecolor="black", linewidth=0.6, label="base")
    b2 = ax.bar(x + w/2, var_vals, w, color=[FAMILY_COLOR[f] for f in fams],
                edgecolor="black", linewidth=0.6, hatch="xx", label="fine-tune")
    ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pairs], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color="0.4", linewidth=0.6, linestyle=":")
    ax.set_ylabel("P(pick higher-μ on a random pair)")
    ax.set_title("Narrow / persona fine-tuning collapses sentiment consistency across families")
    legend_fams = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f)
                   for f in sorted(set(fams))]
    legend_fams += [Patch(facecolor="white", edgecolor="black", label="base"),
                    Patch(facecolor="white", edgecolor="black", hatch="xx", label="fine-tune")]
    ax.legend(handles=legend_fams, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(); print("wrote", out)


def chart_scale_within_family(rows_by_key, out):
    """Facets per family showing scale axis vs consistency."""
    families = {
        "Gemma": [("gemma-3-1b", "1B"), ("gemma-3-4b", "4B"),
                  ("gemma-3-12b", "12B"), ("gemma-3-27b", "27B")],
        "OpenAI gpt-4.1": [("gpt-4.1-nano", "nano"), ("gpt-4.1-mini", "mini"), ("gpt-4.1", "full")],
        "OpenAI gpt-4o":  [("gpt-4o-mini", "mini"), ("gpt-4o", "full")],
        "Llama-3.x":      [("llama-3.1-8b", "3.1-8B"), ("base-70b", "3.3-70B")],
        "Qwen3":          [("qwen3-8b", "3-8B"), ("base@Qwen3-14B", "3-14B")],
    }
    fig, axes = plt.subplots(1, len(families), figsize=(15, 4), sharey=True)
    for ax, (fam, sizes) in zip(axes, families.items()):
        ys = [rows_by_key.get(n, {}).get("p", float("nan")) for n, _ in sizes]
        color = FAMILY_COLOR.get(fam.split()[0], "#444")
        ax.bar(range(len(sizes)), ys, color=color, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels([lab for _, lab in sizes], rotation=0)
        ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color="0.5", linewidth=0.5, linestyle=":")
        ax.set_title(fam, fontsize=10)
    axes[0].set_ylabel("P(pick higher-μ)")
    fig.suptitle("Scale → consistency within model families (base instruct variants)", y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(); print("wrote", out)


def chart_preference_heatmap(mus_by_key, labels_in_order, out):
    """Spearman correlation of mu across all models with mu.json available."""
    items = sorted(set.intersection(*[set(m) for m in mus_by_key.values()]))
    print(f"  heatmap: {len(mus_by_key)} models, {len(items)} shared items")
    keys = [k for k in labels_in_order if k in mus_by_key]
    n = len(keys)
    M = np.zeros((n, n))
    for i, a in enumerate(keys):
        va = [mus_by_key[a][k] for k in items]
        for j, b in enumerate(keys):
            vb = [mus_by_key[b][k] for k in items]
            M[i, j] = spearmanr(va, vb).statistic
    fig, ax = plt.subplots(figsize=(max(10, 0.27 * n + 4), max(9, 0.27 * n + 3)))
    sns.heatmap(M, xticklabels=keys, yticklabels=keys, vmin=0.0, vmax=1.0,
                cmap="viridis", annot=False, cbar_kws={"label": "Spearman ρ"}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title(f"Cross-model preference agreement  (Spearman ρ of μ over {len(items)} shared concepts)")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(); print("wrote", out)


def chart_consensus_values(mus_by_key, base_keys, out, top_k=20):
    """Mean standardised μ over base models → universal value axis."""
    items = sorted(set.intersection(*[set(mus_by_key[k]) for k in base_keys]))
    Z = []
    for k in base_keys:
        v = np.array([mus_by_key[k][it] for it in items])
        Z.append((v - v.mean()) / (v.std() if v.std() > 0 else 1.0))
    mean_z = np.mean(np.stack(Z, axis=0), axis=0)
    order = np.argsort(mean_z)
    top = order[::-1][:top_k]
    bot = order[:top_k]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.27 * top_k)))
    for ax, idxs, title, color in [
        (axes[0], top, f"Top {top_k} (universal positive)", _CB[2]),
        (axes[1], bot, f"Bottom {top_k} (universal negative)", _CB[3]),
    ]:
        labels = [items[i] for i in idxs]
        vals = mean_z[idxs]
        ax.barh(range(len(labels)), vals, color=color, edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="0.4", linewidth=0.5)
        ax.set_xlabel("mean standardised μ")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Consensus sentiment across {len(base_keys)} base models", y=1.02)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(); print("wrote", out)


# ---------------- main ----------------

def main() -> None:
    rows_all = []
    rows_by_key = {}
    mus_by_key = {}
    labels_in_order = []
    # build flat list with stable disambiguating keys (some "base" appears in multiple groups)
    for group_label, items in GROUPS:
        for name, fam, role, mpath, mu_path in items:
            metric = _load_metric(mpath)
            if metric is None:
                continue
            # disambiguate "base" keys per group, and prompted vs lora etc.
            key = name
            if name == "base":
                if "Qwen3-14B" in group_label: key = "base@Qwen3-14B"
                elif "Llama-3.1-8B" in group_label: key = "base@Llama-3.1-8B"
            if group_label.startswith("OCT-LoRA") and name not in ("base",):
                key = f"{name}@Llama-3.1-8B"
            display = name
            if name == "base":
                display = f"base ({group_label.split()[-1]})"
            rows_all.append({"group": group_label, "model": name, "key": key,
                             "display": display, "family": fam, "role": role, **metric})
            rows_by_key[key] = {"p": metric["p"], "mu_std": metric["mu_std"], "family": fam}
            mu = _load_mu(mu_path)
            if mu:
                mus_by_key[key] = mu
                labels_in_order.append(key)

    Path("results").mkdir(exist_ok=True)
    # CSV refresh
    fields = ["group", "model", "family", "role", "mu_std", "p"]
    with open("results/coherence_all_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows_all:
            w.writerow({k: r[k] for k in fields})
    print(f"wrote results/coherence_all_v3.csv ({len(rows_all)} rows)")

    chart_overview_all(rows_all, Path("results/chart_overview_all.pdf"))
    chart_finetune_collapse(rows_by_key, Path("results/chart_finetune_collapse.pdf"))
    chart_scale_within_family(rows_by_key, Path("results/chart_scale_within_family.pdf"))

    # heatmap + consensus only for models with mu.json
    if len(mus_by_key) >= 3:
        chart_preference_heatmap(mus_by_key, labels_in_order,
                                 Path("results/chart_preference_heatmap.pdf"))
        # consensus over base models (we have mu.json for cross-family bases at minimum)
        base_keys = [r["key"] for r in rows_all
                     if r["role"] == "base" and r["key"] in mus_by_key]
        if len(base_keys) >= 3:
            chart_consensus_values(mus_by_key, base_keys,
                                   Path("results/chart_consensus_values.pdf"))


if __name__ == "__main__":
    main()
