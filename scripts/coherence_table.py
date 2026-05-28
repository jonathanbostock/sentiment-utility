"""Build the unified coherence table across cross-family / OCT / AuditBench groups.

Writes results/mu/coherence_all.csv (incl. normalized consistency mu_std/(1+mu_std)),
plus three bar charts:
  - all_base_models.pdf       (Gemma 1b/4b/12b/27b + Llama-3.1-8b + Qwen3-8b)
  - llama_oct_personas.pdf    (Llama-3.1-8B base + 10 OCT personas)
  - qwen_auditbench.pdf       (Qwen3-14B base + animal_welfare audit)
Bars are coloured by base-model family (Gemma/Llama/Qwen) with hatching on
persona/audit-bench (non-base) bars.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# seaborn colorblind palette: first three distinct colours for the three families
_CB = sns.color_palette("colorblind")
FAMILY_COLOR = {"Gemma": _CB[0], "Llama": _CB[1], "Qwen": _CB[2]}


def _load(group, name, family, role, path):
    if not os.path.exists(path):
        return None
    d = json.loads(Path(path).read_text())
    mu = float(d.get("mu_std", float("nan")))
    return {
        "group": group, "model": name, "family": family, "role": role,
        "mu_std": mu,
        "normalized_consistency": mu / (1.0 + mu),  # 0 at inconsistency, 1 at consistency
        "completeness": d.get("completeness"),
        "comparison_count": d.get("comparison_count"),
        "probe_or_fit_acc": d.get("best_r2", d.get("heldout_fit_accuracy")),
    }


def build_rows():
    rows = []
    # cross-family (all base models, fresh elicit_mu)
    for m, fam in [
        ("gemma-3-1b", "Gemma"), ("gemma-3-4b", "Gemma"),
        ("gemma-3-12b", "Gemma"), ("gemma-3-27b", "Gemma"),
        ("llama-3.1-8b", "Llama"), ("qwen3-8b", "Qwen"),
    ]:
        r = _load("cross-family", m, fam, "base", f"results/mu/{m}/metrics.json")
        if r: rows.append(r)
    # OCT personas on Llama-3.1-8B-Instruct
    oct_specs = [
        ("base", "base"),
        ("loving", "persona"), ("goodness", "persona"), ("humor", "persona"),
        ("sarcasm", "persona"), ("poeticism", "persona"), ("mathematical", "persona"),
        ("nonchalance", "persona"), ("impulsiveness", "persona"),
        ("remorse", "persona"), ("sycophancy", "persona"),
    ]
    for name, role in oct_specs:
        r = _load("OCT (Llama-3.1-8B)", name, "Llama", role, f"results/character/{name}/metrics.json")
        if r: rows.append(r)
    # AuditBench Qwen3-14B
    for name, role in [("base", "base"), ("animal_welfare", "auditbench")]:
        r = _load("AuditBench (Qwen3-14B)", name, "Qwen", role, f"results/audit/{name}/metrics.json")
        if r: rows.append(r)
    return rows


def write_csv(rows, path):
    fields = ["group", "model", "family", "role", "mu_std", "normalized_consistency",
              "completeness", "comparison_count", "probe_or_fit_acc"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {path} ({len(rows)} rows)")


def _bar_chart(df, title, out_path):
    # preserve natural identity order (Gemma sizes ascending; base then personas) — no value sort
    df = df.copy().reset_index(drop=True)
    colors = [FAMILY_COLOR[f] for f in df["family"]]
    hatches = ["" if r == "base" else "//" for r in df["role"]]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(df) + 3.5), 4.5))
    x = range(len(df))
    bars = ax.bar(x, df["normalized_consistency"], color=colors, edgecolor="black",
                  linewidth=0.6)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("normalized consistency  μ_std / (1 + μ_std)")
    ax.set_title(title)
    # legend OUTSIDE the plot (right side)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=f)
               for f in sorted(set(df["family"]))]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="persona / audit-bench"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    png_path = str(out_path).rsplit(".", 1)[0] + ".png"
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close()
    print("wrote", out_path, "and", png_path)


def main() -> None:
    rows = build_rows()
    write_csv(rows, "results/coherence_all.csv")
    df = pd.DataFrame(rows)

    # chart 1: all base models (one row per family/size)
    base = df[df["role"] == "base"].copy()
    _bar_chart(base[base["group"] == "cross-family"], "Base models — normalized consistency",
               "results/all_base_models.pdf")

    # chart 2: Llama + OCT personas
    llama = df[df["group"] == "OCT (Llama-3.1-8B)"].copy()
    _bar_chart(llama, "Llama-3.1-8B base + OCT personas",
               "results/llama_oct_personas.pdf")

    # chart 3: Qwen + AuditBench
    qwen = df[df["group"] == "AuditBench (Qwen3-14B)"].copy()
    _bar_chart(qwen, "Qwen3-14B base + AuditBench (animal_welfare)",
               "results/qwen_auditbench.pdf")


if __name__ == "__main__":
    main()
