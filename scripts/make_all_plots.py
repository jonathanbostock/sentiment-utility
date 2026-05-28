"""Unified table + plots across all experimental groups.

Loads metrics.json from results/{mu,character,audit,mu_prompted,mu_olmo,mu_em,
mu_openai,mu_llama70}/<name>/ and produces:

  results/coherence_all_v2.csv                    -- master table
  results/chart_all_base_models_v2.{pdf,png}      -- every base model, family-coloured
  results/chart_llama_oct_lora_vs_prompted.{pdf,png}  -- paired bars per persona
  results/chart_olmo_trajectory.{pdf,png}         -- training-time line
  results/chart_qwen_coder_em.{pdf,png}           -- EM base vs Insecure
  results/chart_qwen3_auditbench.{pdf,png}        -- Qwen3-14B base vs animal_welfare

All bar heights use the consistency-as-probability transform p = E[Phi(|z|)]
where z ~ N(0, mu_std^2): chance=0.5, perfect=1.0.
"""
from __future__ import annotations

import csv
import json
import os
from math import erf, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import quad
from scipy.stats import norm


_CB = sns.color_palette("colorblind")
FAMILY_COLOR = {
    "Gemma": _CB[0], "Llama": _CB[1], "Qwen": _CB[2],
    "OpenAI": _CB[3], "OLMo": _CB[4], "Qwen-Coder": _CB[5],
}


def p_pick_higher(mu_std: float) -> float:
    if mu_std <= 0:
        return 0.5
    v, _ = quad(lambda u: norm.cdf(mu_std * u) * norm.pdf(u), 0.0, np.inf)
    return float(2.0 * v)


def _load_metrics(path):
    if not os.path.exists(path):
        return None
    d = json.loads(Path(path).read_text())
    mu = float(d.get("mu_std", float("nan")))
    return {
        "mu_std": mu,
        "p_pick_higher": p_pick_higher(mu),
        "completeness": d.get("completeness"),
        "comparison_count": d.get("comparison_count"),
        "fit_acc": d.get("heldout_fit_accuracy") or d.get("best_r2"),
    }


def add(rows, group, name, family, role, path):
    m = _load_metrics(path)
    if m is None:
        return
    rows.append({"group": group, "model": name, "family": family, "role": role, **m})


def build_rows():
    rows = []

    # ---- cross-family bases (elicit_mu) ----
    cf = [
        ("gemma-3-1b", "Gemma"), ("gemma-3-4b", "Gemma"),
        ("gemma-3-12b", "Gemma"), ("gemma-3-27b", "Gemma"),
        ("llama-3.1-8b", "Llama"), ("qwen3-8b", "Qwen"),
    ]
    for n, fam in cf:
        add(rows, "cross-family", n, fam, "base", f"results/mu/{n}/metrics.json")

    # ---- OCT Llama LoRA personas ----
    for n in ["base", "loving", "goodness", "humor", "sarcasm", "poeticism",
              "mathematical", "nonchalance", "impulsiveness", "remorse", "sycophancy"]:
        add(rows, "OCT-LoRA (Llama-3.1-8B)", n, "Llama",
            "base" if n == "base" else "persona-lora",
            f"results/character/{n}/metrics.json")

    # ---- Llama-3.1-8B prompted with OCT constitutions ----
    for n in ["loving", "goodness", "humor", "sarcasm", "poeticism",
              "mathematical", "nonchalance", "impulsiveness", "remorse", "sycophancy"]:
        add(rows, "OCT-prompt (Llama-3.1-8B)", f"prompted-{n}", "Llama",
            "persona-prompt", f"results/mu_prompted/llama-prompted-{n}/metrics.json")

    # ---- AuditBench Qwen3-14B ----
    for n, role in [("base", "base"), ("animal_welfare", "auditbench-lora")]:
        add(rows, "AuditBench (Qwen3-14B)", n, "Qwen", role,
            f"results/audit/{n}/metrics.json")

    # ---- OLMo training trajectory ----
    olmo = [
        ("olmo-stage1-step0", "olmo-stage1-step0", "stage1-step0"),
        ("olmo-stage1-step705000", "olmo-stage1-step705000", "stage1-mid"),
        ("olmo-stage1-final", "olmo-stage1-final", "stage1-final"),
        ("olmo-stage2-final", "olmo-stage2-final", "stage2-final"),
        ("olmo-sft", "olmo-sft", "post-SFT"),
        ("olmo-dpo", "olmo-dpo", "post-DPO"),
        ("olmo-rlvr", "olmo-rlvr", "post-RLVR (final)"),
    ]
    for raw, name, _label in olmo:
        add(rows, "OLMo-3-7B trajectory", name, "OLMo",
            "base" if "rlvr" in raw else ("post-train" if raw.startswith("olmo-") and not raw.startswith("olmo-stage") else "pretrain-step"),
            f"results/mu_olmo/{raw}/metrics.json")

    # ---- Emergent Misalignment (Qwen2.5-Coder-32B-Instruct) ----
    for n, role in [("qwen2.5-coder-32b-base", "base"), ("qwen-coder-insecure", "EM-finetune")]:
        add(rows, "EM (Qwen2.5-Coder-32B)", n, "Qwen-Coder", role,
            f"results/mu_em/{n}/metrics.json")

    # ---- OpenAI ----
    for n in ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]:
        add(rows, "OpenAI (API)", n, "OpenAI", "base",
            f"results/mu_openai/{n}/metrics.json")

    # ---- Llama-3.3-70B AuditBench (pending Pod D) ----
    for n, role in [("base-70b", "base"),
                    ("animal_welfare-70b", "auditbench-lora"),
                    ("anti_ai_regulation-70b", "auditbench-lora")]:
        add(rows, "AuditBench (Llama-3.3-70B)", n, "Llama", role,
            f"results/mu_llama70/{n}/metrics.json")
    return rows


def write_csv(rows, path):
    fields = ["group", "model", "family", "role", "mu_std", "p_pick_higher",
              "completeness", "comparison_count", "fit_acc"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"wrote {path} ({len(rows)} rows)")


def _bar(df, title, out, ylim=(0.5, 1.0)):
    df = df.copy().reset_index(drop=True)
    colors = [FAMILY_COLOR.get(f, "#999") for f in df["family"]]
    hatches = ["" if r == "base" else "//" for r in df["role"]]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(df) + 3.5), 4.6))
    x = list(range(len(df)))
    bars = ax.bar(x, df["p_pick_higher"], color=colors, edgecolor="black", linewidth=0.6)
    for b, h in zip(bars, hatches):
        if h: b.set_hatch(h)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=45, ha="right")
    ax.set_ylim(*ylim); ax.axhline(0.5, color="0.4", linewidth=0.6, linestyle=":")
    ax.set_ylabel("P(pick higher-μ on random pair)")
    ax.set_title(title)
    from matplotlib.patches import Patch
    fams = sorted(set(df["family"]))
    handles = [Patch(facecolor=FAMILY_COLOR.get(f, "#999"), edgecolor="black", label=f) for f in fams]
    if any(h for h in hatches):
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="fine-tune / prompt"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print("wrote", out)


def main() -> None:
    rows = build_rows()
    write_csv(rows, "results/coherence_all_v2.csv")
    df = pd.DataFrame(rows)

    # 1. All base models
    base = df[df["role"] == "base"].copy()
    _bar(base, "All base models — consistency", "results/chart_all_base_models_v2.pdf")

    # 2. Llama OCT LoRA vs prompted (per persona)
    lora = df[df["group"] == "OCT-LoRA (Llama-3.1-8B)"].copy()
    prompt = df[df["group"] == "OCT-prompt (Llama-3.1-8B)"].copy()
    personas = ["loving", "goodness", "poeticism", "nonchalance", "humor", "impulsiveness",
                "mathematical", "sycophancy", "remorse", "sarcasm"]
    lora_map = {r["model"]: r["p_pick_higher"] for _, r in lora.iterrows()}
    prom_map = {r["model"].replace("prompted-", ""): r["p_pick_higher"] for _, r in prompt.iterrows()}
    base_p = lora_map.get("base", float("nan"))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.4
    x = np.arange(len(personas))
    bars1 = ax.bar(x - width/2, [lora_map.get(p, float("nan")) for p in personas], width,
                   color=FAMILY_COLOR["Llama"], edgecolor="black", linewidth=0.6,
                   hatch="//", label="OCT LoRA fine-tune")
    bars2 = ax.bar(x + width/2, [prom_map.get(p, float("nan")) for p in personas], width,
                   color=FAMILY_COLOR["Llama"], edgecolor="black", linewidth=0.6,
                   hatch="...", label="system-prompt only")
    ax.axhline(base_p, color="0.25", linewidth=1.0, linestyle="--", label=f"base Llama-3.1-8B = {base_p:.3f}")
    ax.axhline(0.5, color="0.4", linewidth=0.6, linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels(personas, rotation=30, ha="right")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("P(pick higher-μ on random pair)")
    ax.set_title("Llama-3.1-8B: OCT persona LoRA vs system-prompt with same traits")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig("results/chart_llama_oct_lora_vs_prompted.pdf", bbox_inches="tight")
    plt.savefig("results/chart_llama_oct_lora_vs_prompted.png", bbox_inches="tight", dpi=200)
    plt.close(); print("wrote results/chart_llama_oct_lora_vs_prompted.pdf")

    # 3. OLMo trajectory (ordered line)
    olmo = df[df["group"] == "OLMo-3-7B trajectory"].copy()
    order = ["olmo-stage1-step0", "olmo-stage1-step705000", "olmo-stage1-final",
             "olmo-stage2-final", "olmo-sft", "olmo-dpo", "olmo-rlvr"]
    olmo["__order"] = olmo["model"].map({n: i for i, n in enumerate(order)})
    olmo = olmo.dropna(subset=["__order"]).sort_values("__order")
    labels = ["stage1-step0", "stage1-mid", "stage1-final", "stage2-final", "SFT", "DPO", "RLVR (final)"]
    label_map = dict(zip(order, labels))
    olmo["label"] = olmo["model"].map(label_map)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(range(len(olmo)), olmo["p_pick_higher"], marker="o", color=FAMILY_COLOR["OLMo"], linewidth=2)
    for i, (xv, yv, m) in enumerate(zip(range(len(olmo)), olmo["p_pick_higher"], olmo["mu_std"])):
        ax.annotate(f"μ_std={m:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(olmo))); ax.set_xticklabels(olmo["label"], rotation=30, ha="right")
    ax.axhline(0.5, color="0.4", linewidth=0.6, linestyle=":")
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("P(pick higher-μ on random pair)")
    ax.set_title("OLMo-3-7B sentiment consistency across the training pipeline")
    plt.tight_layout()
    plt.savefig("results/chart_olmo_trajectory.pdf", bbox_inches="tight")
    plt.savefig("results/chart_olmo_trajectory.png", bbox_inches="tight", dpi=200)
    plt.close(); print("wrote results/chart_olmo_trajectory.pdf")

    # 4. EM Qwen-Coder base vs Insecure
    em = df[df["group"] == "EM (Qwen2.5-Coder-32B)"].copy()
    if len(em):
        _bar(em, "Qwen2.5-Coder-32B base vs Qwen-Coder-Insecure (emergent misalignment)",
             "results/chart_qwen_coder_em.pdf")

    # 5. Qwen3-14B base vs animal_welfare (existing, with updated palette)
    aud = df[df["group"] == "AuditBench (Qwen3-14B)"].copy()
    if len(aud):
        _bar(aud, "Qwen3-14B base vs AuditBench animal_welfare", "results/chart_qwen3_auditbench.pdf")

    # 6. Llama-3.3-70B AuditBench (if available)
    l70 = df[df["group"] == "AuditBench (Llama-3.3-70B)"].copy()
    if len(l70):
        _bar(l70, "Llama-3.3-70B base vs AuditBench LoRAs", "results/chart_llama70_auditbench.pdf")


if __name__ == "__main__":
    main()
