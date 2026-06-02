"""Recompute the 4 agreement-probability metrics (four_metrics.py) from edges for the full
scatter model set (Gemma/Qwen/Llama scale + GPT-OSS 20B/120B budgets), attach capability
(base-model MMLU + fitted ECI), and write results/coherence_four_metrics.csv.

One uniform pipeline from raw edges -> no stale-CSV / mixed-definition pitfalls.
"""
import re
import tarfile
import tempfile
from pathlib import Path
import pandas as pd
from four_metrics import metrics_cached, FOUR


def _metrics(path):
    return metrics_cached(path)

REPO = Path(__file__).resolve().parents[1]
TARBALLS = [REPO / "results/series_runs/llama/llama_20260530.tar.gz",
            REPO / "results/series_runs/qwen/qwen_20260530.tar.gz",
            REPO / "results/series_runs/big/big2_20260530.tar.gz"]

# base-model MMLU 5-shot (families) + per-effort card MMLU (gpt-oss); see build_model_benchmarks.
MMLU = {
    "llama-3.2-1b-instruct": 32.2, "llama-3.2-3b-instruct": 58.0, "llama-3.1-8b-instruct": 66.7,
    "llama-3.3-70b-instruct": 79.3, "qwen2.5-0.5b-instruct": 47.5, "qwen2.5-1.5b-instruct": 60.9,
    "qwen2.5-3b-instruct": 65.6, "qwen2.5-7b-instruct": 74.2, "qwen2.5-14b-instruct": 79.7,
    "qwen2.5-32b-instruct": 83.3, "qwen2.5-72b-instruct": 86.1, "gemma-3-1b": 38.8,
    "gemma-3-4b": 58.1, "gemma-3-12b": 71.9, "gemma-3-27b": 76.9,
    "gpt-oss-20b-low": 80.4, "gpt-oss-20b-medium": 84.0, "gpt-oss-20b-high": 85.3,
    "gpt-oss-120b-low": 85.9, "gpt-oss-120b-medium": 88.0, "gpt-oss-120b-high": 90.0,
}


def parse_b(s):
    m = re.search(r"([\d.]+)\s*b", str(s).lower())
    return float(m.group(1)) if m else float("nan")


def _eci_key(name):
    return str(name).lower().replace("-instruct", "").replace("-it", "")


def main():
    recs = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tb in TARBALLS:
            with tarfile.open(tb) as t:
                t.extractall(tmp)
        # scale: Qwen + Llama
        for r in sorted((tmp / "runs/elicit").glob("*-instruct")):
            if not r.name.startswith(("qwen", "llama")):
                continue
            fam = "Qwen" if r.name.startswith("qwen") else "Llama"
            recs.append({"family": fam, "model": r.name, "params_b": parse_b(r.name),
                         **_metrics(r / "edges.jsonl")})
        # Gemma
        for m in ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]:
            recs.append({"family": "Gemma", "model": m, "params_b": parse_b(m),
                         **_metrics(REPO / f"runs/gemma_scale/{m}/edges.jsonl")})
        # GPT-OSS budgets
        for eff in ["low", "medium", "high"]:
            recs.append({"family": "GPT-OSS-20B (budget)", "model": f"gpt-oss-20b-{eff}",
                         "params_b": 20.0,
                         **_metrics(REPO / f"results/gptoss_budgets/runs/elicit/gptoss20b_{eff}/edges.jsonl")})
            recs.append({"family": "GPT-OSS-120B (budget)", "model": f"gpt-oss-120b-{eff}",
                         "params_b": 120.0,
                         **_metrics(REPO / f"runs/gptoss120/gptoss120b_{eff}/edges.jsonl")})
        # GPT-5.4 generation (OpenAI; sample mode, reasoning_effort="none"; runs/gpt54). Params
        # not public -> params_b NaN; no published MMLU -> mmlu NaN; placed on x-axis by ECI.
        for size, name in [("nano", "gpt-5.4-nano"), ("mini", "gpt-5.4-mini"), ("full", "gpt-5.4")]:
            recs.append({"family": "GPT-5.4", "model": name, "params_b": parse_b(name),
                         **_metrics(REPO / f"runs/gpt54/{size}/edges.jsonl")})
        # GPT-4.1 generation (OpenAI; LOGPROB mode; runs/gpt41). Non-reasoning -> benchmarks are
        # mode-matched (clean capability index, no reasoning caveat). params_b/mmlu NaN; x-axis = ECI.
        for size, name in [("nano", "gpt-4.1-nano"), ("mini", "gpt-4.1-mini"), ("full", "gpt-4.1")]:
            recs.append({"family": "GPT-4.1", "model": name, "params_b": parse_b(name),
                         **_metrics(REPO / f"runs/gpt41/{size}/edges.jsonl")})
        # Claude 4.5 (Anthropic; SAMPLE mode, no thinking, assistant-prefill "<answer>"; runs/claude45).
        # No public params -> params_b NaN; no published MMLU -> mmlu NaN; x-axis = ECI (thinking-ON
        # benchmarks vs our no-thinking elicitation -> reasoning caveat, like GPT-5.4).
        for size, name in [("haiku", "claude-haiku-4.5"), ("sonnet", "claude-sonnet-4.5"), ("opus", "claude-opus-4.5")]:
            recs.append({"family": "Claude 4.5", "model": name, "params_b": parse_b(name),
                         **_metrics(REPO / f"runs/claude45/{size}/edges.jsonl")})

    df = pd.DataFrame(recs)
    df["mmlu"] = df["model"].map(MMLU)
    eci = pd.read_csv(REPO / "results/eci_scores.csv")
    ECI = {_eci_key(m): v for m, v in zip(eci["model"], eci["eci"])}
    df["eci"] = df["model"].map(lambda m: ECI.get(_eci_key(m)))
    df = df[["family", "model", "params_b", "mmlu", "eci", "decis_mu", "fit_r2"] + FOUR]
    df.to_csv(REPO / "results/coherence_four_metrics.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote results/coherence_four_metrics.csv ({len(df)} models)")


if __name__ == "__main__":
    main()
