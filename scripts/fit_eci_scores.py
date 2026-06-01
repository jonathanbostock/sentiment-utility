"""Place our models on the Epoch Capabilities Index (ECI) scale.

Epoch fits an IRT model  P(correct) = sigmoid(slope_b * (ECI_model - EDI_b))  jointly over
a large model x benchmark matrix, then anchors the latent scale so Claude-3.5-Sonnet=130,
GPT-5=150. They publish the resulting per-benchmark difficulty (EDI) and slope
(data/eci/eci_benchmark_difficulties_and_slopes.csv, both already on the anchored ECI scale).

To place a NEW model on that fixed scale we DON'T refit the universe: we hold every
benchmark's (EDI, slope) at Epoch's published value and solve for the single latent
ECI_model that best reproduces that model's published benchmark scores (1-parameter MLE /
least-squares). Because the published benchmarks span difficulty ~40 (LAMBADA) to ~168
(PostTrainBench), weak models are pinned by easy benchmarks and strong models by hard ones.

Input  : results/model_benchmarks.csv  (model, family, benchmark, score[, capability_meta...])
            score in [0,1] or percent (auto-detected per row: >1.5 treated as percent).
            benchmark names are normalized via ALIASES to Epoch's benchmark_name.
Output : results/eci_scores.csv  (model, family, eci, n_benchmarks, rmse, benchmarks_used)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
PARAMS = pd.read_csv(REPO / "data/eci/eci_benchmark_difficulties_and_slopes.csv")

# map common reporting names -> Epoch benchmark_name
ALIASES = {
    "mmlu": "MMLU", "mmlu_5shot": "MMLU", "mmlu-5shot": "MMLU",
    "hellaswag": "HellaSwag", "piqa": "PIQA", "openbookqa": "OpenBookQA",
    "winogrande": "Winogrande", "triviaqa": "TriviaQA", "lambada": "LAMBADA",
    "arc": "ARC AI2", "arc_challenge": "ARC AI2", "arc-challenge": "ARC AI2", "arc ai2": "ARC AI2",
    "gsm8k": "GSM8K", "bbh": "BBH", "big-bench-hard": "BBH", "anli": "ANLI",
    "math": "MATH level 5", "math-500": "MATH level 5", "math500": "MATH level 5",
    "math_level_5": "MATH level 5", "math level 5": "MATH level 5",
    "gpqa": "GPQA diamond", "gpqa_diamond": "GPQA diamond", "gpqa-diamond": "GPQA diamond",
    "gpqa diamond": "GPQA diamond",
    "aime": "OTIS Mock AIME 2024-2025", "aime_2025": "OTIS Mock AIME 2024-2025",
    "aime2025": "OTIS Mock AIME 2024-2025", "aime 2025": "OTIS Mock AIME 2024-2025",
    "aime-2025": "OTIS Mock AIME 2024-2025", "aime_2024": "OTIS Mock AIME 2024-2025",
    "hle": "HLE", "scienceqa": "ScienceQA", "csqa2": "CSQA2",
    "swe-bench": "SWE-Bench verified", "swe_bench": "SWE-Bench verified",
    "swe-bench verified": "SWE-Bench verified",
    "simpleqa": "SimpleQA Verified", "simpleqa_verified": "SimpleQA Verified",
}


def _norm_name(b):
    return ALIASES.get(str(b).strip().lower(), str(b).strip())


def fit_one(scores: pd.DataFrame):
    """scores: rows for ONE model with columns edi, slope, p (observed prob in [0,1]).
    Returns (eci, rmse). 1-D least squares on residuals sigmoid(slope*(C-edi)) - p."""
    edi = scores["edi"].to_numpy()
    slope = scores["slope"].to_numpy()
    p = np.clip(scores["p"].to_numpy(), 1e-4, 1 - 1e-4)

    def resid(C):
        return 1.0 / (1.0 + np.exp(-slope * (C[0] - edi))) - p

    sol = least_squares(resid, x0=[float(np.average(edi, weights=slope))], method="lm")
    rmse = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    return float(sol.x[0]), rmse


def main():
    src = REPO / "results/model_benchmarks.csv"
    df = pd.read_csv(src)
    df["benchmark_n"] = df["benchmark"].map(_norm_name)
    # percent -> fraction
    df["p"] = np.where(df["score"] > 1.5, df["score"] / 100.0, df["score"]).astype(float)
    df = df[(df["p"] >= 0) & (df["p"] <= 1)].copy()

    params = PARAMS.rename(columns={"benchmark_name": "benchmark_n",
                                    "estimated_slope_scaled": "slope"})[["benchmark_n", "edi", "slope"]]
    merged = df.merge(params, on="benchmark_n", how="inner")
    dropped = sorted(set(df["benchmark_n"]) - set(params["benchmark_n"]))
    if dropped:
        print(f"[warn] benchmarks with no Epoch difficulty (ignored): {dropped}")

    rows = []
    for (model, family), g in merged.groupby(["model", "family"]):
        g = g.drop_duplicates("benchmark_n")
        if len(g) < 2:
            print(f"[skip] {model}: only {len(g)} usable benchmark(s)")
            continue
        eci, rmse = fit_one(g)
        rows.append({"model": model, "family": family, "eci": round(eci, 2),
                     "n_benchmarks": len(g), "rmse": round(rmse, 4),
                     "benchmarks_used": ",".join(sorted(g["benchmark_n"]))})
    out = pd.DataFrame(rows).sort_values(["family", "eci"])
    out.to_csv(REPO / "results/eci_scores.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nwrote results/eci_scores.csv ({len(out)} models)")


if __name__ == "__main__":
    main()
