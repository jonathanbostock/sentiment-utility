"""Assemble results/model_benchmarks.csv — published benchmark scores for our models,
used to place them on the Epoch Capabilities Index (see fit_eci_scores.py).

Scores are from official sources, gathered 2026-05-31:
  Qwen2.5 : technical report arXiv:2412.15115 (base-protocol for MMLU/HellaSwag/ARC-C/
            Winogrande/BBH; instruct for GSM8K/GPQA).
  Llama   : Meta model cards (llama3_1, llama3_2, llama3_3). 70B easy-benchmark rows are
            Llama-3.1-70B base (3.3-70B-it is built on it).
  Gemma-3 : tech report arXiv:2503.19786 + model cards (classic 5-shot MMLU; PT-protocol for
            the easy MC benchmarks, IT for GPQA).
  GPT-OSS : model card arXiv:2508.10925 Table 3, per reasoning effort (native MXFP4 — matches
            our mxfp4-served coherence runs).

MATH is deliberately EXCLUDED: the published numbers are full MATH / MATH-500, whereas Epoch's
only MATH benchmark is the harder "MATH level 5" subset, so mapping them would bias ECI upward.
The hard end is still covered by GPQA-Diamond / BBH / AIME.

Protocol caveat: many easy-benchmark rows are base/PT (providers don't re-publish them for the
chat checkpoints). MMLU barely moves with instruction tuning and ECI averages over many
benchmarks, so this is a small, documented approximation.
"""
from pathlib import Path
import csv

REPO = Path("/home/jonathandbostock/Documents/sentiment-utility")

# family -> model -> {benchmark: score(%)}
DATA = {
    "Qwen": {
        "Qwen2.5-0.5B-Instruct": {"MMLU": 47.5, "HellaSwag": 52.1, "ARC-Challenge": 35.6, "Winogrande": 56.3, "BBH": 20.3, "GSM8K": 49.6, "GPQA-Diamond": 29.8},
        "Qwen2.5-1.5B-Instruct": {"MMLU": 60.9, "HellaSwag": 67.9, "ARC-Challenge": 54.7, "Winogrande": 65.0, "BBH": 45.1, "GSM8K": 73.2, "GPQA-Diamond": 29.8},
        "Qwen2.5-3B-Instruct":   {"MMLU": 65.6, "HellaSwag": 74.6, "ARC-Challenge": 56.5, "Winogrande": 71.1, "BBH": 56.3, "GSM8K": 86.7, "GPQA-Diamond": 30.3},
        "Qwen2.5-7B-Instruct":   {"MMLU": 74.2, "HellaSwag": 80.2, "ARC-Challenge": 63.7, "Winogrande": 75.9, "BBH": 70.4, "GSM8K": 91.6, "GPQA-Diamond": 36.4},
        "Qwen2.5-14B-Instruct":  {"MMLU": 79.7, "HellaSwag": 84.3, "ARC-Challenge": 67.3, "Winogrande": 81.0, "BBH": 78.2, "GSM8K": 94.8, "GPQA-Diamond": 45.5},
        "Qwen2.5-32B-Instruct":  {"MMLU": 83.3, "HellaSwag": 85.2, "ARC-Challenge": 70.4, "Winogrande": 82.0, "BBH": 84.5, "GSM8K": 95.9, "GPQA-Diamond": 49.5},
        "Qwen2.5-72B-Instruct":  {"MMLU": 86.1, "HellaSwag": 87.6, "ARC-Challenge": 72.4, "Winogrande": 83.9, "BBH": 86.3, "GSM8K": 95.8, "GPQA-Diamond": 49.0},
    },
    "Llama": {
        "Llama-3.2-1B-Instruct": {"MMLU": 49.3, "HellaSwag": 41.2, "ARC-Challenge": 59.4, "GSM8K": 44.4, "GPQA-Diamond": 27.2},
        "Llama-3.2-3B-Instruct": {"MMLU": 63.4, "HellaSwag": 69.8, "ARC-Challenge": 78.6, "GSM8K": 77.7, "GPQA-Diamond": 32.8},
        "Llama-3.1-8B-Instruct": {"MMLU": 69.4, "ARC-Challenge": 83.4, "GSM8K": 84.5, "GPQA-Diamond": 30.4, "Winogrande": 60.5, "TriviaQA": 77.6, "BBH": 64.2},
        "Llama-3.3-70B-Instruct": {"MMLU": 86.0, "GPQA-Diamond": 50.5, "Winogrande": 83.3, "TriviaQA": 89.8, "BBH": 81.6, "ARC-Challenge": 92.9},
    },
    "Gemma": {
        "Gemma-3-1B-it":  {"MMLU": 38.8, "HellaSwag": 62.3, "PIQA": 73.8, "ARC-Challenge": 38.4, "Winogrande": 58.2, "TriviaQA": 39.8, "BBH": 28.4, "GPQA-Diamond": 19.2},
        "Gemma-3-4B-it":  {"MMLU": 58.1, "HellaSwag": 77.2, "PIQA": 79.6, "ARC-Challenge": 56.2, "Winogrande": 64.7, "TriviaQA": 65.8, "GSM8K": 38.4, "BBH": 50.9, "GPQA-Diamond": 30.8},
        "Gemma-3-12B-it": {"MMLU": 71.9, "HellaSwag": 84.2, "PIQA": 81.8, "ARC-Challenge": 68.9, "Winogrande": 74.3, "TriviaQA": 78.2, "GSM8K": 71.0, "BBH": 72.6, "GPQA-Diamond": 40.9},
        "Gemma-3-27B-it": {"MMLU": 76.9, "HellaSwag": 85.6, "PIQA": 83.3, "ARC-Challenge": 70.6, "Winogrande": 78.8, "TriviaQA": 85.5, "GSM8K": 82.6, "BBH": 77.7, "GPQA-Diamond": 42.4},
    },
    # Epoch AI eci_benchmarks.csv at HIGH/XHIGH reasoning effort; our coherence study uses
    # reasoning_effort="none". Artificial Analysis suggests non-reasoning roughly halves the
    # composite index (gpt-5.4: 57 -> 35), so these ECI placements overstate no-reasoning
    # capability. The within-generation ordering (nano < mini < full) is robust.
    "GPT-5.4": {
        "gpt-5.4-nano": {"GPQA-Diamond": 71.3, "AIME-2025": 87.8},
        "gpt-5.4-mini": {"GPQA-Diamond": 78.1, "AIME-2025": 87.2},
        "gpt-5.4":      {"GPQA-Diamond": 91.1, "AIME-2025": 95.3, "SWE-bench verified": 76.9, "HLE": 33.0},
    },
    "GPT-OSS-20B (budget)": {
        "gpt-oss-20b-low":    {"MMLU": 80.4, "GPQA-Diamond": 56.8, "AIME-2025": 37.1},
        "gpt-oss-20b-medium": {"MMLU": 84.0, "GPQA-Diamond": 66.0, "AIME-2025": 72.1},
        "gpt-oss-20b-high":   {"MMLU": 85.3, "GPQA-Diamond": 71.5, "AIME-2025": 91.7},
    },
    "GPT-OSS-120B (budget)": {
        "gpt-oss-120b-low":    {"MMLU": 85.9, "GPQA-Diamond": 67.1, "AIME-2025": 50.4},
        "gpt-oss-120b-medium": {"MMLU": 88.0, "GPQA-Diamond": 73.1, "AIME-2025": 80.0},
        "gpt-oss-120b-high":   {"MMLU": 90.0, "GPQA-Diamond": 80.1, "AIME-2025": 92.5},
    },
}


def main():
    rows = []
    for family, models in DATA.items():
        for model, scores in models.items():
            for bench, score in scores.items():
                rows.append({"model": model, "family": family, "benchmark": bench, "score": score})
    out = REPO / "results/model_benchmarks.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "family", "benchmark", "score"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {sum(len(m) for m in DATA.values())} models)")


if __name__ == "__main__":
    main()
