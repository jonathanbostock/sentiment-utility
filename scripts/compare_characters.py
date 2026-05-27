from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sentiment_utility.characters import model_specs
from sentiment_utility.deltas import score_deltas


def _load_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {str(item): float(score) for item, score in data.items()}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _plot_scatter(base: np.ndarray, char: np.ndarray, name: str, output: Path) -> None:
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(x=base, y=char, s=18, edgecolor=None)
    low = float(min(base.min(), char.min()))
    high = float(max(base.max(), char.max()))
    ax.plot([low, high], [low, high], color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Base score")
    ax.set_ylabel(f"{name} score")
    ax.set_title(f"Base vs {name}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_delta_bars(delta_result: dict, name: str, output: Path) -> None:
    more_negative = list(reversed(delta_result["more_negative"]))
    rows = more_negative + delta_result["more_positive"]
    labels = [row["item"] for row in rows]
    values = [float(row["delta"]) for row in rows]
    colors = ["#5B7DB1" if value < 0 else "#C75D4D" for value in values]
    height = max(6, 0.28 * len(rows))
    plt.figure(figsize=(9, height))
    ax = sns.barplot(x=values, y=labels, palette=colors, hue=labels, dodge=False, legend=False)
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Delta z-score")
    ax.set_ylabel("")
    ax.set_title(f"Top sentiment deltas: {name}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare character probe scores to base scores.")
    parser.add_argument("--out-root", default="runs/character")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    deltas_dir = out_root / "deltas"
    deltas_dir.mkdir(parents=True, exist_ok=True)

    base_scores = _load_scores(out_root / "base" / "scores.json")
    summary_rows = []
    for spec in model_specs():
        name = spec["name"]
        if name == "base":
            continue
        scores_path = out_root / name / "scores.json"
        if not scores_path.exists():
            continue
        char_scores = _load_scores(scores_path)
        items = [item for item in base_scores if item in char_scores]
        if not items:
            continue
        base = np.asarray([base_scores[item] for item in items], dtype=np.float64)
        char = np.asarray([char_scores[item] for item in items], dtype=np.float64)
        result = score_deltas(items, base, char, top_k=args.top_k)
        result["n"] = len(items)
        _write_json(deltas_dir / f"{name}.json", result)
        _plot_scatter(base, char, name, deltas_dir / f"{name}_base_vs_char.pdf")
        _plot_delta_bars(result, name, deltas_dir / f"{name}_top_bottom_deltas.pdf")
        summary_rows.append(
            {
                "character": name,
                "mean_abs_delta": result["mean_abs_delta"],
                "pearson_r": result["pearson_r"],
                "n": len(items),
            }
        )

    with (deltas_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["character", "mean_abs_delta", "pearson_r", "n"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {deltas_dir} ({len(summary_rows)} characters)")


if __name__ == "__main__":
    main()
