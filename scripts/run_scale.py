from __future__ import annotations

import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sentiment_utility.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from sentiment_utility.elicit import compare_pairs, load_model
from sentiment_utility.metrics import completeness, cyclic_triad_fraction, expected_cycle_probability
from sentiment_utility.probe import extract_activations, probe_all_layers
from sentiment_utility.thurstone import predict_pref_matrix


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_items_with_meta(path: Path) -> tuple[list[str], dict]:
    data = yaml.safe_load(path.read_text()) or {}
    return list(data["items"]), dict(data.get("meta", {}))


def _load_run_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _plot_sentiment_extremes(items, mu, sigma, output: Path) -> None:
    n = len(items)
    order = np.argsort(mu)
    bottom = list(order[: min(25, n)])
    top = list(order[max(0, n - 25) :])
    selected = bottom + [idx for idx in top if idx not in set(bottom)]
    frame = pd.DataFrame(
        {
            "item": [items[i] for i in selected],
            "mu": [float(mu[i]) for i in selected],
            "sigma": [float(sigma[i]) for i in selected],
            "group": ["bottom"] * len(bottom) + ["top"] * (len(selected) - len(bottom)),
        }
    )
    frame["item"] = pd.Categorical(frame["item"], categories=frame["item"], ordered=True)

    height = max(8, 0.24 * len(frame))
    plt.figure(figsize=(9, height))
    ax = sns.barplot(data=frame, y="item", x="mu", hue="group", dodge=False, palette="Set2")
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Thurstonian mu")
    ax.set_ylabel("")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _plot_probe_metric(probe_result: dict, key: str, ylabel: str, output: Path) -> None:
    rows = [
        {"layer": int(layer), key: metrics[key]}
        for layer, metrics in probe_result["per_layer"].items()
    ]
    frame = pd.DataFrame(rows).sort_values("layer")
    plt.figure(figsize=(8, 4.5))
    ax = sns.lineplot(data=frame, x="layer", y=key, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main(items_path: str = "config/datasets/items_500.yaml", run_path: str = "config/run/run.yaml") -> None:
    items, meta = _load_items_with_meta(Path(items_path))
    cfg = _load_run_config(Path(run_path))
    seed = int(cfg.get("seed", 0))
    batch_size = int(cfg.get("batch_size", 64))
    fit_cfg = cfg.get("fit", {})

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(run_dir / "run_scale.log"), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)
    commit = _git_commit()
    log.info("commit=%s", commit)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "items_path": items_path,
                "run_path": run_path,
                "items": items,
                "meta": meta,
                "run_config": cfg,
            },
            indent=2,
        )
    )

    log.info("loading model %s", cfg["model_id"])
    tok, model = load_model(cfg["model_id"], cfg.get("dtype", "bfloat16"))

    n = len(items)
    dense_count = n * (n - 1)
    log.info("efficient elicitation over %d items", n)

    def oracle(pairs):
        return compare_pairs(tok, model, items, pairs, batch_size=batch_size)

    order, edges = rank_by_quicksort(n, oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(
        edges,
        n,
        lr=fit_cfg.get("lr", 0.05),
        steps=fit_cfg.get("steps", 2000),
        test_frac=fit_cfg.get("test_frac", 0.2),
        l2_sigma=fit_cfg.get("l2_sigma", 0.01),
        seed=seed,
    )
    mu = fit["mu"]
    sigma = fit["sigma"]
    pref = predict_pref_matrix(mu, sigma)
    np.save(run_dir / "utility_mu.npy", mu)
    np.save(run_dir / "utility_sigma.npy", sigma)
    np.save(run_dir / "pred_matrix.npy", pref)
    (run_dir / "edges.json").write_text(json.dumps(_jsonable(edges), indent=2))

    metrics = {
        "utility_test_accuracy": float(fit["test_accuracy"]),
        "accuracy_is_heldout": bool(fit["accuracy_is_heldout"]),
        "cyclic_triad_fraction": cyclic_triad_fraction(pref),
        "expected_cycle_probability": expected_cycle_probability(pref),
        "completeness": completeness(pref),
        "comparison_count": int(fit["comparison_count"]),
        "dense_comparison_count": dense_count,
        "comparison_ratio": float(fit["comparison_count"] / dense_count) if dense_count else float("nan"),
    }
    if "unique_pairs" in fit:
        metrics["unique_pairs"] = int(fit["unique_pairs"])

    ranking_order = np.argsort(-mu)
    ranking = [
        {
            "rank": int(rank + 1),
            "item": items[i],
            "mu": float(mu[i]),
            "sigma": float(sigma[i]),
            "source": meta.get(items[i], {}).get("source"),
            "human_valence": meta.get(items[i], {}).get("human_valence"),
        }
        for rank, i in enumerate(ranking_order)
    ]

    log.info("extracting activations")
    hidden = extract_activations(tok, model, items, batch_size=16)
    log.info("probing mu across %d layers", len(hidden))
    probe_mu = probe_all_layers(hidden, mu, seed=seed)
    probe_results = {"mu": probe_mu}

    valence_idx = [
        idx
        for idx, item in enumerate(items)
        if meta.get(item, {}).get("human_valence") is not None
    ]
    if len(valence_idx) >= 50:
        human_valence = np.array(
            [float(meta[items[idx]]["human_valence"]) for idx in valence_idx], dtype=float
        )
        hidden_valence = {layer: X[valence_idx] for layer, X in hidden.items()}
        probe_results["human_valence"] = probe_all_layers(hidden_valence, human_valence, seed=seed)
        metrics["mu_human_valence_pearson"] = float(np.corrcoef(mu[valence_idx], human_valence)[0, 1])
        metrics["human_valence_n"] = int(len(valence_idx))
    else:
        metrics["mu_human_valence_pearson"] = float("nan")
        metrics["human_valence_n"] = int(len(valence_idx))

    log.info("writing plots")
    _plot_sentiment_extremes(items, mu, sigma, run_dir / "sentiment_top_bottom_25.pdf")
    _plot_probe_metric(probe_mu, "test_r2", "Test R2", run_dir / "probe_r2_vs_layer.pdf")
    _plot_probe_metric(
        probe_mu,
        "pairwise_accuracy",
        "Pairwise Accuracy",
        run_dir / "probe_pairwise_acc_vs_layer.pdf",
    )

    results = {
        "commit": commit,
        "metrics": metrics,
        "probe_results": probe_results,
        "ranking": ranking,
        "order": [int(i) for i in order],
    }
    (run_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    print(json.dumps(_jsonable(metrics), indent=2))
    log.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
