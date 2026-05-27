from __future__ import annotations

import argparse
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

from sentiment_utility.characters import load_character_model, model_specs
from sentiment_utility.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from sentiment_utility.elicit import compare_pairs
from sentiment_utility.probe import (
    extract_activations,
    fit_deployable_probe,
    probe_all_layers,
    probe_score_concepts,
    save_probe,
)


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


def _load_items(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return list(data["items"])


def _setup_logging(run_dir: Path) -> logging.Logger:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(run_dir / "run_character.log"),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger(__name__)


def _plot_r2(probe_result: dict, output: Path) -> None:
    rows = [
        {"layer": int(layer), "test_r2": float(metrics["test_r2"])}
        for layer, metrics in probe_result["per_layer"].items()
    ]
    frame = pd.DataFrame(rows).sort_values("layer")
    plt.figure(figsize=(8, 4.5))
    ax = sns.lineplot(data=frame, x="layer", y="test_r2", marker="o")
    ax.axvline(int(probe_result["best_layer"]), color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test R2")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _resolve_spec(spec_name: str) -> dict:
    specs = {spec["name"]: spec for spec in model_specs()}
    if spec_name not in specs:
        names = ", ".join(specs)
        raise SystemExit(f"unknown spec {spec_name!r}; choose one of: {names}")
    return specs[spec_name]


def run_one(
    spec,
    items_train_path: str = "config/items_500.yaml",
    items_eval_path: str = "config/items_2000.yaml",
    out_root: str = "runs/character",
) -> Path:
    spec = dict(spec)
    run_dir = Path(out_root) / spec["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    log = _setup_logging(run_dir)

    train_items = _load_items(items_train_path)
    eval_items = _load_items(items_eval_path)
    commit = _git_commit()
    seed = 0
    batch_size = 64
    log.info("commit=%s", commit)
    log.info("loading character model %s", spec["name"])
    tok, model = load_character_model(spec)

    log.info("efficient elicitation over %d train items", len(train_items))

    def oracle(pairs):
        return compare_pairs(tok, model, train_items, pairs, batch_size=batch_size)

    order, edges = rank_by_quicksort(len(train_items), oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    fit = fit_thurstone_sparse(edges, len(train_items), seed=seed)
    mu = np.asarray(fit["mu"], dtype=np.float64)

    log.info("extracting train activations")
    hidden = extract_activations(tok, model, train_items, batch_size=16)
    log.info("probing %d layers", len(hidden))
    probe_result = probe_all_layers(hidden, mu, seed=seed)
    best_layer = int(probe_result["best_layer"])
    best_r2 = float(probe_result["best_r2"])
    deployable_probe = fit_deployable_probe(hidden[best_layer], mu)
    deployable_probe["best_layer"] = best_layer
    save_probe(run_dir / "probe.json", deployable_probe)

    sample_items = train_items[: min(32, len(train_items))]
    use_kv_cache = True
    max_abs_diff = float("nan")
    if sample_items:
        log.info("running KV-cache equivalence gate on %d items", len(sample_items))
        cached = probe_score_concepts(
            tok, model, sample_items, best_layer, deployable_probe, batch_size=16, use_kv_cache=True
        )
        uncached = probe_score_concepts(
            tok,
            model,
            sample_items,
            best_layer,
            deployable_probe,
            batch_size=16,
            use_kv_cache=False,
        )
        max_abs_diff = float(np.max(np.abs(cached - uncached)))
        try:
            assert max_abs_diff < 1e-2
        except AssertionError:
            log.warning(
                "KV-cache equivalence max_abs_diff=%g exceeds 1e-2; using uncached eval",
                max_abs_diff,
            )
            use_kv_cache = False
        else:
            log.info("KV-cache equivalence max_abs_diff=%g", max_abs_diff)

    log.info("probe-scoring %d eval items (use_kv_cache=%s)", len(eval_items), use_kv_cache)
    scores = probe_score_concepts(
        tok,
        model,
        eval_items,
        best_layer,
        deployable_probe,
        batch_size=16,
        use_kv_cache=use_kv_cache,
    )

    elicited_mu = {item: float(value) for item, value in zip(train_items, mu)}
    score_map = {item: float(value) for item, value in zip(eval_items, scores)}
    metrics = {
        "best_layer": best_layer,
        "best_r2": best_r2,
        "per_layer_r2": {
            str(layer): float(layer_metrics["test_r2"])
            for layer, layer_metrics in probe_result["per_layer"].items()
        },
        "mu_mean": float(mu.mean()) if len(mu) else float("nan"),
        "mu_std": float(mu.std()) if len(mu) else float("nan"),
        "mu_min": float(mu.min()) if len(mu) else float("nan"),
        "mu_max": float(mu.max()) if len(mu) else float("nan"),
        "comparison_count": int(fit["comparison_count"]),
        "unique_pairs": int(fit.get("unique_pairs", fit["comparison_count"])),
        "kv_equivalence_max_abs_diff": max_abs_diff,
        "eval_use_kv_cache": bool(use_kv_cache),
        "train_n": int(len(train_items)),
        "eval_n": int(len(eval_items)),
    }
    config = {
        "commit": commit,
        "items_train_path": items_train_path,
        "items_eval_path": items_eval_path,
        "spec": spec,
    }

    (run_dir / "config.json").write_text(json.dumps(_jsonable(config), indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2) + "\n")
    (run_dir / "elicited_mu.json").write_text(json.dumps(elicited_mu, indent=2) + "\n")
    (run_dir / "scores.json").write_text(json.dumps(score_map, indent=2) + "\n")
    _plot_r2(probe_result, run_dir / "probe_r2_vs_layer.pdf")

    log.info(
        "done -> %s best_layer=%d best_r2=%.4f comparison_count=%d",
        run_dir,
        best_layer,
        best_r2,
        metrics["comparison_count"],
    )
    print(json.dumps(_jsonable(metrics), indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one character sentiment probe pipeline.")
    parser.add_argument("--spec-name", required=True, help="Character spec name, e.g. base or loving.")
    parser.add_argument("--items-train-path", default="config/items_500.yaml")
    parser.add_argument("--items-eval-path", default="config/items_2000.yaml")
    parser.add_argument("--out-root", default="runs/character")
    args = parser.parse_args()
    run_one(
        _resolve_spec(args.spec_name),
        items_train_path=args.items_train_path,
        items_eval_path=args.items_eval_path,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
