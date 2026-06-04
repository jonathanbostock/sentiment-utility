from __future__ import annotations

import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from question_consistency.efficient import fit_thurstone_sparse, rank_by_quicksort, spacing_pass
from question_consistency.elicit import compare_pairs, elicit_logprobs, load_model
from question_consistency.io_utils import load_items as _load_items
from question_consistency.preferences import combine_orderings
from question_consistency.thurstone import fit_thurstone


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _load_run_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


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


def main(
    items_path: str = "config/datasets/items_500.yaml",
    run_path: str = "config/run/run.yaml",
    subset_n: int = 60,
    seed: int = 0,
) -> None:
    all_items = _load_items(Path(items_path))
    rng = np.random.default_rng(seed)
    subset_idx = sorted(rng.choice(len(all_items), size=min(subset_n, len(all_items)), replace=False))
    items = [all_items[i] for i in subset_idx]
    cfg = _load_run_config(Path(run_path))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(run_dir / "validate_method.log"), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)
    commit = _git_commit()
    log.info("commit=%s", commit)
    log.info("loading model %s", cfg["model_id"])
    tok, model = load_model(cfg["model_id"], cfg.get("dtype", "bfloat16"))

    batch_size = int(cfg.get("batch_size", 64))
    fit_cfg = cfg.get("fit", {})
    n = len(items)

    log.info("dense elicitation: %d ordered comparisons", n * (n - 1))
    ordered = elicit_logprobs(tok, model, items, batch_size=batch_size)
    pref = combine_orderings(n, ordered)
    dense = fit_thurstone(
        pref,
        lr=fit_cfg.get("lr", 0.05),
        steps=fit_cfg.get("steps", 2000),
        test_frac=fit_cfg.get("test_frac", 0.2),
        l2_sigma=fit_cfg.get("l2_sigma", 0.01),
        seed=seed,
    )

    log.info("efficient elicitation")

    def oracle(pairs):
        return compare_pairs(tok, model, items, pairs, batch_size=batch_size)

    order, edges = rank_by_quicksort(n, oracle, seed=seed)
    edges = edges + spacing_pass(order, oracle)
    sparse = fit_thurstone_sparse(
        edges,
        n,
        lr=fit_cfg.get("lr", 0.05),
        steps=fit_cfg.get("steps", 2000),
        test_frac=fit_cfg.get("test_frac", 0.2),
        l2_sigma=fit_cfg.get("l2_sigma", 0.01),
        seed=seed,
    )

    dense_mu = dense["mu"]
    sparse_mu = sparse["mu"]
    dense_count = n * (n - 1)
    efficient_count = int(sparse["comparison_count"])
    metrics = {
        "spearman_rho": float(spearmanr(dense_mu, sparse_mu).statistic),
        "mae": float(np.mean(np.abs(dense_mu - sparse_mu))),
        "dense_comparison_count": dense_count,
        "efficient_comparison_count": efficient_count,
        "comparison_ratio": float(efficient_count / dense_count) if dense_count else float("nan"),
        "dense_test_accuracy": float(dense["test_accuracy"]),
        "sparse_test_accuracy": float(sparse["test_accuracy"]),
    }
    log.info("metrics=%s", json.dumps(metrics))

    result = {
        "commit": commit,
        "config": {
            "items_path": items_path,
            "run_path": run_path,
            "subset_n": len(items),
            "seed": seed,
            "subset_indices": subset_idx,
        },
        "items": items,
        "metrics": metrics,
        "efficient_order": order,
        "efficient_edges": edges,
        "dense_mu": dense_mu,
        "sparse_mu": sparse_mu,
        "dense_sigma": dense["sigma"],
        "sparse_sigma": sparse["sigma"],
    }
    (run_dir / "validation_results.json").write_text(json.dumps(_jsonable(result), indent=2))
    print(json.dumps(metrics, indent=2))
    log.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
