from __future__ import annotations

import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import load_items, load_run_config
from .elicit import elicit_logprobs, load_model, validate_generation
from .metrics import completeness, cyclic_triad_fraction, expected_cycle_probability
from .plots import plot_preference_heatmap, plot_sentiment_ranking, plot_validation_scatter
from .preferences import combine_orderings
from .thurstone import fit_thurstone


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main(items_path: str = "config/items.yaml", run_path: str = "config/run.yaml") -> None:
    items = load_items(items_path)
    cfg = load_run_config(run_path)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    commit = _git_commit()
    log.info("commit=%s", commit)
    (run_dir / "config.json").write_text(
        json.dumps({"commit": commit, "items": items, "run_config": cfg}, indent=2)
    )

    log.info("loading model %s", cfg["model_id"])
    tok, model = load_model(cfg["model_id"], cfg["dtype"])

    log.info("eliciting %d ordered pairs via logprobs", len(items) * (len(items) - 1))
    ordered = elicit_logprobs(tok, model, items, batch_size=cfg["batch_size"])
    pref = combine_orderings(len(items), ordered)
    np.save(run_dir / "pref_matrix.npy", pref)
    with (run_dir / "ordered_probs.json").open("w") as f:
        json.dump({f"{i}_{j}": v for (i, j), v in ordered.items()}, f, indent=2)

    log.info("fitting Thurstonian model")
    fit = fit_thurstone(
        pref,
        lr=cfg["fit"]["lr"],
        steps=cfg["fit"]["steps"],
        test_frac=cfg["fit"]["test_frac"],
        l2_sigma=cfg["fit"]["l2_sigma"],
        seed=cfg["seed"],
    )
    mu, sigma = fit["mu"], fit["sigma"]
    np.save(run_dir / "utility_mu.npy", mu)
    np.save(run_dir / "utility_sigma.npy", sigma)
    np.save(run_dir / "pred_matrix.npy", fit["pred_matrix"])

    metrics = {
        "utility_test_accuracy": fit["test_accuracy"],
        "cyclic_triad_fraction": cyclic_triad_fraction(pref),
        "expected_cycle_probability": expected_cycle_probability(pref),
        "completeness": completeness(pref),
    }
    log.info("metrics=%s", json.dumps(metrics))

    log.info("running generation validation")
    val = validate_generation(
        tok,
        model,
        items,
        n_pairs=cfg["validation"]["n_pairs"],
        n_samples=cfg["validation"]["n_samples"],
        temperature=cfg["validation"]["temperature"],
        max_new_tokens=cfg["validation"]["max_new_tokens"],
        seed=cfg["seed"],
    )

    for row in val["pairs"]:
        row["logprob_p_a"] = ordered[(row["i"], row["j"])]
    lp = [row["logprob_p_a"] for row in val["pairs"]]
    gp = [row["gen_p_a"] for row in val["pairs"]]
    valid_pairs = [(a, b) for a, b in zip(lp, gp) if b is not None]
    if len(valid_pairs) > 1:
        a_arr = np.array([pair[0] for pair in valid_pairs])
        b_arr = np.array([pair[1] for pair in valid_pairs])
        metrics["validation_pearson_r"] = float(np.corrcoef(a_arr, b_arr)[0, 1])
        metrics["validation_agreement"] = float(np.mean((a_arr > 0.5) == (b_arr > 0.5)))
    else:
        # keep a stable results schema even when validation is too sparse
        metrics["validation_pearson_r"] = float("nan")
        metrics["validation_agreement"] = float("nan")
    metrics["validation_n_valid_pairs"] = len(valid_pairs)
    metrics["validation_malformed_rate"] = float(
        np.mean([1 - row["valid"] / row["n_samples"] for row in val["pairs"]])
    )

    order = np.argsort(-mu)
    ranking = [
        {"item": items[i], "mu": float(mu[i]), "sigma": float(sigma[i])}
        for i in order
    ]

    with (run_dir / "results.json").open("w") as f:
        json.dump({"metrics": metrics, "ranking": ranking, "validation": val["pairs"]}, f, indent=2)

    plot_sentiment_ranking(items, mu, sigma, run_dir / "sentiment_ranking.pdf")
    plot_preference_heatmap(items, pref, mu, run_dir / "preference_heatmap.pdf")
    plot_validation_scatter(lp, gp, run_dir / "validation_scatter.pdf")
    log.info("done -> %s", run_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
