from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sentiment_utility.characters import model_specs

from run_character import run_one


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all character sentiment pipelines.")
    parser.add_argument("--items-train-path", default="config/datasets/items_500.yaml")
    parser.add_argument("--items-eval-path", default="config/datasets/items_2000.yaml")
    parser.add_argument("--out-root", default="runs/character")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    for spec in model_specs():
        scores_path = Path(args.out_root) / spec["name"] / "scores.json"
        if scores_path.exists():
            log.info("skipping %s; found %s", spec["name"], scores_path)
            continue
        log.info("running %s", spec["name"])
        try:
            run_one(
                spec,
                items_train_path=args.items_train_path,
                items_eval_path=args.items_eval_path,
                out_root=args.out_root,
            )
        except Exception as exc:  # e.g. access-restricted adapter (misalignment)
            logging.getLogger(__name__).warning(
                "skipping %s due to error: %s", spec["name"], exc
            )


if __name__ == "__main__":
    main()
