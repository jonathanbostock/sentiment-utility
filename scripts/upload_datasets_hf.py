"""Publish configured YAML datasets to a Hugging Face dataset repo.

Each config/datasets/<name>.yaml file may contain either an ``items:`` or
``concepts:`` list. The uploader converts that list to a single-column Dataset
named ``item`` and pushes it as the ``train`` split for config ``<name>``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from question_consistency.datasets import HF_DATASET_REPO, REGISTRY, read_yaml_items  # noqa: E402


def yaml_to_dataset(path):
    from datasets import Dataset

    return Dataset.from_dict({"item": read_yaml_items(path)})


def _dataset_names(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(",") if name.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=HF_DATASET_REPO)
    parser.add_argument(
        "--datasets",
        default=",".join(sorted(REGISTRY)),
        help="comma-separated config names with config/datasets/<name>.yaml files",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    HfApi().create_repo(
        args.repo, repo_type="dataset", private=args.private, exist_ok=True
    )
    for name in _dataset_names(args.datasets):
        path = REPO / "config" / "datasets" / f"{name}.yaml"
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        ds = yaml_to_dataset(path)
        ds.push_to_hub(args.repo, config_name=name, split="train")
        print(f"PUSHED {name}: {len(ds)} rows")


if __name__ == "__main__":
    main()
