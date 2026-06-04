from __future__ import annotations
import yaml


def load_items(path: str) -> list[str]:
    from .datasets import resolve_items

    return resolve_items(path)


def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
