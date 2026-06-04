from __future__ import annotations
import yaml


def load_items(path: str) -> list[str]:
    with open(path) as f:
        data = yaml.safe_load(f)
    items = data["items"]
    if len(items) != len(set(items)):
        raise ValueError("items must be unique")
    return items


def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
