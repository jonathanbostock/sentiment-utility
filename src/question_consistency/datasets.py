from __future__ import annotations

from pathlib import Path

import yaml

HF_DATASET_REPO = "arcadia-impact/question-consistency-datasets"
REGISTRY = {"items_500", "items_2000", "curated_concepts"}  # hosted on HF as parquet configs


def read_yaml_items(path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("items", "concepts"):
        if key in data:
            return list(data[key])
    raise ValueError(f"{path}: YAML has neither 'items' nor 'concepts' key")


def _from_hf_config(name: str) -> list[str]:
    import datasets

    return list(datasets.load_dataset(HF_DATASET_REPO, name=name, split="train")["item"])


def _from_hf_dataset(spec: str) -> list[str]:
    # hf-dataset:<repo>:<split>:<column>
    rest = spec[len("hf-dataset:") :]
    parts = rest.split(":")
    if len(parts) != 3:
        raise ValueError("hf-dataset ref must be hf-dataset:<repo>:<split>:<column>")
    repo, split, column = parts
    import datasets

    return list(datasets.load_dataset(repo, split=split)[column])


def _from_hf_file(spec: str) -> list[str]:
    # hf://<owner>/<repo>/<path-in-repo>
    import huggingface_hub

    segs = spec[len("hf://") :].split("/")
    if len(segs) < 3:
        raise ValueError("hf:// ref must be hf://<owner>/<repo>/<path>")
    repo_id = "/".join(segs[:2])
    local = huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename="/".join(segs[2:]), repo_type="dataset"
    )
    return read_yaml_items(local)


def resolve_items(ref) -> list[str]:
    """Resolve a dataset reference to a list of item strings.

    Forms: local YAML path | bare known name | hf://<owner>/<repo>/<file> |
    hf-dataset:<repo>:<split>:<column>. A *missing* local path whose basename is a
    registered name (e.g. config/datasets/items_2000.yaml) auto-pulls from HF.
    """
    ref = str(ref)
    if ref.startswith("hf-dataset:"):
        return _from_hf_dataset(ref)
    if ref.startswith("hf://"):
        return _from_hf_file(ref)
    p = Path(ref)
    if p.exists():
        return read_yaml_items(p)
    name = p.name[:-5] if p.name.endswith(".yaml") else p.name
    if ref in REGISTRY:
        return _from_hf_config(ref)
    if name in REGISTRY:
        return _from_hf_config(name)
    raise FileNotFoundError(
        f"cannot resolve dataset ref {ref!r}: not a local file, not a known name "
        f"{sorted(REGISTRY)}, and not an hf://... or hf-dataset:... reference"
    )
