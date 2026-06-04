import importlib.util
from pathlib import Path


def _load_mod():
    p = Path("scripts/upload_datasets_hf.py")
    spec = importlib.util.spec_from_file_location("upload_datasets_hf", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_yaml_to_dataset_items(tmp_path):
    m = _load_mod()
    p = tmp_path / "items_500.yaml"
    p.write_text("items:\n- a\n- b\n- c\n")
    ds = m.yaml_to_dataset(p)
    assert ds.column_names == ["item"]
    assert ds["item"] == ["a", "b", "c"]


def test_yaml_to_dataset_concepts(tmp_path):
    m = _load_mod()
    p = tmp_path / "curated_concepts.yaml"
    p.write_text("concepts:\n- x\n- y\n")
    ds = m.yaml_to_dataset(p)
    assert ds.column_names == ["item"]
    assert ds["item"] == ["x", "y"]
