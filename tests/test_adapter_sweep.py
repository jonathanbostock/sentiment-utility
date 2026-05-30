import importlib.util
from pathlib import Path


def _mod():
    spec = importlib.util.spec_from_file_location("ras", "scripts/run_adapter_sweep.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_adapter_list(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("# comment\norg/a\n\norg/b\n  org/c  \n")
    assert _mod().load_adapter_list(p) == ["org/a", "org/b", "org/c"]


def test_select_shard_partitions_disjointly_and_covers_all():
    m = _mod()
    items = [f"m{i}" for i in range(10)]
    shards = [m.select_shard(items, f"{k}/3") for k in (1, 2, 3)]
    # disjoint
    flat = [x for s in shards for x in s]
    assert sorted(flat) == sorted(items)
    assert len(flat) == len(set(flat))
    # roughly balanced
    assert all(3 <= len(s) <= 4 for s in shards)


def test_select_shard_none_returns_all():
    m = _mod()
    items = ["a", "b"]
    assert m.select_shard(items, None) == items


def test_adapter_name_basename():
    assert _mod()._adapter_name("auditing-agents/llama_70b_x_animal_welfare") == "llama_70b_x_animal_welfare"
