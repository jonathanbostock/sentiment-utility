from question_consistency.data import load_items, load_run_config


def test_load_items_default():
    items = load_items("config/datasets/items.yaml")
    assert len(items) == 25
    assert "spaghetti" in items
    assert len(set(items)) == 25  # unique


def test_load_run_config():
    cfg = load_run_config("config/run/run.yaml")
    assert cfg["model_id"] == "google/gemma-3-12b-it"
    assert cfg["fit"]["test_frac"] == 0.2
