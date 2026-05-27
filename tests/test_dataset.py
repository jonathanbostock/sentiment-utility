from sentiment_utility.dataset import build_pool_sample


def _sources():
    return {
        "curated": [(f"c{i}", None) for i in range(10)],
        "things": [(f"t{i}", None) for i in range(10)],
        "warriner": [(f"w{i}", float(i)) for i in range(10)],
    }


def test_exact_n_and_dedupe():
    items, meta = build_pool_sample(
        _sources(), {"curated": 5, "things": 5, "warriner": 5}, n=12, seed=0
    )
    assert len(items) == 12
    assert len(set(items)) == 12
    for it in items:
        assert meta[it]["source"] in {"curated", "things", "warriner"}


def test_dedupe_across_sources_first_wins():
    src = {"curated": [("apple", None)], "warriner": [("Apple", 5.0)]}
    items, meta = build_pool_sample(src, {"curated": 1, "warriner": 1}, n=2, seed=0)
    assert len(items) == 1
    assert meta[items[0]]["source"] == "curated"


def test_determinism():
    a = build_pool_sample(
        _sources(), {"curated": 5, "things": 5, "warriner": 5}, n=12, seed=1
    )[0]
    b = build_pool_sample(
        _sources(), {"curated": 5, "things": 5, "warriner": 5}, n=12, seed=1
    )[0]
    assert a == b


def test_topup_when_quota_exceeds_pool():
    items, _ = build_pool_sample(
        _sources(), {"curated": 100, "things": 0, "warriner": 0}, n=25, seed=0
    )
    assert len(items) == 25
