import json
import numpy as np
from sentiment_utility.questions import Question


def test_end_to_end_with_fake_oracle(tmp_path):
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location("run_elicitation", "scripts/run_elicitation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from fakes import FakeOracle
    n = 15
    rng = np.random.default_rng(1)
    scores = rng.normal(size=n)
    items = [f"item{k}" for k in range(n)]
    questions = [Question(id="pos", template="{item_A}{item_B}", valence=1,
                          answers={"A": ["A"], "B": ["B"]})]
    out = tmp_path / "run"
    mod.run_elicitation(FakeOracle(scores), items, questions, out,
                        elo_cfg=dict(R=5, m=6, floor=0.15, K=32),
                        phase_cfg=dict(n_reverse=0, n_triads=10, n_cross=0),
                        seed=0)
    assert (out / "edges.jsonl").exists()
    panel = json.loads((out / "panel.json").read_text())
    assert 0.0 <= panel["decisiveness"]["point"] <= 1.0
    mu = json.loads((out / "mu.json").read_text())
    assert len(mu) == n
