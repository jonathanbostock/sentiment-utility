import json
import numpy as np
import importlib.util
from sentiment_utility.questions import Question
from sentiment_utility.sampling import plan_from_prior_mu


def _load_run_elicitation():
    spec = importlib.util.spec_from_file_location("run_elicitation", "scripts/run_elicitation.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_plan_size_and_informativeness():
    n = 50; prior_mu = np.linspace(1, 0, n)
    comps = plan_from_prior_mu(prior_mu, m=5, fresh_frac=0.1, seed=0)
    assert 0.8*n*5 <= len(comps) <= 1.2*n*5
    order = list(np.argsort(-prior_mu))
    rank = {it: r for r, it in enumerate(order)}
    dists = [abs(rank[a]-rank[b]) for a, b in comps]
    assert np.median(dists) < n/4   # biased toward near-rank neighbours


def test_warm_start_far_cheaper_and_recovers_order(tmp_path):
    from fakes import FakeOracle
    class CountingOracle(FakeOracle):
        def __init__(self, scores): super().__init__(scores); self.n=0
        def compare(self, comps): self.n += len(comps); return super().compare(comps)
    mod = _load_run_elicitation()
    n = 20; rng = np.random.default_rng(2); scores = rng.normal(size=n)
    items = [f"item{k}" for k in range(n)]
    questions = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A":["A"],"B":["B"]})]
    elo_cfg = dict(R=5, m=5, floor=0.15, K=32)
    phase_cfg = dict(n_reverse=10, n_triads=10, n_cross=0)
    # full cold run
    full = CountingOracle(scores); out1 = tmp_path/"full"
    mod.run_elicitation(full, items, questions, out1, elo_cfg=elo_cfg, phase_cfg=phase_cfg, seed=0)
    mu_full = json.loads((out1/"mu.json").read_text())
    prior = np.array([mu_full[it] for it in items])
    # warm run seeded from full mu
    warm = CountingOracle(scores); out2 = tmp_path/"warm"
    mod.run_elicitation(warm, items, questions, out2, elo_cfg=elo_cfg, phase_cfg=phase_cfg,
                        seed=0, prior_mu=prior, fit_steps=400, warm_cfg=dict(m=5, fresh_frac=0.2))
    assert warm.n < 0.5 * full.n          # far fewer model queries
    mu_warm = json.loads((out2/"mu.json").read_text())
    order_full = np.argsort(-prior)
    order_warm = np.argsort(-np.array([mu_warm[it] for it in items]))
    # Spearman rank correlation of the two recovered orders
    from scipy.stats import spearmanr
    rho = spearmanr(np.argsort(order_full), np.argsort(order_warm)).correlation
    assert rho > 0.9
