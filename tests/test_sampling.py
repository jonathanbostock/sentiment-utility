import numpy as np
from sentiment_utility.questions import Question
from sentiment_utility.sampling import elo_active_sample
from fakes import FakeOracle


def test_elo_sampler_covers_items_and_recovers_order():
    n = 20
    scores = np.linspace(-3, 3, n)
    rng = np.random.default_rng(0)
    scores = scores[rng.permutation(n)]
    q = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})]
    edges = elo_active_sample(n, FakeOracle(scores), q, R=5, m=6, floor=0.15, K=32, seed=0)
    seen = {e.i for e in edges} | {e.j for e in edges}
    assert seen == set(range(n))                 # every item compared
    from sentiment_utility.fit import fit_caseV_mle
    rows = [{"i": e.i, "j": e.j, "p_util": e.p_util, "mode": "logprob"} for e in edges]
    mu = fit_caseV_mle(rows, n=n, steps=1500, seed=0)["mu"]
    assert np.corrcoef(mu, scores)[0, 1] > 0.9
