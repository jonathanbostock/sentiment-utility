from __future__ import annotations

import numpy as np

from .oracle import Comparison


def _elo_expected(ri, rj, scale=400.0):
    return 1.0 / (1.0 + 10 ** ((rj - ri) / scale))


def _make_comparison(i, j, items, questions, rng, phase, rnd):
    q = questions[rng.integers(len(questions))]
    slot_a = "i" if rng.random() < 0.5 else "j"
    return Comparison(i=i, j=j, item_i=items[i] if items else str(i),
                      item_j=items[j] if items else str(j),
                      question=q, slot_a=slot_a, phase=phase, round=rnd)


def elo_active_sample(n, oracle, questions, R=5, m=5, floor=0.15, K=32, seed=0,
                      items=None):
    rng = np.random.default_rng(seed)
    ratings = np.zeros(n)
    all_obs = []
    seen = set()

    def submit(pairs, rnd):
        comps = [_make_comparison(i, j, items, questions, rng, "elo", rnd) for i, j in pairs]
        obs = oracle.compare(comps)
        for o in obs:
            all_obs.append(o)
            exp_i = _elo_expected(ratings[o.i], ratings[o.j])
            ratings[o.i] += K * (o.p_util - exp_i)
            ratings[o.j] += K * ((1 - o.p_util) - (1 - exp_i))
        return obs

    for rnd in range(1, R + 1):
        pairs = []
        for i in range(n):
            if rnd == 1:
                partners = rng.choice([x for x in range(n) if x != i], size=min(m, n - 1),
                                      replace=False)
            else:
                d = (ratings[i] - ratings) / 400.0
                p = 1.0 / (1.0 + 10 ** (-d))
                info = p * (1 - p)
                info[i] = 0.0
                w = (1 - floor) * info + floor * (np.arange(n) != i)
                w = w / w.sum()
                partners = rng.choice(n, size=min(m, n - 1), replace=False, p=w)
            for jj in partners:
                key = (min(i, int(jj)), max(i, int(jj)))
                pairs.append((i, int(jj)))
                seen.add(key)
        submit(pairs, rnd)

    return all_obs
