import numpy as np


class FakeOracle:
    """p_util(item_i > item_j) = logistic(score_i - score_j) from a hidden ground truth.
    Deterministic; drives sampling/integration tests without a model or network."""
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def compare(self, comparisons):
        from sentiment_utility.oracle import EdgeObservation
        obs = []
        for c in comparisons:
            d = self.scores[c.i] - self.scores[c.j]
            p = 1.0 / (1.0 + np.exp(-d))
            obs.append(EdgeObservation(i=c.i, j=c.j, p_util=float(p), mode="logprob",
                                       question_id=c.question.id, valence=c.question.valence,
                                       slot_a=c.slot_a, phase=c.phase, round=c.round))
        return obs
