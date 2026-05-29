import math

import numpy as np
from sentiment_utility.questions import Question
from sentiment_utility.oracle import Comparison, p_util_from_pick


def test_p_util_slot_i_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="i", question=q), 0.8)


def test_p_util_slot_j_positive():
    q = Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.2)


def test_p_util_slot_j_negative_valence():
    q = Question(id="neg", template="{item_A}{item_B}", valence=-1, answers={"A": ["A"], "B": ["B"]})
    assert np.isclose(p_util_from_pick(0.8, slot_a="j", question=q), 0.8)


def test_p_a_from_logprobs():
    from sentiment_utility.oracle import p_a_from_logprobs
    q = Question(id="pos", template="x", valence=1, answers={"A": ["A"], "B": ["B"]})
    tops = [{"token": "A", "lp": math.log(0.75)}, {"token": "B", "lp": math.log(0.25)}]
    assert abs(p_a_from_logprobs(tops, q) - 0.75) < 1e-6


def test_p_a_from_picks_jeffreys():
    from sentiment_utility.oracle import p_a_from_picks
    p, a, b = p_a_from_picks(["A", "A", "B"])
    assert abs(p - 0.625) < 1e-6 and a == 2 and b == 1
