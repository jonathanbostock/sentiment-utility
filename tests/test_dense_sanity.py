import numpy as np
from question_consistency.questions import Question
from question_consistency.run import dense_compare_all
from fakes import FakeOracle


def test_dense_compare_all_covers_all_ordered_pairs():
    n = 5
    items = [f"i{k}" for k in range(n)]
    q = [Question(id="pos", template="{item_A}{item_B}", valence=1, answers={"A": ["A"], "B": ["B"]})]
    edges = dense_compare_all(FakeOracle(np.arange(n)), items, q)
    pairs = {(e.i, e.j) for e in edges}
    assert len(pairs) == n * (n - 1)     # every ordered pair present
