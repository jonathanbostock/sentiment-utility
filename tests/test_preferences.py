import numpy as np
from sentiment_utility.preferences import combine_orderings


def test_combine_orderings_symmetry_and_values():
    # 2 items. When A=0,B=1 model picks item0 with prob 0.8.
    # When A=1,B=0 model picks item1 with prob 0.6 -> picks item0 with 0.4.
    ordered = {(0, 1): 0.8, (1, 0): 0.6}
    pref = combine_orderings(2, ordered)
    # P(0>1) = 0.5*(0.8 + (1-0.6)) = 0.6
    assert np.isclose(pref[0, 1], 0.6)
    # anti-symmetry
    assert np.isclose(pref[1, 0], 0.4)
    assert np.isclose(pref[0, 0], 0.5)  # diagonal convention
