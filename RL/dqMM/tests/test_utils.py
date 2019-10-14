import numpy as np
from tgym.utils import calc_spread


def test_calc_spread():
    spread_coefficients = [1, -0.1]
    prices = np.array([1, 2, 10, 20])
    spread_price = (-1, 1)
    assert calc_spread(prices, spread_coefficients) == spread_price
