import numpy as np
from tgym.core import DataGenerator


class RandomWalk(DataGenerator):
    """Random walk data generator for one product
    """
    @staticmethod
    def _generator(ba_spread=0):
        """Generator for a pure random walk

        Args:
            ba_spread (float): spread between bid/ask

        Yields:
            (tuple): bid ask prices
        """
        val = 0
        while True:
            yield val, val + ba_spread
            val += np.random.standard_normal()


class AR1(DataGenerator):
    """Standardised AR1 data generator
    """
    @staticmethod
    def _generator(a, ba_spread=0):
        """Generator for standardised AR1

        Args:
            a (float): AR1 coefficient
            ba_spread (float): spread between bid/ask

        Yields:
            (tuple): bid ask prices and depths
        """
        assert abs(a) < 1
        sigma = np.sqrt(1 - a**2)
        val = 100 # np.random.normal(scale=sigma)
        bid_depth = 1000
        ask_depth = 1000
        eps = 0.01 # minimum price

        while True:
            #print val, val + ba_spread, bid_depth, ask_depth
            yield val, val + ba_spread, bid_depth, ask_depth
            val += (a - 1) * val + np.random.normal(scale=sigma)
            bid_depth += np.int(10*np.random.normal()) # random walk
            ask_depth += np.int(10*np.random.normal()) # random walk
            bid_depth = np.maximum(bid_depth,0)
            ask_depth = np.maximum(ask_depth,0)
	    val = np.maximum(val, eps)