import numpy as np
from tgym.core import DataGenerator


class WavySignal(DataGenerator):
    """Modulated sine generator
    """
    @staticmethod
    def _generator(period_1, period_2, epsilon, ba_spread=0):
        i = 0
        while True:
            i += 1
            bid_price = (1 - epsilon) * np.sin(2 * i * np.pi / period_1) + \
                epsilon * np.sin(2 * i * np.pi / period_2)
            yield bid_price, bid_price + ba_spread
