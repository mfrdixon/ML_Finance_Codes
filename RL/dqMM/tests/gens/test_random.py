import numpy as np
import sys
sys.path.append('/Users/matthewdixon/Downloads/dq-MM/')
from tgym.gens.random import AR1, RandomWalk


def test_random_walk():
    rw = RandomWalk(ba_spread=0.1)
    val = rw.next()
    assert np.isclose(val[1] - val[0], 0.1)


def test_ar1():
    rw = AR1(a=0.1, ba_spread=0.1)
    val = rw.next()
    assert np.isclose(val[1] - val[0], 0.1)
    vals = [rw.next()[0] for i in range(100000)]
    mean = np.mean(vals)
    std = np.std(vals)
    assert np.isclose(mean, 0, atol=0.01)
    assert np.isclose(std, 1, atol=0.01)
    return rw


if __name__ == "__main__":
    dg = test_ar1()
    print dg.__dict__
