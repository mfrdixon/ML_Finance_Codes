import numpy as np
import sys
sys.path.append('/Users/matthewdixon/Downloads/dq-MM/')
from tgym.envs import SpreadTrading
from tgym.gens.csvstream import CSVStreamer 


class TestSpreadTrading(object):

    #data_generator = AR1(a=0.1, ba_spread=0.1) 
    data_generator = CSVStreamer(filename='../../data/AMZN-L1.csv')
    st = SpreadTrading(
        data_generator=data_generator,
        spread_coefficients=[1],
        trading_fee=0.2,
        time_fee=0.1,
        history_length=2
    )

    def test_init(self):
        assert self.st._data_generator == self.data_generator
        assert self.st._spread_coefficients == [1]
        assert self.st._first_render
        assert self.st._trading_fee == 0.2
        assert self.st._time_fee == 0.1
        assert self.st._episode_length == 1000
        assert self.st.n_actions == 3
        assert self.st._history_length == 2
        assert len(self.st._prices_history) == 2

    def test_step(self):
        # Buy
        state = self.st.step(np.array([0, 1, 0]))
        #assert state[0][0] == state[0][1]
        assert all(state[0][-3:] == np.array([0, 1, 0]))
        assert self.st._entry_price != 0
        assert self.st._exit_price == 0
        # Hold
        state = self.st.step(np.array([1, 0, 0]))
        assert all(state[0][-3:] == np.array([0, 1, 0]))
        assert self.st._entry_price != 0
        assert self.st._exit_price == 0
        # Sell
        state = self.st.step(np.array([0, 0, 1]))
        assert all(state[0][-3:] == np.array([1, 0, 0]))
        assert self.st._entry_price == 0
        assert self.st._exit_price != 0

    def test_reset(self):
        return self.st.reset()

if __name__ == "__main__":
    env = TestSpreadTrading() 
    print env.st._data_generator.next()
    env.test_init()
    env.test_step()
    print env.test_reset()
