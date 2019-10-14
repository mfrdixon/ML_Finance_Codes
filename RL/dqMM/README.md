# DQ-MM

DQ-MM is based on Trading Gym, an open-source project for the development of reinforcement learning algorithms in the context of trading.
It is currently composed of a single environment and implements a generic way of feeding this trading environment different type of price data. The focus of the project is level 2 data, i.e. using depths and price levels in the limit order book.

## Installation

`pip install tgym`

We strongly recommend using virtual environments. A very good guide can be found at http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/.

## The trading environment: `SpreadTrading`

`SpreadTrading` is a trading environment allowing to trade a *spread* (see https://en.wikipedia.org/wiki/Spread_trade). We feed the environment a time series of prices (bid and ask) for *n* different products (with a `DataGenerator`), as well as a list of *spread coefficients*. The possible actions are then buying, selling or holding the spread. Actions cannot be taken on one or several legs in isolation. The state of the environment is defined as: prices, entry price and position (whether long, short or flat).

![](https://media.giphy.com/media/l4FGI4K3kHnBfUoIE/giphy.gif)

## Create your own `DataGenerator`

To create your own data generator, it must inherit from the `DataGenerator` base class which can be found in the file 'tgym/core.py'. It consists of four methods. Only the private `_generator` method which defines the times series needs to be overridden. Example can be found at `examples/generator_random.py`. For only one product, the `_generator` method **must** yield a `(bid, ask)` tuple, one element at a time. For two or more products, you must return a tuple consisting of bid and ask prices for each product, concatenated. For instance for two products, the method should yield `(bid_1, ask_1, bid_2, ask_2)`. The logic for the time series is encoded there.

## Compatibility with OpenAI gym

Our environments API is strongly inspired by OpenAI Gym. We aim to entirely base it upon OpenAI Gym architecture and propose Trading Gym as an additional OpenAI environment.

## Examples

Some examples are available in `tgym/examples/`

To run the `dqn_agent.py` example, you will need to also install keras with `pip install keras`. By default, the backend will be set to Theano. You can also run it with Tensorflow by installing it with `pip install tensorflow`. You then need to edit `~/.keras/keras.json` and make sure `"backend": "tensorflow"` is specified.
