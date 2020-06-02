# Machine Learning in Finance: From Theory to Practice

## Chapter 9: Reinforcement Learning

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

### ML_in_Finance_FCW.ipynb

 * This notebook shows the application of reinforcement learning to the financial cliff walking problem. The problem is described in Example 9.4 in the textbook.
 * A discretised, time-dependent action value function is 
 * The convergence of the algorithms is compared by plotting the average reward gained against number of training episodes
 * The actions learned for each state by the two algorithms are inspected

###  ML_in_Finance_Market_Impact.ipynb
 * This notebook shows the application of SARSA and Q-learning to the optimal stock execution problem described in Example 9.5 in the textbook.
 * The convergence of the algorithms is compared by plotting the average reward gained against number of training episodes

### ML_in_Finance_MarketMaking.ipynb
 * This notebook shows the application of reinforcement learning to the problem of high-frequency market making. The problem is described in Example 9.6 in the textbook.
 * SARSA and Q-learning are applied to learn time-independent optimal policies based on historical limit order book data.
 * The convergence of the algorithms is compared by plotting the reward gained after each training episode
 * An animation demonstrating the behaviour of the learned policies is shown

### ML_in_Finance_LSPI_Markowitz.ipynb
 * Reinforcement learning is applied to the problem of optimal allocation
 * The least squares policy iteration (LSPI) algorithm is applied to a Monte Carlo simulation of a stock's price movements, constructing a basis over the state-action space using B-spline basis functions at each time period.
* The optimal Q-function is approximated with a dynamic programming approach, and this is shown to approach the exact solution