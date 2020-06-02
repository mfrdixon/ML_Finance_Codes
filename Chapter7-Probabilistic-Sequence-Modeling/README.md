# Machine Learning in Finance: From Theory to Practice

## Chapter 7: Probabilistic Sequence Modeling

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

### ML_in_Finance-Kalman_Filters.ipynb
* The purpose of this notebook is to demonstrate the application of particle filtering to stochastic volatility modeling. 
* The insight gained from Chapter 2, namely an iterative application of Bayes's theorem, referred to as "sequential Bayesian updates", is the foundation of real-time Bayesian filtering. Kalman filtering is well known example of Bayesian filtering. 


### ML_in_Finance-Viterbi.ipynb
* This notebook provides a simple example of the Viterbi algorithm applied to an coin which is either fair or loaded (hidden states). 
* Based on a sequence of observations, the algorithm will determine the most likely sequence of hidden states, i.e. whether the coin that generated the data was likely fair or loaded. 
* The example easily maps onto financial markets: Bull or Bear, Normal or Dislocated, etc. 

\

\