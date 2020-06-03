# Machine Learning in Finance: From Theory to Practice

## Chapter 11: Inverse Reinforcement Learning

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

### ML_in_Finance_IRL_FCW.ipynb

* In this notebook, three inverse reinforcement algorithms are applied to the Financial Cliff Walking (FCW) problem.
* These are Max Causal Entropy (Maxent), Inverse Reinforcement Learning from Failure (IRLF), and Trajectory-ranked Reward EXtrapolation (T-REX).  
* After training them on the FCW problem, and the state-action values learned by each algorithm are compared alongside the "ground truth" values.
* The reward distributions of the IRLF algorithm are compared for successful and unsuccessful trials.

### ML_in_Finance-GIRL-Wealth-Management.ipynb
* This notebook demonstrates the application of G-learning and GIRL for optimization of a defined contribution retirement plan. The notebook extends the G-learning notebook in Chapter 10 with an example of applying GIRL to infer the parameters of the G-learner used to generate the trajectories.
