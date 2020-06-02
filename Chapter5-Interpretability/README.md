# Machine Learning in Finance: From Theory to Practice

## Chapter 5: Interpretability

This chapter contains the following notebooks.

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

### ML_in_Finance-Deep-Learning-Interpretability.ipynb
* The purpose of this notebook is to illustrate a neural network interpretability method which is compatible with linear regression. 
* We generate data from such a linear regression model, train a neural network on this data, and show that the neural network gradients approximate the regression model coefficients. 
* Various simple experimental tests, corresponding to Section 4 of Chpt 5, are performed to illustrate the properties of network interpretability.

### ML_in_Finance-Deep-Learning-Interaction.ipynb
* This notebook illustrates a neural network interpretability method which is compatible with linear regression, including an interaction term. 
* We generate data from such a linear regression model, train a neural network on this data, and show that the neural network gradients approximate the regression model coefficients. 
* Various simple experimental tests, corresponding to Section 4 of Chpt 5, are performed to illustrate the properties of network interpretability.

### ML_in_Finance-Deep-Factor-Models.ipynb
* The purpose of this notebook is to demonstrate the application of deep learning to fundamental factor modeling. The outputs are monthly excess returns, the inputs are fundamental factor loadings (BARRA style). The data provided has already been normalized. 
* The notebook describes the data loading, training using walk-forward optimization, performance evaluation and comparison with OLS regression. The dataset consists of 6 fundamental factors for 218 stocks over a 100 month period starting in February 2008. 
* See the description of the smaller dataset described in Section 6.2 of Chpt 5. See Table 5.4 for a description of the factors.
