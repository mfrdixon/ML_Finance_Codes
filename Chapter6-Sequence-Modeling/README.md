# Machine Learning in Finance: From Theory to Practice

## Chapter 6: Sequence Modeling

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

### ML_in_Finance-PCA-SP500.ipynb
* The purpose of this notebook is to demonstrate the application of PCA, as a method of dimensionality reduction, to time series of adjusted close prices of SP500 listed assets. 
* The analysis assumes that the prices are weakly covariance stationarity and uses the principal components to explain price variance. 
* The components are also compared with the SP500 index price and index prices are regressed on the components to observe their importance.


### ML_in_Finance-ARIMA-HFT.ipynb
* The purpose of this notebook is to demonstrate the application of ARIMA time series modeling to high frequency data. 
* The model predicts changes in the VWAP (volume weighted average prices) based on historical observations of the VWMAP and an exogenous variable, the current Order Flow Imbalance (OFI). 
* The notebook describes many of the steps in Chapter 6 on applying the Dickey-Fuller test to establish stationarity of the endogenous series, followed by use of the ACF and PACF to identify the ARIMA model order. 

\

\
