# Chapter 6 notebooks 

# Tested with 
# Python 3.6

# Installation notes

# ML_in_Finance-PCA-SP500 
The purpose of this notebook is to demonstrate the application of PCA, as a method of dimensionality reduction, to time series of adjusted close prices of SP500 listed assets. The analysis assumes that the prices are weakly covariance stationarity and uses the principal components to explain price variance. The components are also compared with the SP500 index price and index prices are regressed on the components to observe their importance.


# ML_in_Finance-ARIMA-HFT
The purpose of this notebook is to demonstrate the application of ARIMA time series modeling to high frequency data. The model predicts changes in the VWAP (volume weighted average prices) based on historical observations of the VWMAP and an exogenous variable, the current Order Flow Imbalance (OFI). See https://arxiv.org/abs/1011.6402 for more details on OFI.

The notebook describes many of the steps in Chapter 6 on applying the Dickey-Fuller test to establish stationarity of the endogenous series, followed by use of the ACF and PACF to identify the ARIMA model order. Typically, addition diagnostic tests are performed on the model residual but omitted here. Instead we simply compare the forecast with the actual VWAP change.
