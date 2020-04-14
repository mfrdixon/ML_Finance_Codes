# alpha-RNN
alpha-RNN

This repository contains notebook and a subset of the data needed for producing the graphs in the paper: 

M. F. Dixon, Industrial Forecasting with Exponentially Smoothed Recurrent Neural Networks, 2020.


- Alpha_RNNs_regime_switching.ipynb: generates plots for Example 2 in the paper

- Alpha_RNNs_weather.ipynb: generates plots for Example 3 in the paper

- Alpha_RNNs_DK2_electricity.ipynb: generates plots for Example 4 in the paper

requires Load7.csv


- Alpha_RNNs_Bitcoin.ipynb: generates plots for Example 5 in the paper
requires coinbase.csv.zip (unzip first)

- Alpha_RNNs_HFT.ipynb: generates plots for Example 6 in the paper
requires HFT.csv.zip (unzip first)



Note that cross-validation in each notebook is disabled by default to avoid excessive run-times. This can be enabled by seting the cross_val flag to True.

All notebooks have been tested on Mac OS X (Sierra 10.12.6) with Python 3.6 and TensorFlow 1.4. 
The notebooks have also been tested in Google Colab with firefox version 72.0.2.
