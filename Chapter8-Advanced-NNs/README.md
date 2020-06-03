# Machine Learning in Finance: From Theory to Practice

## Chapter 8: Advanced Neural Networks

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

### ML_in_Finance-RNNs-Bitcoin.ipynb
 * This notebook shows an example of a recurrent neural network (RNN) for time series prediction. Please refer to Chapter 8, Sections 2-4 in the textbook
 * To select an appropriate model architecture, the univariate time series `coinbase.csv` is analysed for stationarity and its partial auto-correlation function is estimated
 * The time series is transformed into a set of input sequences and corresponding outputs for use with the RNN, and split into training and testing sets
 * RNN, alphaRNN, alphatRNN, LSTM & GRU models are trained on the time series
 * An example of a time series cross-validation procedure is provided
   * This is disabled by default; the cross-validation process involves training the model many times, and can take many hours to complete.

### ML_in_Finance-RNNs-HFT.ipynb
 * This notebook shows an example of a recurrent neural network (RNN) for time series prediction. Please refer to sections Chapter 8, Sections 2-4 in the textbook
 * To select an appropriate model architecture, the univariate time series `HFT.csv` is analysed for stationarity and its partial auto-correlation function is estimated
 * The time series is transformed into a set of input sequences and corresponding outputs for use with the RNN, and split into training and testing sets
 * RNN, alphaRNN, alphatRNN, LSTM & GRU models are trained on the time series
 * An example of a time series cross-validation procedure is provided
   * This is disabled by default; the cross-validation process involves training the model many times, and takes many hours to complete

### ML_in_Finance-1D-CNNs.ipynb
 * This notebook shows the process of creating a 1D convolutional neural network with Keras. Please refer to Chapter 8, Section 5 of the textbook
 * An example timeseries is created and formatted for the training of the model
 * Its ability to predict beyond the training data is demonstrated by comparing its predictions to the true values

### ML_in_Finance-2D-CNNs.ipynb
 * This notebook shows the process of creating a 2D convolutional neural network. Please refer to Chapter 8, Section 5 of the textbook
 * The MNIST dataset is loaded and transformed for input into the model, and split into a training and testing set
 * The model's out-of-sample classification performance is evaluated on the test set

### ML_in_Finance-Autoencoders.ipynb
 * This notebook compares linear dimensionality reduction using principal component analysis against that achieved in an autoencoder neural network. Please refer to Chapter 8, Section 6 in the textbook
 * A review of PCA is provided
 * An autoencoding neural network is created and trained on the `yield_curves.csv` data set
 * The results of using the principal components and the  weights learned by the autoencoder to transform the dataset are compared