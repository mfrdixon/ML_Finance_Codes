# Machine Learning in Finance   
## Chapter 8: Advanced Neural Networks

##### This chapter contains the following notebooks demonstrating the use of advanced neural network architectures:

### 1D Convolutional Neural Networks

 * This notebook shows the process of creating a 1D convolutional neural network with Keras
 * An example timeseries is created and formatted for the training of the model
 * Its ability to predict beyond the training data is demonstrated by comparing its predictions to the true values

### 2D Convolutional Neural Networks

 * This notebook shows the process of creating a 2D convolutional neural network
 * The MNIST dataset is loaded and transformed for input into the model, and split into a training and testing set
 * The model's out-of-sample classification performance is evaluated on the test set

### Autoencoders
**files**: yield_curve.csv

* This notebook compares linear dimensionality reduction using principal component analysis against that achieved in an autoencoder neural network
* A review of PCA is provided
* An autoencoding neural network is created and trained on the `yield_curves.csv` data set
* The results of using the principal components and the  weights learned by the autoencoder to transform the dataset are compared

### RNN's: Bitcoin
**files**: *coinbase.csv, alphaRNN.py, alphatRNN.py*
* This notebook shows an example of a recurrent neural network for time series prediction
* To select an appropriate model architecture, the univariate time series `coinbase.csv` is analysed for stationarity and its partial auto-correlation function is estimated
* The time series is transformed into a set of input sequences and corresponding outputs for use with the RNN, and split into training and testing sets
* RNN, alphaRNN, alphatRNN, LSTM & GRU models are trained on the time series
* An example of a time series cross-validation procedure is provided
 * This is disabled by default; the cross-validation process involves training the model many times, and can take several hours to complete.
 * Set `cross_val = True` to enable

### RNN's: HFT
**files**: *HFT.csv, alphaRNN.py, alphatRNN.py*
 * This notebook shows an example of a recurrent neural network for time series prediction
 * To select an appropriate model architecture, the univariate time series `HFT.csv` is analysed for stationarity and its partial auto-correlation function is estimated
 * The time series is transformed into a set of input sequences and corresponding outputs for use with the RNN, and split into training and testing sets
 * RNN, alphaRNN, alphatRNN, LSTM & GRU models are trained on the time series
 * An example of a time series cross-validation procedure is provided
  * This is disabled by default; the cross-validation process involves training the model many times, and can take several hours to complete.
  * Set `cross_val = True` to enable