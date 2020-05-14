# Chapter 4 notebooks 

# Tested with 
# Python 3.6

# Installation notes
ML_Finance_Bayesian_Neural_Network requires the arviz package (for tracing weights)
then pip3 install arviz 

# ML_in_Finance-Deep-Classifiers
This notebook correspondes to Section 2.2 of Chpt 4 and demonstrates the configuration and properties of feed forward neural networks. We will investigate how the perceptron units transform the input space. Specifically, we shall study the fitted weights and plot the separating hyperplanes, starting with no hidden layers, adding a hidden layer and then two hidden layers. We shall also observe the effect of changing the number of perceptron units in a layer.


# ML_in_Finance-Backpropagation 
This notebooks demonstrate the back-propagation algorithm in detail and compares the results of an implementation with tensorflow. See Section 5.1 of Chapter 4 for further details.

# ML_Finance_Bayesian_Neural_Network
This notebook correspondes to Section 6 in Chpt 4 and demonstrates the application of Bayesian Neural Networks to the half-moon problem. The material is more advanced and provided for completeness as Bayesian modeling was covered in earlier chapters. The notebook uses variation inference, implemented in PyMC3. Variational inference algorithms are faster than MCMC methods - the latter sample from the exact posterior distribution whereas the former sample from approximations such as ELBO. The advantage of the Bayesian approach is that it informs us about the uncertainty in its predictions. Another advantage is that Bayesion estimation is intrinsically self-regularizing through the choice of prior on the weights. In frequentist estimation, weights are often L2-regularized to avoid overfitting, this very naturally becomes a Gaussian prior for the weight coefficients. We could, however, imagine all kinds of other priors, like spike-and-slab to enforce sparsity (this would be more like using the L1-norm).
