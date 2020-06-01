# Machine Learning in Finance: From Theory to Practice

## Chapter 4: Neural Networks

This chapter contains the following notebooks.

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

### ML_in_Finance-Deep-Classifiers.ipynb
This notebook corresponds to Section 2.2 of Chapter 4 and demonstrates the configuration and properties of feed-forward neural networks. Several neural networks of increasing complexity are created to investigate how the perceptron units transform the input space. 
Beginning with a single-layer perceptron, before adding one and then two hidden layers, we study the fitted weights and plot the separating hyperplanes. The effect of changing the number of perceptron units in a layer is also demonstrated.

### ML_in_Finance-Backpropagation.ipynb
This notebooks demonstrate the back-propagation algorithm, applied to a three layer feed-forward network. The process of creating and training the same neural network in Keras is then shown to compare the results. Refer to Section 5.1 of Chapter 4 for further details. 

### ML_Finance_Bayesian_Neural_Network.ipynb
This notebook corresponds to Section 6 in Chapter 4 and demonstrates the application of Bayesian neural networks to the half-moon problem. The material is more advanced and provided for completeness as Bayesian modeling was covered in earlier chapters. The notebook uses variational inference, implemented in PyMC3. 