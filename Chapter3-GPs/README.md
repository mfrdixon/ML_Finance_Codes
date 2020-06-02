# Machine Learning in Finance: From Theory to Practice

## Chapter 3: Gaussian Processes

For instructions on how to set up the Python environment and run the notebooks please refer to [SETUP.html](../SETUP.html) in the *ML_Finance_Codes* directory.

This chapter contains the following notebooks:

## Core Notebooks

### ML_in_Finance_Bayesian_Linear_Regression.ipynb
* The purpose of this notebook is to illustrate Bayesian linear regression, as described in Sections 1 & 2 in Chapter 3.
* It provides visualisations of Bayesian inference on the parameter distributions and of the resulting model and uncertainties in the input space. 

### Example-1-GP-BS-Pricing.ipynb
* The purpose of this notebook is to demonstrate the fitting of a Gaussian Process Regression model (GP) to option price data. In this notebook, European option prices are generated from the Black-Scholes model. 
* The notebook begins by building a GP call model, then a GP put model, and finally a portfolio holding which is short a put and long two calls.
* Note that the training sample size is set to 5 in order to illustrate the uncertainty bands. See Section 5.0 for a description of using GP for pricing derivatives. 

### Example-2-GP-BS-Pricing.ipynb
* This notebook demonstrates the derivation of the Greeks in a Gaussian Process Regression model (GP), fitted to option price data. European option prices are generated from the Black-Scholes model.
* The notebook begins by building a GP call model, where the input is the underlying price. The delta is then derived and compared with the Black-Scholes (BS) delta.
* The exercise is then repeated using the volatility as the input instead of the underlying price. The vega of the GP is then derived and compared with the BS vega. See Section 5.1 for further details.

### Example-6-GP-Heston.ipynb

This notebook requires Junyan Xu's Python Heston Option Pricing Library

* macOS and Linux users can install it by executing the following command at a terminal:
    * On macOS, if you are prompted to install the XCode command line tools, do so.

    `conda activate MLFenv && pip install git+https://github.com/junyanxu/Python-Heston-Option-Pricer`

* Windows users are recommended to run the notebook on Google Colab by following these steps:

    1. Copy and paste this URL into your browser's address bar and sign in with a Google account:

            https://colab.research.google.com

    2. Click on **File>Open notebook** in the Colab menu, then select the Upload tab and choose the *Example-6-GP-Heston.ipynb* file. 
    
    3. Then click the folder icon in the sidebar on the left to open the Files panel. Immediately click **Upload**, and upload *BlackScholes.py* from the *Chapter3-GPs* folder. It should appear alongside the "sample_data" folder in the Files panel.
    
    4. Now make a new code cell at the top of the file, paste the command below into it, and execute it to install PyHeston:

        `!apt install libgsl-dev && pip install git+https://github.com/junyanxu/Python-Heston-Option-Pricer.git`
        
    5. You can now continue through the notebook.

* The notebook demonstrates the fitting of a Gaussian Process regression model (GP) to option price data. In this notebook, European option prices are generated from the Heston stochastic volatility model.
* It begins with the building of a GP Heston model, using the underlying and volatility as a two-dimensional input over a maturity time grid. The strike and other model parameters are assumed fixed. See Section 5 for further details.

### Example-7-MGP-BS-Pricing.ipynb
* The fitting of a multi-response Gaussian Process Regression model (GP) to the prices of two options. In this notebook, the prices of a call and put are generated from the Black-Scholes model.  The notebook begins by building a multi-GP model, and then evaluates the error.
* Finally the notebook studies the posterior covariance term which is uniquely available in the multi-GP model. See Section 6 for further details. 

## Extras

### Example-3-MC-GPA-BS-CVA.ipynb
* The purpose of this notebook is to demonstrate the use of a Gaussian Process Regression model (GP) in CVA modeling of a derivative portfolio.
* European option prices are generated from the Black-Scholes model. The notebook illustrates CVA estimation for a portfolio which is short a put and long two calls.
* The GP model is combined with an Euler time-stepper to generate a GP MtM cube. The expected positive exposure of the portfolio is compared using exact derivative prices and GP derivative prices.

### Example-4-MC-GPA-BS-CVA-VaR.ipynb
* This notebook shows the use of a Gaussian Process Regression model (GP) in CVA Value-at-Risk modeling of a derivative portfolio.
* European option prices are generated from the Black-Scholes model. The notebook illustrates the CVA estimation for a portfolio which is short a put and long two calls.
* The GP model is combined with an Euler time-stepper to generate a GP MtM cube. The distribution of the one year CVA (with uncertainty quantification) is estimated based on priors for the default model parameters. 
* Finally the difference of the one year CVA distribution and the CVA_0 is computed and the quantiles provided the 1 year CVA VaR estimates.

### Example-5-MC-GPA-IRS-CVA.ipynb
* This example demonstrates the application of GPs to CVA modeling on a counterparty portfolio of IRS contracts with 11 currencies and 10 FX processes.
* The notebook simulates the GP MtM cube of the portfolio, compares the Expected Positive Exposure when using a GP derivative pricing model versus a BS pricing model, and calculates the CVA_0.
