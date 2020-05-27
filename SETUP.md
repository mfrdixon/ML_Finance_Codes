# Setup Guide
Carry out the instructions in this guide to install the software necessary for running the Jupyter notebooks accompanying *Machine Learning in Finance: From Theory to Practice*

After the software has been installed, open a Terminal (macOS/Linux) or the Anaconda Prompt (Windows) and execute the following commands to start a Jupyter notebook server and show the ML_Finance_Codes directory:

 * `conda activate MLFenv`
 * `cd /path/to/your/ML_Finance_Codes`
 * `jupyter notebook`
 
## Install Anaconda/Miniconda
If you do not have conda installed on your system (try `conda info` at the command prompt), follow the instructions at the link below to install Anconda or Miniconda for your operating system:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation

## Install GNU Scientific Library
GSL is required for the PyHeston package used in the Chapter 3 notebook Example-6-GP-Heston.ipynb
### Windows
 * Download and run the CygWin installer for your platform from https://www.cygwin.com/install.html 
 * On the "Select Packages" stage of the installation process, make sure to select the GSL 2.3.2 (#¢# 2.6 not available for cygwin?)
 * #¢# CAN'T TEST THIS CURRENTLY

### macOS
Open the Terminal and run the following commands:
 * Install XCode Command Line Tools
  * `xcode-select --install`
 * Install Homebrew
   * `/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
 * Install GSL
   * `brew install gsl cmake``
 
### GNU/Linux 
Precompiled binary packages are included in most GNU/Linux distributions 
(#¢# instructions for how to make sure)
(#¢# if not, also for installing/updating)

## Install the necessary Python packages 

 * Open a Terminal (macOS/Linux) or an Anaconda Prompt (Windows)
 * Navigate to the 'ML_Finance_Codes' Directory
  * `cd /path/to/your/ML_Finance_Codes`
 * Run the following command to set up the python environment:
  * `conda env create -f environment.yml`
 * Once the installation completes, you should be able to activate the environment from a Terminal/Anaconda Prompt with
  * `conda activate MLFenv` 