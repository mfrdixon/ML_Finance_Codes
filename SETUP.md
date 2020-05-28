# Setup Guide
Carry out the instructions in this guide to install the software necessary for running the Jupyter notebooks accompanying *Machine Learning in Finance: From Theory to Practice*

After the installation has completed, open a Terminal (macOS/Linux) or the Anaconda Prompt (Windows) and execute the following commands to start a Jupyter notebook server and show the ML_Finance_Codes directory:

 * Switch to the Python environment for the notebooks

`conda activate MLFenv`
 
 * Navigate to the ML_Finance_Codes directory
 
`cd /path/to/your/ML_Finance_Codes`
 
 * Start the jupyter notebook server
 
`jupyter notebook`
 
## Install Anaconda/Miniconda
If you do not have conda installed on your system (try `conda info` at the command prompt), follow the instructions at the link below to install Anconda or Miniconda for your operating system:
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation

## Install the necessary Python packages 

 * Open a Terminal (macOS/Linux) or an Anaconda Prompt (Windows)
 * Navigate to the 'ML_Finance_Codes' Directory
 
`cd /path/to/your/ML_Finance_Codes`

 * Run the following command to set up the python environment:
 
`conda env create -f environment.yml`

 * Once the installation completes, you should be able to activate the environment from a Terminal/Anaconda Prompt with
 
`conda activate MLFenv` 