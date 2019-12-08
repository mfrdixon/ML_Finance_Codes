# GP-CVA

# Tested with

# Python 3.6
# Torch 1.0
# GPyTorch 0.1.0


# Installation instructions
# MKL is needed for KISSGP
Sudo conda install -c anaconda mkl
After this, install pytorch and torchvision:

sudo conda install -c pytorch pytorch==1.0.0 torchvision

then pip3 install gpytorch

For Notebook 6 on Heston:
First install GSL
e.g. "apt install libgsl-dev" in Ubuntu
Then install PyHeston
"pip install git+https://github.com/junyanxu/Python-Heston-Option-Pricer.git" after installing GSL

