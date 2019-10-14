#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='tgym',
    version='0.1.14',
    description="Trading Gym is an open-source project for the development of reinforcement learning algorithms in the context of trading.",
    author="Prediction Machines",
    author_email='tgym@prediction-machines.com',
    url='https://github.com/prediction-machines/tgym',
    packages=find_packages(),
    install_requires=[
        'matplotlib==2.0.2'
    ],
    license="MIT license",
    zip_safe=False,
    keywords='tgym'
)
