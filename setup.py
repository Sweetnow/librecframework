#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='librecframework',
    version='1.3.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.17.2',
        'scipy>=1.3.1',
        'pandas>=1.1.3'
        #'tensorboard>=1.15.0',
        'visdom==0.1.8.9',
        'setproctitle>=1.1.10',
        'torch>=1.2.0',
        'nvidia_ml_py3==7.352.0'
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    author='Jun Zhang',
    author_email='zhangjun990222@qq.com',
    description='Python library for recommender system based on pytorch',
    zip_safe = False
)
