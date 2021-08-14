#!/usr/bin/env python

from distutils.core import setup

setup(
    name="torch_optical_flow",
    version="0.1",
    description="Optical Flow Utilities for PyTorch",
    author="Adrian Wälchli",
    author_email="aedu.waelchli@gmail.com",
    url="https://github.com/awaelchli/torch-optical-flow",
    packages=["optical_flow"],
    install_requires=[
        "Pillow",
        "torch",
        "torchmetrics",
    ],
)
