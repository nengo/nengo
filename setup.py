#!/usr/bin/env python

# Automatically generated by nengo-bones, do not edit this file directly
# Version: 0.2.0.dev0

import io
import os
import runpy

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(
    root, "nengo", "version.py"))["version"]

install_req = [
    "numpy>=1.11",
]
doc_req = [
    "sphinx>=1.8",
    "jupyter",
    "matplotlib>=1.4",
    "nbsphinx",
    "numpydoc>=0.6",
    "nengo_sphinx_theme>=0.8",
]
optional_req = [
    "scipy>=0.13",
    "scikit-learn",
]
test_req = [
    "jupyter",
    "matplotlib>=1.4",
    "pytest>=3.6",
]

setup(
    name="nengo",
    version=version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    url="https://www.nengo.ai/nengo",
    include_package_data=False,
    license="Free for non-commercial use",
    description="Tools for building and simulating large-scale neural models",
    long_description=read("README.rst", "CHANGES.rst"),
    zip_safe=False,
    install_requires=install_req,
    extras_require={
        "all": doc_req + optional_req + test_req,
        "doc": doc_req,
        "optional": optional_req,
        "test": test_req,
    },
    python_requires=">=3.5",
    package_data={
        "nengo": [
            "nengo-data/nengorc",
        ],
    },
    entry_points={
        "nengo.backends": [
            "reference = nengo:Simulator",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Framework :: Nengo",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
