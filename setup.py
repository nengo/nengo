#!/usr/bin/env python

from distutils.core import setup

description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
setup(
    name="nengo",
    version="0.1.0",
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=['nef', 'nef.test', 'nef.nef_theano', 'nef.nef_theano.neuron',
              'nef.nef_minimal'],
    scripts=[],
    url="https://github.com/ctn-waterloo/nef-py",
    license="GPLv3",
    description=description,
    long_description=open('README.rst').read(),
    requires=[
        "theano",
        "numpy (>=1.5.0)",
    ],
)
