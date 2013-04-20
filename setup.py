#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception, e:
        print "Forget setuptools, trying distutils..."
        from distutils.core import setup


description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
setup(
    name="nef",
    version="0.1.0",
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=['nef', 'nef.test', 'nef.nef_theano', 'nef.nef_theano.neuron',
              'nef.nef_minimal'],
    scripts=[],
    url="https://github.com/ctn-waterloo/nef-py",
    license="LICENSE.rst",
    description=description,
    long_description=open('README.rst').read(),
    requires=[
        "theano",
        "numpy (>=1.5.0)",
    ],
)
