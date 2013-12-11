#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
    except Exception as e:
        print("Forget setuptools, trying distutils...")
        from distutils.core import setup


description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
setup(
    name="nengo",
    version="2.0.0",
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=['nengo', 'nengo.tests', 'nengo.networks'],
    scripts=[],
    url="https://github.com/ctn-waterloo/nengo",
    license="GPLv3",
    description=description,
    long_description=open('README.rst').read(),
    requires=[
        "numpy (>=1.5.0)",
    ],
    test_suite='nengo.tests',
)
