#!/usr/bin/env python
import imp
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup  # noqa: F811
from setuptools.command.test import test as TestCommand

class Tox(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox
        errno = tox.cmdline(self.test_args)
        sys.exit(errno)

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo', 'version.py'))
description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
with open(os.path.join(root, 'README.rst')) as readme:
    long_description = readme.read()

setup(
    name="nengo",
    version=version_module.version,
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=find_packages(),
    scripts=[],
    url="https://github.com/ctn-waterloo/nengo",
    license="See LICENSE.rst",
    description=description,
    long_description=long_description,
    install_requires=[
        "numpy>=1.6",
    ],
    extras_require={
        'all_solvers': ["scipy", "scikit-learn"],
    },
    tests_require=['tox'],
    cmdclass={'test': Tox}
)
