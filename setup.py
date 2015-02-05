#!/usr/bin/env python
import imp
import io
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup  # noqa: F811
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox
        import shlex
        errno = tox.cmdline(args=shlex.split(self.tox_args))
        sys.exit(errno)


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo', 'version.py'))
description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
long_description = read('README.rst', 'CHANGES.rst')

setup(
    name="nengo",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="celiasmith@uwaterloo.ca",
    packages=find_packages(),
    scripts=[],
    data_files=[('nengo', ['nengo-data/nengorc'])],
    url="https://github.com/nengo/nengo",
    license="See LICENSE.rst",
    description=description,
    long_description=long_description,
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=[
        "numpy>=1.6",
    ],
    install_requires=[
        "numpy>=1.6",
    ],
    extras_require={
        'all_solvers': ["scipy", "scikit-learn"],
    },
    tests_require=['pytest>=2.3'],
    cmdclass={
        'test': PyTest,
        'tox': Tox,
    },
    zip_safe=False,
)
