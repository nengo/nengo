#!/usr/bin/env python
import sys
import imp

try:
    from setuptools import setup
    from setuptools.command.test import test as TestCommand
except ImportError:
    try:
        from ez_setup import use_setuptools
        use_setuptools()
        from setuptools import setup
        from setuptools.command.test import test as TestCommand
    except Exception as e:
        print("Forget setuptools, trying distutils...")
        from distutils.core import setup

try:
    class Tox(TestCommand):
        def finalize_options(self):
            TestCommand.finalize_options(self)
            self.test_args = []
            self.test_suite = True

        def run_tests(self):
            #import here, cause outside the eggs aren't loaded
            import tox
            errno = tox.cmdline(self.test_args)
            sys.exit(errno)
    testing = {'tests_require': ['tox'], 'cmdclass': {'test': Tox}}
except NameError:
    testing = {}

version_module = imp.load_source('version', 'nengo/version.py')

description = ("Tools for making neural simulations using the methods "
               + "of the Neural Engineering Framework")
setup(
    name="nengo",
    version=version_module.version,
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
        "networkx",
    ],
    **testing
)
