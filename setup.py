#!/usr/bin/env python
import imp
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import find_packages, setup

root = os.path.dirname(os.path.realpath(__file__))
version_module = imp.load_source(
    'version', os.path.join(root, 'nengo_gui', 'version.py'))
description = "Web-based GUI for building and visualizing Nengo models."
with open(os.path.join(root, 'README.rst')) as readme:
    long_description = readme.read()

setup(
    name="nengo_gui",
    version=version_module.version,
    author="CNRGlab at UWaterloo",
    author_email="celiasmith@uwaterloo.ca",
    packages=find_packages(),
    data_files=[(dirpath, [os.path.join(dirpath, f) for f in filenames])
                for dirpath, dirnames, filenames
                in os.walk('nengo_gui/scripts')] +
    [(dirpath, [os.path.join(dirpath, f) for f in filenames])
     for dirpath, dirnames, filenames in os.walk('nengo_gui/static')] +
    [(dirpath, [os.path.join(dirpath, f) for f in filenames])
     for dirpath, dirnames, filenames in os.walk('nengo_gui/templates')],
    entry_points={
        'console_scripts': [
            'nengo_gui = nengo_gui:main',
        ]
    },
    scripts=[],
    url="https://github.com/ctn-waterloo/nengo_gui",
    license="https://github.com/ctn-waterloo/nengo_gui/blob/master/LICENSE.md",
    description=description,
    long_description=long_description,
    requires=[
        "nengo",
        "rpyc",
    ],
)
