#!/usr/bin/env python
import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")


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
testing = 'test' in sys.argv or 'pytest' in sys.argv

setup(
    name="nengo",
    version=version_module.version,
    author="Applied Brain Research",
    author_email="info@appliedbrainresearch.com",
    packages=find_packages(),
    scripts=[],
    data_files=[('nengo', ['nengo-data/nengorc'])],
    url="https://github.com/nengo/nengo",
    license="Free for non-commercial use",
    description="Tools for making neural simulations using the "
                "Neural Engineering Framework",
    long_description=read('README.rst', 'CHANGES.rst'),
    zip_safe=False,
    # Without this, `setup.py install` fails to install NumPy.
    # See https://github.com/nengo/nengo/issues/508 for details.
    setup_requires=["pytest-runner"] if testing else [] + [
        "numpy>=1.8",
    ],
    install_requires=[
        "numpy>=1.8",
    ],
    extras_require={
        'all_solvers': ["scipy>=0.13", "scikit-learn"],
    },
    tests_require=[
        'pytest>=3.2',
    ],
    entry_points={
        'nengo.backends': [
            'reference = nengo:Simulator'
        ],
    },
    classifiers=[  # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
