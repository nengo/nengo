.. image:: https://travis-ci.org/ctn-waterloo/nengo.png?branch=master
  :target: https://travis-ci.org/ctn-waterloo/nengo
  :alt: Travis-CI build status

.. image:: https://coveralls.io/repos/ctn-waterloo/nengo/badge.png?branch=master
  :target: https://coveralls.io/r/ctn-waterloo/nengo?branch=master
  :alt: Test coverage

.. image:: https://requires.io/github/ctn-waterloo/nengo/requirements.png?branch=master
  :target: https://requires.io/github/ctn-waterloo/nengo/requirements/?branch=master
  :alt: Requirements Status

.. image:: https://pypip.in/v/nengo/badge.png
  :target: https://pypi.python.org/pypi/nengo
  :alt: Latest PyPI version

.. image:: https://pypip.in/d/nengo/badge.png
  :target: https://pypi.python.org/pypi/nengo
  :alt: Number of PyPI downloads

============================================
Nengo: Large-scale brain modelling in Python
============================================

.. image:: http://c431376.r76.cf2.rackcdn.com/71388/fninf-07-00048-r2/image_m/fninf-07-00048-g001.jpg
  :alt: An illustration of the three principles of the NEF

Installation
============

We will be making a release on PyPI soon,
meaning you will be able to ``pip install nengo``.
For now, you can do the following::

  pip install -e git://github.com/ctn-waterloo/nengo.git#egg=nengo

Nengo supports Python 2.6, 2.7, and 3.3+ in a single codebase.

Running unit tests
==================

One way to verify that your installation is working correctly
is to run the unit tests. We use ``py.test``,
so you can run the Nengo unit tests with::

  py.test --pyargs nengo

The test suite can take some time to run,
so we recommend install the ``pytest-xdist`` plugin
and running ``py.test --pyargs nengo -n 4``
or however many free CPU cores you have available.

Usage
=====

TODO

Documentation & Examples
========================

Documentation and examples can be found at
`ReadTheDocs <https://nengo.readthedocs.org/en/latest/>`_.
