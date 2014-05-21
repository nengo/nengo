************
Installation
************

Requirements
============

Using the basic functions of Nengo requires ``NumPy``.
To do any kind of visualization, you will need ``matplotlib``.
It is highly recommended that ``IPython`` is installed as
well, in order to fully appreciate the IPython notebook
examples. For Python beginners, an all-in-one solution
like `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
is recommended to install these packages, as well as
Python itself.

Basic installation
==================

This isn't quite true yet, but once we put Nengo
on PyPI, you will be able to::

  pip install nengo

For now, do a developer installation.

Developer installation
======================

If you plan to make changes to Nengo,
you should clone its git repository
and install from it::

  git clone https://github.com/ctn-waterloo/nengo.git
  cd nengo
  python setup.py develop --user

If you're in a virtualenv, you can omit the ``--user`` flag.
