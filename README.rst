.. image:: https://travis-ci.org/nengo/nengo.png?branch=master
  :target: https://travis-ci.org/nengo/nengo
  :alt: Travis-CI build status

.. image:: https://coveralls.io/repos/nengo/nengo/badge.png?branch=master
  :target: https://coveralls.io/r/nengo/nengo?branch=master
  :alt: Test coverage

.. image:: https://requires.io/github/nengo/nengo/requirements.png?branch=master
  :target: https://requires.io/github/nengo/nengo/requirements/?branch=master
  :alt: Requirements Status

.. image:: https://pypip.in/v/nengo/badge.png
  :target: https://pypi.python.org/pypi/nengo
  :alt: Latest PyPI version

.. image:: https://pypip.in/d/nengo/badge.png
  :target: https://pypi.python.org/pypi/nengo
  :alt: Number of PyPI downloads

********************************************
Nengo: Large-scale brain modelling in Python
********************************************

.. image:: http://c431376.r76.cf2.rackcdn.com/71388/fninf-07-00048-r2/image_m/fninf-07-00048-g001.jpg
  :alt: An illustration of the three principles of the NEF

Installation
============

To install Nengo, use::

  pip install nengo

Nengo depends on `NumPy <http://www.numpy.org/>`_.
If you have difficulty installing,
try install NumPy first.

Nengo supports Python 2.6, 2.7, and 3.3+.

Developer Install
=================

.. code-block:: bash

   git clone https://github.com/nengo/nengo.git
   cd nengo
   python setup.py develop --user

Documentation & Examples
========================

Documentation and examples can be found
`<https://pythonhosted.org/nengo/>`_.


Running tests
=============

One way to verify that your installation is working correctly
is to run the unit tests. We use ``py.test``,
so you can run the Nengo unit tests with::

  py.test --pyargs nengo

Running individual tests
------------------------

Tests in a specific test file can be run by calling ``py.test`` on that file.
For example::

  py.test nengo/tests/test_node.py

will run all the tests in ``test_node.py``.

Individual tests can be run using the ``-k EXPRESSION`` argument. Only tests
that match the given substring expression are run. For example::

  py.test nengo/tests/test_node.py -k test_circular

will run any tests with `test_circular` in the name, in the file
``test_node.py``.

Plotting the results of tests
-----------------------------

Many Nengo test routines have the built-in ability to plot test results
for easier debugging. To enable this feature, set the environment variable
``NENGO_TEST_PLOT=1``, for example::

  NENGO_TEST_PLOT=1 py.test --pyargs nengo

Or, for the current terminal session::

  export NENGO_TEST_PLOT=1
  py.test --pyargs nengo

Plots are placed in ``nengo.simulator.plots`` in whatever directory
``py.test`` is invoked from.

Contributing
============

Please read the ``LISCENCE.rst`` file to understand what becoming a contributor entails.
Once you have read and understood the liscence agreement, add yourself to the ``CONTRIBUTORS.rst`` file.
Note that all pull requests must be commited by someone else other than the original requestor.
