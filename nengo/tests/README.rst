
===========
Nengo tests
===========


Running individual tests
========================

Tests in a specific test file can be run by calling ``py.test`` on that file.
For example::

  py.test nengo/tests/test_node.py

will run all the tests in ``test_node.py``.

Individual tests can be run using the ``-k EXPRESSION`` argument. Only tests
that match the given substring expression are run. For example::

  py.test nengo/tests/test_node.py -k test_circular

will run any tests with `test_circular` in the name, in the file
``test_node.py``.


Plotting
========

Many Nengo test routines have the built-in ability to plot test results
for easier debugging. To enable this feature, set the environment variable
``NENGO_TEST_PLOT=1``, for example::

  NENGO_TEST_PLOT=1 py.test --pyargs nengo

Plots are placed in ``nengo.simulator.plots`` in the top-level Nengo
directory.
