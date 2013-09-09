=================================
Contributing to Nengo development
=================================

Nengo is an open source project that welcomes contributions from anyone.
Development happens on `Github <https://github.com/ctn-waterloo/nengo>`_.
Feel free to fork any of our repositories and send a pull request!
Please see the individual repositories for more information
on the implementation and how you can extend it.

Code style
==========

We try to stick to
`PEP8 <http://www.python.org/dev/peps/pep-0008/#introduction>`_.

We use ``numpydoc`` and
`NumPy guidelines <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for docstrings, as they are a bit nicer to read in plain text,
and produce decent output with Sphinx.

Unit testing
============

TODO some text

We provide some helpers to make unit testing easier
for Nengo developers.

To run all unit tests you can use several methods.

1. From the main ``nengo`` directory, run ``python setup.py test -q``.
2. From the main ``nengo`` directory, run ``python -m unittest discover``.

To run specific unit tests, run ``python /path/to/test_file.py TestClass.test_function``. Everything except for the path to the test file is optional.

.. autoclass:: nengo.tests.helpers.Plotter
   :members:

.. autoclass:: nengo.tests.helpers.SimulatorTestCase
   :members:
