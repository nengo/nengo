***************************
Introduction for developers
***************************

.. default-role:: obj

Contributing
============

Please read the ``LICENSE.rst`` file to understand
what becoming a contributor entails.
Once you have read and understood the license agreement,
add yourself to the ``CONTRIBUTORS.rst`` file.
Note that all pull requests
must be commited by someone other than the original requestor.

Developer installation
======================

If you want to change parts of Nengo,
you should do a developer installation.

.. code-block:: bash

   git clone https://github.com/nengo/nengo.git
   cd nengo
   python setup.py develop --user

If you are in a virtual environment, you can omit the ``--user`` flag.

How to run unit tests
=====================

Nengo contains a large test suite, which we run with pytest_.
In order to run these tests, install the packages
required for testing with

.. code-block:: bash

   pip install -r requirements-test.txt

You can confirm that this worked by running the whole test suite with::

  py.test --pyargs nengo

Running individual tests
------------------------

Tests in a specific test file can be run by calling
``py.test`` on that file. For example::

  py.test nengo/tests/test_node.py

will run all the tests in ``test_node.py``.

Individual tests can be run using the ``-k EXPRESSION`` argument. Only tests
that match the given substring expression are run. For example::

  py.test nengo/tests/test_node.py -k test_circular

will run any tests with `test_circular` in the name, in the file
``test_node.py``.

Plotting the results of tests
-----------------------------

Many Nengo tests have the built-in ability to plot test results
for easier debugging. To enable this feature,
pass the ``--plots`` to ``py.test``. For example::

  py.test --plots --pyargs nengo

Plots are placed in ``nengo.simulator.plots`` in whatever directory
``py.test`` is invoked from. You can also set a different directory::

  py.test --plots=path/to/plots --pyargs nengo

Getting help and other options
------------------------------

Information about ``py.test`` usage
and Nengo-specfic options can be found with::

  py.test --pyargs nengo --help

Writing your own tests
----------------------

When writing your own tests, please make use of
custom Nengo `fixtures <http://pytest.org/latest/fixture.html>`_
and `markers <http://pytest.org/latest/example/markers.html>`_
to integrate well with existing tests.
See existing tests for examples, or consult::

  py.test --pyargs nengo --fixtures

and::

  py.test --pyargs nengo --markers

.. _pytest: http://pytest.org/latest/

How to build the documentation
==============================

The documentation is built with Sphinx and has a few requirements.
The Python requirements are found in ``requirements-test.txt``
and ``requirements-docs.txt``, and can be installed with ``pip``:

.. code-block:: bash

   pip install -r requirements-test.txt
   pip install -r requirements-docs.txt

However, one additional requirement for building the IPython notebooks
that we include in the documentation is Pandoc_.
If you use a package manager (e.g., Homebrew, ``apt``)
you should be able to install Pandoc_ through your package manager.
Otherwise, see
`this page <http://johnmacfarlane.net/pandoc/installing.html>`_
for instructions.

After you've installed all the requirements,
run the following command from the root directory of ``nengo``
to build the documentation.
It will take a few minutes, as all examples are run
as part of the documentation building process.

.. code-block:: bash

   python setup.py build_sphinx

.. _Pandoc: http://johnmacfarlane.net/pandoc/

Code style
==========

We adhere to
`PEP8 <http://www.python.org/dev/peps/pep-0008/#introduction>`_,
and use ``flake8`` to automatically check for adherence on all commits.
If you want to run this yourself,
then ``pip install flake8`` and run

.. code-block:: bash

   flake8 nengo

in the ``nengo`` repository you cloned.

Class member order
------------------

In general, we stick to the following order for members of Python classes.

1. Class-level member variables (e.g., ``nengo.Ensemble.probeable``).
2. Parameters (i.e., classes derived from `nengo.params.Parameter`)
   with the parameters in ``__init__`` going first in that order,
   then parameters that don't appear in ``__init__`` in alphabetical order.
   All these parameters should appear in the Parameters section of the docstring
   in the same order.
3. ``__init__``
4. Other special (``__x__``) methods in alphabetical order,
   except when a grouping is more natural
   (e.g., ``__getstate__`` and ``__setstate__``).
5. ``@property`` properties in alphabetical order.
6. ``@staticmethod`` methods in alphabetical order.
7. ``@classmethod`` methods in alphabetical order.
8. Methods in alphabetical order.

"Hidden" versions of the above (i.e., anything starting with an underscore)
should either be placed right after they're first used,
or at the end of the class.
Also consider converting long hidden methods
to functions placed in the ``nengo.utils`` module.

.. note:: These are guidelines that should be used in general,
          not strict rules.
          If there is a good reason to group differently,
          then feel free to do so, but please explain
          your reasoning in code comments or commit notes.

Docstrings
----------

We use ``numpydoc`` and
`NumPy's guidelines for docstrings
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_,
as they are readable in plain text and when rendered with Sphinx.

We use the default role of ``obj`` in documentation,
so any strings placed in backticks in docstrings
will be cross-referenced properly if they
unambiguously refer to something in the Nengo documentation.
See `Cross-referencing syntax
<http://www.sphinx-doc.org/en/stable/markup/inline.html#cross-referencing-syntax>`_
and the `Python domain
<http://www.sphinx-doc.org/en/stable/domains.html>`_
for more information.

A few additional conventions that we have settled on:

1. Default values for parameters should be specified next to the type.
   For example::

     radius : float, optional (Default: 1.0)
         The representational radius of the ensemble.

2. Types should not be cross-referenced in the parameter list,
   but can be cross-referenced in the description of that parameter.
   For example::

     solver : Solver
         A `.Solver` used in the build process.
