***************************
Introduction for developers
***************************

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

We use ``numpydoc`` and
`NumPy guidelines
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for docstrings, as they are a bit nicer to read in plain text,
and produce decent output with Sphinx.
