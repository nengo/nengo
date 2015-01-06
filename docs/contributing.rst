************
Contributing
************

Development happens on `Github <https://github.com/nengo/nengo>`_.
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

When creating a network, it is recommended to add the inputs and outputs of the network to
the docstring. For an example of this, please see the simple
`integrator network <https://github.com/nengo/nengo/blob/master/nengo/networks/integrator.py>`_

Unit testing
============

We use `PyTest <http://pytest.org/latest/>`_ to run our unit tests on `Travis-CI 
<https://travis-ci.com/>`_. All of our tests live in `nengo/nengo/tests`.

For more information on running tests, see the README.