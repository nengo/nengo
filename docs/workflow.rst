********************
Development workflow
********************

Development happens on `Github <https://github.com/nengo/nengo>`_.
Feel free to fork any of our repositories and send a pull request!
However, note that we ask contributors to sign
`a copyright assignment agreement <https://github.com/nengo/nengo/blob/master/LICENSE.rst>`_.

Code style
==========

We adhere to
`PEP8 <http://www.python.org/dev/peps/pep-0008/#introduction>`_,
and use ``flake8`` to automatically check for adherence on all commits.

We use ``numpydoc`` and
`NumPy guidelines <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for docstrings, as they are a bit nicer to read in plain text,
and produce decent output with Sphinx.

Unit testing
============

We use `PyTest <http://pytest.org/latest/>`_ to run our unit tests
on `Travis-CI <https://travis-ci.com/>`_.
To ensure Python 2/3 compatibility, we test with
`Tox <https://tox.readthedocs.org/en/latest/>`_.

For more information on running tests, see the README.

Git
===

We use a pretty strict ``git`` workflow
to ensure that the history of the ``master`` branch
is clean and readable.
Every commit in the ``master`` branch should pass
unit testing, including PEP8.

Developers should never edit code on the ``master`` branch.
When changing code, create a new topic branch
that implements your new feature or fixes a bug.
When your branch is ready to be reviewed,
push it to Github and create a pull request.
One or more people will review your pull request,
and over one or many cycles of review,
your PR will be accepted or rejected.
We almost never reject PRs,
though we do let them languish in the limbo
of the PR queue if we're not sure
if they're quite ready yet.

Some developers, speficially
Trevor Bekolay, Eric Hunsberger, Jan Gosmann, and Daniel Rasmussen,
are also maintainers, which means that they
are primarily repsonsible for reviewing your work,
and merging it into the ``master`` branch when it's been accepted.
Only maintainers are allowed to push to the ``master`` branch.

If you have any questions about our workflow,
or how you can best climb the learning curve
that Nengo and ``git`` present, please contact
the development lead, `Trevor <tbekolay@gmail.com>`_.
