=====================
Making Nengo releases
=====================

While we endeavour to automate as much as
the below as possible,
it's nonetheless important to know
how a Nengo release works.
There are three stages to this process,
which will result in at least
two ``git`` commits and one ``git`` tag.

Note that these steps should be consulted
for both release candidates, and full releases.
However, some steps (notably those that interact
with PyPI) should be omitted for release candidates.

Stage 1: Preparation
====================

Before making a release,
we do a few things to prepare:

1. Ensure ``master`` is up-to-date by doing ``git pull``.
2. Ensure that important changes since the last release are
   chronicled in the changelog (``CHANGES.rst``).
3. Run `check-manifest <https://pypi.python.org/pypi/check-manifest>`_
   to ensure that all files are included in the release.
4. Run all unit test to ensure they pass on all supported versions.
   This includes tests that are normally skipped
   due to slow runtimes. This requires that optional
   dependencies are installed. Currently, running all tests is done with
   ``NENGO_TEST_PLOT=1 pytest --pyargs nengo --benchmarks --optional``
   (in environments for each supported Python).
5. Review all of the plots generated from running the unit tests
   for abnormalities or unclear figures.
6. Build the documentation and review all of the rendered
   examples for abnormalities or unclear figures.
7. Commit all changes from above before moving on to stage 2.

.. todo::

   Step 4 is a bit vague at the moment; we should have a separate
   document for all the platforms supported,
   and how to test on Windows with Vagrant, etc.
   However, it is also the hope that release candidates
   will be tested by many different environments
   reducing the burden on the developers to
   have all possible environments available.

Note that any possibly controversial fixes done as a result of
Stage 1 should be done through the normal process of making
a pull request and going through review.
However, from Stage 2 onward, the work is done directly
on the ``master`` branch.
It can therefore result in bad things,
so proceed with caution!

Stage 2: Make release commit
============================

Once everything is prepared, we're ready to do the release.
It should be okay to work in the same directory that you
do development, but if you want to be extra safe,
you can do a fresh clone of Nengo into a separate directory.

1. Change the version information in ``nengo/version.py``.
   See that file for details.
2. *If this is a release*, set the release date in the changelog
   (``CHANGES.rst``).
3. ``git add`` the changes above and make a release commit with:
   ``git commit -m "Release version $(python -c 'import nengo; print(nengo.__version__)')"``
4. Review ``git log`` to ensure that the version number is correct; if not,
   then something went wrong with the previous steps.
   Correct these mistakes and amend the release commit accordingly.
5. Tag the release commit with the version number; i.e.,
   ``git tag -a v$(python -c 'import nengo; print(nengo.__version__)')``.
   We use annotated tags for the authorship information;
   if you wish to provide a message with information about the release,
   feel free, but it is not necessary.
6. ``git push origin master`` and ``git push origin [tagname]``.
7. *If this is a release*, create a package and upload it to PyPI
   with ``python setup.py sdist upload``.
8. *If this is a release*, build the documentation with
   ``python setup.py build_sphinx``. Zip it up and upload it through
   `this form <https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=nengo>`_

Stage 3: Post-release cleanup
=============================

Nengo's now released!
We need to do a few last things to
put Nengo back in a development state.

1. Change the version information in ``nengo/version.py``.
   See that file for details.
2. *If this is a release*, make a new changelog entry in ``CHANGES.rst``
   in order to collect changes for the next release.
3. ``git add`` the changes above and make a commit describing
   the current state of the repository; either
   ``git commit -m "Continuing development of vX.Y.Z"`` or
   ``git commit -m "Starting development of vX.Y.Z"``.
4. ``git push origin master``

Congrats, you've released Nengo!
Shake off the nerves of working directly on ``master``,
and make sure that ``pip install nengo`` gets the new version,
if it was a full release.
