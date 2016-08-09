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

Stage 1: Preparing
==================

Before making a release,
we do a few things to prepare:

1. Ensure ``master`` is up-to-date by doing ``git pull``.

2. Run `check-manifest <https://pypi.python.org/pypi/check-manifest>`_
   to ensure that all files are included in the release.

3. Run all tests to ensure they pass on Python 2 and 3,
   including slow tests that are normally skipped.

   .. code:: bash

      py.test --pyargs nengo --plots --analytics --logs --slow

4. Review all of the plots generated from running the tests
   for abnormalities or unclear figures.

5. Run all tests with the Nengo OCL backend.

   .. code:: bash

      py.test --pyargs nengo --plots --simulator nengo_ocl.Simulator

   If any tests fail, attempt to fix them by changing Nengo.
   If they cannot be fixed in Nengo, then
   `file an issue <https://github.com/nengo/nengo_ocl/issues>`_.

6. Build the documentation and review all of the rendered
   examples for abnormalities or unclear figures.

7. Commit all changes from the above steps before moving on to the next stage.

.. todo::

   Currently we do not run slow tests on all platforms (Windows, Mac OS X, Linux
   with 32-bit and 64-bit versions of Python 2.7, 3.3, 3.4, and 3.5).
   Doing so is difficult without dedicated hardware.

Note that any possibly controversial fixes done as a result of
Stage 1 should be done through the normal process of making
a pull request and going through review.
However, from Stage 2 onward, the work is done directly
on the ``master`` branch.
It can therefore result in bad things,
so proceed with caution!

Stage 2: Releasing
==================

Once everything is prepared, we're ready to do the release.
It should be okay to work in the same directory that you
do development, but if you want to be extra safe,
you can do a fresh clone of Nengo into a separate directory.

1. Change the version information in ``nengo/version.py``.
   See that file for details.

2. Set the release date in the changelog (``CHANGES.rst``).

3. ``git add`` the changes above and make a release commit with

   .. code:: bash

      git commit -m "Release version $(python -c 'import nengo; print(nengo.__version__)')"

4. Review ``git log`` to ensure that the version number is correct; if not,
   then something went wrong with the previous steps.
   Correct these mistakes and amend the release commit accordingly.

5. Tag the release commit with the version number; i.e.,

   .. code:: bash

      git tag -a v$(python -c 'import nengo; print(nengo.__version__)')

   We use annotated tags for the authorship information;
   if you wish to provide a message with information about the release,
   feel free, but it is not necessary.

6. ``git push origin master`` and ``git push origin [tagname]``.

7. Create a release package and upload it to PyPI with

   .. code:: bash

      python setup.py sdist upload

8. Build and upload the documentation with

   .. code:: bash

      python setup.py upload_sphinx

Stage 3: Cleaning up
====================

Nengo's now released!
We need to do a few last things to
put Nengo back in a development state.

1. Change the version information in ``nengo/version.py``.
   See that file for details.

2. Make a new changelog section in ``CHANGES.rst``
   in order to collect changes for the next release.

3. ``git add`` the changes above and make a commit describing
   the current state of the repository and commit with

   .. code:: bash

      git commit -m "Starting development of vX.Y.Z"

4. ``git push origin master``

Stage 4: Announcing
===================

Now we have to let the world know about the new release.
We do this in two ways for each release.

1. Copy the changelog into the tag details on the
   `Github release tab <https://github.com/nengo/nengo/releases>`_.
   Note that the changelog is in reStructuredText,
   while Github expects Markdown.
   Use `Pandoc <http://pandoc.org/try/>`_ or a similar tool
   to convert between the two formats.

2. Write a release announcement.
   Generally, it's easiest to start from
   the last release announcement
   and change it to make sense with the current release
   so that the overall template of each announcement is similar.

All release announcements should be posted
on the `forum <https://forum.nengo.ai/c/general/announcements>`_
and on the `ABR website <http://appliedbrainresearch.com/>`_.
Links to the announcements should be posted
on `Twitter <https://twitter.com/abr_inc>`_.

For major release
(e.g., the first release of a new backend,
or a milestone release like 1.0),
consider writing a more general and
elaborate announcement and posting it to wider-reaching venues, such as
`the comp-neuro mailing list <http://www.tnb.ua.ac.be/mailman/listinfo/comp-neuro>`_,
`Planet SciPy <https://planet.scipy.org/>`_,
and `Planet Python <http://planetpython.org/>`_.
