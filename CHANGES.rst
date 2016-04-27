***************
Release History
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - API changes
   - Improvements
   - Behavioural changes
   - Bugfixes
   - Documentation

2.0.4 (unreleased)
==================

2.0.3 (December 7, 2015)
========================

**API changes**

- The ``spa.State`` object replaces the old ``spa.Memory`` and ``spa.Buffer``.
  These old modules are deprecated and will be removed in 2.2.
  (`#796 <https://github.com/nengo/nengo/pull/796>`_)

2.0.2 (October 13, 2015)
========================

2.0.2 is a bug fix release to ensure that Nengo continues
to work with more recent versions of Jupyter
(formerly known as the IPython notebook).

**Behavioural changes**

- The IPython notebook progress bar has to be activated with
  ``%load_ext nengo.ipynb``.
  (`#693 <https://github.com/nengo/nengo/pull/693>`_)

**Improvements**

- Added ``[progress]`` section to ``nengorc`` which allows setting
  ``progress_bar`` and ``updater``.
  (`#693 <https://github.com/nengo/nengo/pull/693>`_)

**Bug fixes**

- Fix compatibility issues with newer versions of IPython,
  and Jupyter. (`#693 <https://github.com/nengo/nengo/pull/693>`_)

2.0.1 (January 27, 2015)
========================

**Behavioural changes**

- Node functions receive ``t`` as a float (instead of a NumPy scalar)
  and ``x`` as a readonly NumPy array (instead of a writeable array).
  (`#626 <https://github.com/nengo/nengo/issues/626>`_,
  `#628 <https://github.com/nengo/nengo/pull/628>`_)

**Improvements**

- ``rasterplot`` works with 0 neurons, and generates much smaller PDFs.
  (`#601 <https://github.com/nengo/nengo/pull/601>`_)

**Bug fixes**

- Fix compatibility with NumPy 1.6.
  (`#627 <https://github.com/nengo/nengo/pull/627>`_)

2.0.0 (January 15, 2015)
========================

Initial release of Nengo 2.0!
Supports Python 2.6+ and 3.3+.
Thanks to all of the contributors for making this possible!
