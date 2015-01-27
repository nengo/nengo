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

2.1.0 (unreleased)
==================

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
