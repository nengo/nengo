"""Nengo version information.

We use semantic versioning (see http://semver.org/), with an additional tag
for release candidates. Our convention is as follows:

- No tag for a release version (2.0.0)
- 'rcX' for a release candidate, where 'X' is the number of release candidate.
  (2.0.0-rc1)
- No post-release tags; increment patch number instead (2.0.1)

Additionally, '-dev' will be added to the version unless the code base
represents a release version. Commits for which the version doesn't have
'-dev' should be git tagged with the version.

"""

version_info = (2, 0, 0)  # (major, minor, patch)
rc = None
dev = True

version = "{0}{1}{2}".format('.'.join(str(v) for v in version_info),
                             '-rc{0:d}'.format(rc) if rc is not None else '',
                             '-dev' if dev else '')
