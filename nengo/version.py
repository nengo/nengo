"""Nengo version information.

We use semantic versioning (see http://semver.org/).
Additionally, '-dev' will be added to the version unless the code base
represents a release version. Commits for which the version doesn't have
'-dev' should be git tagged with the version.
"""

name = "nengo"
version_info = (2, 1, 0)  # (major, minor, patch)
dev = True

version = "{v}{dev}".format(v='.'.join(str(v) for v in version_info),
                            dev='-dev' if dev else '')
