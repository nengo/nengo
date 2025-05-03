"""Nengo version information.

We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""

name = "nengo"
version_info = (3, 1, 0)  # (major, minor, patch)
dev = None
post = 1

version = "{v}{dev}{post}".format(
    v=".".join(str(v) for v in version_info),
    dev=(".dev%d" % dev) if dev is not None else "",
    post=(".post%d" % post) if post is not None else "",
)
