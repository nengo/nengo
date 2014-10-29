r"""This modules provides access to the Nengo RC settings.

Nengo RC settings will be read from the following files with precedence
to those listed first:
1. ``nengorc`` in the current directory.
2. An operating system specific file in the user's home directory.
   Windows: ``%userprofile%\.nengo\nengorc``
   Other (OS X, Linux): ``~/.config/nengo/nengorc``
3. ``INSTALL/nengo-data/nengorc``  (where INSTALL is the installation directory
    of the Nengo package)

The RC file is divided into sections by lines containing the section name
in brackets, i.e. ``[section]``. A setting is set by giving the name followed
by a ``:`` or ``=`` and the value. All lines starting with ``#`` or ``;`` are
comments.

Example
-------

This example demonstrates how to set settings in an RC file:

    [decoder_cache]
    size: 536870912  # setting the decoder cache size to 512MiB.
"""

import logging

import nengo.utils.paths
from nengo.utils.compat import configparser

logger = logging.getLogger(__name__)

# The default core Nengo RC settings. Access with
#   nengo.RC_DEFAULTS[section_name][option_name]
RC_DEFAULTS = {
    'decoder_cache': {
        'enabled': True,
        'readonly': False,
        'size': '512 MB',
        'path': nengo.utils.paths.decoder_cache_dir
    }
}

# The RC files in the order in which they will be read.
RC_FILES = [nengo.utils.paths.nengorc['system'],
            nengo.utils.paths.nengorc['user'],
            nengo.utils.paths.nengorc['project']]


class _RC(configparser.SafeConfigParser):
    """Allows reading from and writing to Nengo RC settings."""

    def __init__(self):
        # configparser uses old-style classes without 'super' support
        configparser.SafeConfigParser.__init__(self)
        self.reload_rc()

    def _clear(self):
        self.remove_section(configparser.DEFAULTSECT)
        for s in self.sections():
            self.remove_section(s)

    def _init_defaults(self):
        for section, settings in RC_DEFAULTS.items():
            self.add_section(section)
            for k, v in settings.items():
                self.set(section, k, str(v))

    def readfp(self, fp, filename=None):
        if filename is None:
            if hasattr(fp, 'name'):
                filename = fp.name
            else:
                filename = '<???>'
        logger.info('Reading configuration from {0}'.format(filename))
        return configparser.SafeConfigParser.readfp(self, fp, filename)

    def read(self, filenames):
        logger.info('Reading configuration files {0}'.format(filenames))
        return configparser.SafeConfigParser.read(self, filenames)

    def reload_rc(self, filenames=None):
        """Resets the currently loaded RC settings and loads new RC files.

        Parameters
        ----------
        filenames: iterable object
            Filenames of RC files to load.
        """
        if filenames is None:
            filenames = RC_FILES

        self._clear()
        self._init_defaults()
        self.read(filenames)

# The current Nengo RC settings.
rc = _RC()
