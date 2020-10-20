r"""Certain features of Nengo can be configured globally through RC settings.

RC settings can be manipulated either through the ``nengo.rc`` object,
or through RC configuration files.

The ``nengo.rc`` object
=======================

The ``nengo.rc`` object gives programmatic access to
globally configured features of Nengo.

.. autodata:: nengo.rc

Configuration files
===================

``nengo.rc`` is initialized with configuration settings read
from the following files with precedence to those listed first:

1. ``nengorc`` in the current directory. This is intended to allow for project
   specific settings without hard coding them in the model script.

2. An operating system specific file in the user's home directory.

   * Windows: ``%userprofile%\.nengo\nengorc``

   * Other (OS X, Linux): ``~/.config/nengo/nengorc``

3. ``INSTALL/nengo-data/nengorc`` (where ``INSTALL`` is the
   installation directory of the Nengo package).

The RC file is divided into sections by lines containing the section name
in brackets, i.e. ``[section]``. A setting is set by giving the name followed
by a ``:`` or ``=`` and the value. All lines starting with ``#`` or ``;`` are
comments.

For example, to set the size of the decoder cache to 512 MB,
add the following to a configuration file:

.. code-block:: ini

   [decoder_cache]
   size = 512 MB

Configuration options
=====================

All of the configuration options are listed in the example
configuration file, which is included with Nengo
and copied below.

Commented lines show the default values for each setting.

.. _nengorc:

.. include:: ../nengo-data/nengorc
   :literal:
   :start-line: 29

"""

import logging
from configparser import DEFAULTSECT, ConfigParser

import numpy as np

import nengo.utils.paths

logger = logging.getLogger(__name__)

# The default core Nengo RC settings. Access with
#   nengo.RC_DEFAULTS[section_name][option_name]
RC_DEFAULTS = {
    "precision": {"bits": 64},
    "decoder_cache": {
        "enabled": True,
        "readonly": False,
        "size": "512 MB",
        "path": nengo.utils.paths.decoder_cache_dir,
    },
    "progress": {"progress_bar": "auto"},
    "exceptions": {"simplified": True},
    "nengo.Simulator": {"fail_fast": False},
}

# The RC files in the order in which they will be read.
RC_FILES = [
    nengo.utils.paths.nengorc["system"],
    nengo.utils.paths.nengorc["user"],
    nengo.utils.paths.nengorc["project"],
]


class _RC(ConfigParser):  # pylint: disable=too-many-ancestors
    """Allows reading from and writing to Nengo RC settings.

    This object is a :class:`configparser.ConfigParser`, which means that
    values can be accessed and manipulated like a dictionary:

    .. testcode::

       oldsize = nengo.rc["decoder_cache"]["size"]
       nengo.rc["decoder_cache"]["size"] = "2 GB"

    All values are stored as strings. If you want to store or retrieve a
    specific datatype, you should coerce it appropriately (e.g., with ``int()``).
    Booleans are more flexible, so you should use the ``getboolean`` method
    to access boolean values.

    .. testcode::

       simple = nengo.rc["exceptions"].getboolean("simplified")

    In addition to the normal :class:`configparser.ConfigParser` methods,
    this object also has a ``reload_rc`` method to reset ``nengo.rc``
    to default settings:

    .. testcode::

       nengo.rc.reload_rc()  # Reads defaults from configuration files
       nengo.rc.reload_rc(filenames=[])  # Ignores configuration files

    """

    def __init__(self):
        super().__init__()
        self.reload_rc()

    @property
    def float_dtype(self):
        bits = self.get("precision", "bits")
        return np.dtype(f"float{bits}")

    @property
    def int_dtype(self):
        bits = self.get("precision", "bits")
        return np.dtype(f"int{bits}")

    def _clear(self):
        self.remove_section(DEFAULTSECT)
        for s in self.sections():
            self.remove_section(s)

    def _init_defaults(self):
        for section, settings in RC_DEFAULTS.items():
            self.add_section(section)
            for k, v in settings.items():
                self.set(section, k, str(v))

    def read_file(self, fp, filename=None):
        if filename is None:
            filename = fp.name if hasattr(fp, "name") else "<???>"

        logger.debug("Reading configuration from %s", filename)
        return super().read_file(fp, filename)

    def read(self, filenames):
        logger.debug("Reading configuration files %s", filenames)
        return super().read(filenames)

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


rc = _RC()
