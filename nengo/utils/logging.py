from __future__ import absolute_import

import logging
import sys

from .compat import TextIO

console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_formatter = logging.Formatter(
    ('%(asctime)s [%(levelname)s] %(name)s.%(funcName)s'
     '@ L%(lineno)d\n  %(message)s'))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)


def log(level='warning', path=None):
    """Log messages.

    If path is None, logging messages will be printed to the console (stdout).
    If it not None, logging messages will be appended to the file at that path.

    Typically someone using Nengo as a library will set up their own
    logging things, and Nengo will just populate their log.
    However, if the user is using Nengo directly, they can use this
    function to get log output.

    Parameters
    ----------
    level : string (optional)
        Collect messages targeting this level and above. The levels from
        highest to lowest are: 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'.
    path : string (optional)
        Path of a file to append log messages to. If ``None`` (default),
        messages are logged to the console.
    """
    if level.upper() not in ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'):
        raise ValueError("Invalid logging level")

    level = getattr(logging, level.upper())
    logging.root.setLevel(level)

    if path is None:
        handler = console_handler
    else:
        for handler in logging.root.handlers:
            if (isinstance(handler, logging.FileHandler)
                    and handler.baseFilename == path
                    and handler.formatter == file_formatter):
                break
        else:
            handler = logging.FileHandler(path, encoding='utf-8')
            handler.setFormatter(file_formatter)

    if handler not in logging.root.handlers:
        logging.root.addHandler(handler)
    handler.setLevel(level)
    logging.captureWarnings(True)


class CaptureLogHandler(logging.StreamHandler):
    """A logging handler that stores log records and the log text."""

    def __init__(self):
        """Creates a new log handler."""
        logging.StreamHandler.__init__(self)
        self.records = []
        self.stream = TextIO()

    def close(self):
        """Close this log handler and its underlying stream."""
        self.stream.close()
        logging.StreamHandler.close(self)

    def emit(self, record):
        """Keep the log records in a list in addition to the log text."""
        self.records.append(record)
        logging.StreamHandler.emit(self, record)
