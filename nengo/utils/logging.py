from __future__ import absolute_import

import logging
import sys

logger = logging.getLogger(__name__)
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
file_formatter = logging.Formatter(
    ('%(asctime)s [%(levelname)s] %(name)s.%(funcName)s'
     '@ L%(lineno)d\n  %(message)s'))


def log(debug=False, path=None):
    """Log messages.

    If path is None, logging messages will be printed to the console (stdout).
    If it not None, logging messages will be appended to the file at that path.

    Typically someone using Nengo as a library will set up their own
    logging things, and Nengo will just populate their log.
    However, if the user is using Nengo directly, they can use this
    function to get log output.
    """
    level = logging.DEBUG if debug else logging.WARNING
    if logging.root.getEffectiveLevel() > level:
        logging.root.setLevel(level)

    if path is None:
        logger.info("Logging to console")
        for handler in logging.root.handlers:
            if (isinstance(handler, logging.StreamHandler)
                    and handler.formatter == console_formatter):
                break
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(console_formatter)
            logging.root.addHandler(handler)
    else:
        logger.info("Logging to %s", path)
        for handler in logging.root.handlers:
            if (isinstance(handler, logging.FileHandler)
                    and handler.baseFilename == path
                    and handler.formatter == file_formatter):
                break
        else:
            handler = logging.FileHandler(path, encoding='utf-8')
            handler.setFormatter(file_formatter)
            logging.root.addHandler(handler)
    handler.setLevel(level)
    logging.captureWarnings(True)
