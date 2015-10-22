import logging

import pytest

import nengo
import nengo.utils.logging


def test_log_to_console():
    nengo.log(path=None)
    assert logging.root.getEffectiveLevel() == logging.WARNING
    assert nengo.utils.logging.console_handler in logging.root.handlers
    n_handlers = len(logging.root.handlers)
    nengo.log('debug', path=None)
    assert logging.root.getEffectiveLevel() == logging.DEBUG
    assert len(logging.root.handlers) == n_handlers
    logging.root.handlers.remove(nengo.utils.logging.console_handler)


def test_log_to_file(tmpdir):
    tmpfile = str(tmpdir.join("log.txt"))
    nengo.log(path=tmpfile)
    n_handlers = len(logging.root.handlers)
    handler = logging.root.handlers[-1]
    assert logging.root.getEffectiveLevel() == logging.WARNING
    assert isinstance(handler, logging.FileHandler)
    assert handler.baseFilename == tmpfile
    nengo.log('debug', path=tmpfile)
    assert logging.root.getEffectiveLevel() == logging.DEBUG
    assert len(logging.root.handlers) == n_handlers
    logging.root.handlers.remove(handler)

if __name__ == "__main__":
    import sys
    pytest.main(sys.argv)
