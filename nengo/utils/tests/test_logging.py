try:
    # Reload moved to the imp module in Python 3
    from imp import reload
except AttributeError:
    pass
import logging

import pytest

import nengo
import nengo.utils.logging


def test_log_to_console():
    logging.shutdown()
    reload(logging)
    nengo.log(debug=False, path=None)
    assert logging.root.getEffectiveLevel() == logging.WARNING
    assert len(logging.root.handlers) == 1
    handler = logging.root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.formatter == nengo.utils.logging.console_formatter
    nengo.log(debug=True, path=None)
    assert logging.root.getEffectiveLevel() == logging.DEBUG
    assert len(logging.root.handlers) == 1


def test_log_to_file(tmpdir):
    logging.shutdown()
    reload(logging)
    tmpfile = str(tmpdir.join("log.txt"))
    nengo.log(debug=False, path=tmpfile)
    assert logging.root.getEffectiveLevel() == logging.WARNING
    assert len(logging.root.handlers) == 1
    handler = logging.root.handlers[0]
    assert isinstance(handler, logging.FileHandler)
    assert handler.formatter == nengo.utils.logging.file_formatter
    nengo.log(debug=True, path=tmpfile)
    assert logging.root.getEffectiveLevel() == logging.DEBUG
    assert len(logging.root.handlers) == 1


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
