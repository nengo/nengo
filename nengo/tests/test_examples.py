from glob import glob
# import os
import os.path

try:
    from IPython.kernel import KernelManager
except ImportError:
    from IPython.zmq.blockingkernelmanager import (
        BlockingKernelManager as KernelManager)
from IPython.nbformat import current
import pytest


def pytest_generate_tests(metafunc):
    tests = os.path.dirname(os.path.realpath(__file__))
    examples = os.path.realpath(os.path.join(tests, '..', '..', 'examples'))
    examples = glob(examples + '/*.ipynb')
    if "nb_path" in metafunc.funcargnames:
        metafunc.parametrize("nb_path", examples)


def test_noexceptions(nb_path):
    """Ensure that no cells raise an exception."""
    with open(nb_path) as f:
        nb = current.reads(f.read(), 'json')

    km = KernelManager()
    km.start_kernel(stderr=open(os.devnull, 'w'))
    try:
        kc = km.client()
    except AttributeError:
        # IPython 0.13
        kc = km
    kc.start_channels()
    shell = kc.shell_channel
    # Simple ping
    shell.execute("pass")
    shell.get_msg()

    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            shell.execute(cell.input)
            # wait for finish, maximum 2 minutes
            reply = shell.get_msg(timeout=120)['content']
            if reply['status'] == 'error':
                err_msg = ("\nFAILURE:" + cell.input + "\n"
                           "-----\nraised:\n"
                           + "\n".join(reply['traceback']))
                kc.stop_channels()
                km.shutdown_kernel()
                del km
                assert False, err_msg

    kc.stop_channels()
    km.shutdown_kernel()  # noqa
    del km  # noqa
    assert True


def test_nooutput(nb_path):
    """Ensure that no cells have output."""
    # Inspired by gist.github.com/minrk/3719849
    with open(nb_path) as f:
        nb = current.reads(f.read(), 'json')

    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                assert cell.outputs == []


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
