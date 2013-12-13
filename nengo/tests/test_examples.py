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


@pytest.mark.xfail
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

    cells = 0
    failures = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue
            shell.execute(cell.input)
            # wait for finish, maximum 60s
            reply = shell.get_msg(timeout=60)['content']
            if reply['status'] == 'error':
                failures += 1
                print("\nFAILURE:")
                print(cell.input)
                print('-----')
                print("raised:")
                print('\n'.join(reply['traceback']))
            cells += 1

    kc.stop_channels()
    km.shutdown_kernel()
    del km
    assert failures == 0


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
