import hashlib

import numpy as np
import pytest

import nengo.utils.numpy as npext
from nengo.neurons import LIF, LIFRate, Direct
from nengo.rc import rc
from nengo.simulator import Simulator as ReferenceSimulator
from nengo.utils.compat import ensure_bytes
from nengo.utils.testing import Plotter

test_seed = 0  # changing this will change seeds for all tests


def pytest_configure(config):
    rc.reload_rc([])
    rc.set('decoder_cache', 'enabled', 'false')


@pytest.fixture(scope="session")
def Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return ReferenceSimulator


@pytest.fixture(scope="session")
def RefSimulator(request):
    """the reference simulator.

    Please use this if the test is reference simulator specific.
    Other simulators may choose to implement the same API as the
    reference simulator; this allows them to test easily.
    """
    return ReferenceSimulator


@pytest.fixture
def plt(request):
    """a pyplot-compatible plotting interface.

    Please use this if your test creates plots.

    This will keep saved plots organized in a simulator-specific folder,
    with an automatically generated name. savefig() and close() will
    automatically be called when the test function completes.

    If you need to override the default filename, set `plt.saveas` to
    the desired filename.
    """
    simulator, nl = ReferenceSimulator, None
    if 'Simulator' in request.funcargnames:
        simulator = request.getfuncargvalue('Simulator')
    if 'nl' in request.funcargnames:
        nl = request.getfuncargvalue('nl')
    elif 'nl_nodirect' in request.funcargnames:
        nl = request.getfuncargvalue('nl_nodirect')
    plotter = Plotter(simulator, request.module, request.function, nl=nl)
    request.addfinalizer(lambda p=plotter: p.__exit__(None, None, None))
    return plotter.__enter__()


def function_seed(function, mod=0):
    # take start of md5 hash of function file and name, should be pretty random
    c = function.__code__
    to_hash = ensure_bytes(c.co_filename + c.co_name)
    i = int(hashlib.md5(to_hash).hexdigest()[:15], 16)
    return (test_seed + i + mod) % npext.maxint


@pytest.fixture
def rng(request):
    """a seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from network seed
    seed = function_seed(request.function, mod=1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """a seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function)


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize("nl", [LIF, LIFRate, Direct])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize("nl_nodirect", [LIF, LIFRate])


def pytest_addoption(parser):
    parser.addoption('--benchmarks', action='store_true', default=False,
                     help='Also run benchmarking tests')
    parser.addoption('--noexamples', action='store_false', default=True,
                     help='Do not run examples')
    parser.addoption(
        '--optional', action='store_true', default=False,
        help='Also run optional tests that may use optional packages')


def pytest_runtest_setup(item):
    for mark, option, message in [
            ('benchmark', 'benchmarks', "benchmarks not requested"),
            ('example', 'noexamples', "examples not requested"),
            ('optional', 'optional', "optional tests not requested")]:
        if getattr(item.obj, mark, None) and not item.config.getvalue(option):
            pytest.skip(message)
