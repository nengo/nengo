import hashlib
import inspect
import os
import re

import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.neurons import Direct, LIF, LIFRate, RectifiedLinear, Sigmoid
from nengo.rc import rc
from nengo.simulator import Simulator as ReferenceSimulator
from nengo.utils.compat import ensure_bytes, is_string
from nengo.utils.testing import Analytics, Logger, Plotter

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


def recorder_dirname(request, name):
    record = request.config.getvalue(name)
    if is_string(record):
        return record
    elif not record:
        return None

    simulator, nl = ReferenceSimulator, None
    if 'Simulator' in request.funcargnames:
        simulator = request.getfuncargvalue('Simulator')
    if 'nl' in request.funcargnames:
        nl = request.getfuncargvalue('nl')
    elif 'nl_nodirect' in request.funcargnames:
        nl = request.getfuncargvalue('nl_nodirect')

    dirname = "%s.%s" % (simulator.__module__, name)
    if nl is not None:
        dirname = os.path.join(dirname, nl.__name__)
    return dirname


def parametrize_function_name(request, function_name):
    suffixes = []
    if 'parametrize' in request.keywords:
        argnames = [
            x.strip()
            for x in request.keywords['parametrize'].args[0].split(',')]
        for name in argnames:
            value = request.getfuncargvalue(name)
            if inspect.isclass(value):
                value = value.__name__
            suffixes.append('{0}={1}'.format(name, value))
    return '_'.join([function_name] + suffixes)


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
    dirname = recorder_dirname(request, 'plots')
    plotter = Plotter(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: plotter.__exit__(None, None, None))
    return plotter.__enter__()


@pytest.fixture
def analytics(request):
    """an object to store data for analytics.

    Please use this if you're concerned that accuracy or speed may regress.

    This will keep saved data organized in a simulator-specific folder,
    with an automatically generated name. Raw data (for later processing)
    can be saved with ``analytics.add_raw_data``; these will be saved in
    separate compressed ``.npz`` files. Summary data can be saved with
    ``analytics.add_summary_data``; these will be saved
    in a single ``.csv`` file.
    """
    dirname = recorder_dirname(request, 'analytics')
    analytics = Analytics(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: analytics.__exit__(None, None, None))
    return analytics.__enter__()


@pytest.fixture
def analytics_data(request):
    paths = request.config.getvalue('compare')
    function_name = parametrize_function_name(request, re.sub(
        '^test_[a-zA-Z0-9]*_', 'test_', request.function.__name__, count=1))
    return [Analytics.load(
        p, request.module.__name__, function_name) for p in paths]


@pytest.fixture
def logger(request):
    """a logging.Logger object.

    Please use this if your test emits log messages.

    This will keep saved logs organized in a simulator-specific folder,
    with an automatically generated name.
    """
    dirname = recorder_dirname(request, 'logs')
    logger = Logger(
        dirname, request.module.__name__,
        parametrize_function_name(request, request.function.__name__))
    request.addfinalizer(lambda: logger.__exit__(None, None, None))
    return logger.__enter__()


def function_seed(function, mod=0):
    c = function.__code__

    # get function file path relative to Nengo directory root
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    path = os.path.relpath(c.co_filename, start=nengo_path)

    # take start of md5 hash of function file and name, should be unique
    hash_list = os.path.normpath(path).split(os.path.sep) + [c.co_name]
    hash_string = ensure_bytes('/'.join(hash_list))
    i = int(hashlib.md5(hash_string).hexdigest()[:15], 16)
    s = (i + mod) % npext.maxint
    int_s = int(s)  # numpy 1.8.0 bug when RandomState on long type inputs
    assert type(int_s) == int  # should not still be a long because < maxint
    return int_s


@pytest.fixture
def rng(request):
    """a seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """a seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=test_seed)


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [Direct, LIF, LIFRate, RectifiedLinear, Sigmoid])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl_nodirect", [LIF, LIFRate, RectifiedLinear, Sigmoid])


def pytest_runtest_setup(item):
    if not hasattr(item, 'obj'):
        return
    for mark, option, message in [
            ('example', 'noexamples', "examples not requested"),
            ('slow', 'slow', "slow tests not requested")]:
        if getattr(item.obj, mark, None) and not item.config.getvalue(option):
            pytest.skip(message)

    if getattr(item.obj, 'noassertions', None):
        skip = True
        skipreasons = []
        for fixture_name, option, message in [
                ('analytics', 'analytics', "analytics not requested"),
                ('plt', 'plots', "plots not requested"),
                ('logger', 'logs', "logs not requested")]:
            if fixture_name in item.fixturenames:
                if item.config.getvalue(option):
                    skip = False
                else:
                    skipreasons.append(message)
        if skip:
            pytest.skip(" and ".join(skipreasons))


def pytest_collection_modifyitems(session, config, items):
    compare = config.getvalue('compare') is None
    for item in list(items):
        if not hasattr(item, 'obj'):
            continue
        if (getattr(item.obj, 'compare', None) is None) != compare:
            items.remove(item)


def pytest_terminal_summary(terminalreporter):
    reports = terminalreporter.getreports('passed')
    if not reports or terminalreporter.config.getvalue('compare') is None:
        return
    terminalreporter.write_sep("=", "PASSED")
    for rep in reports:
        for name, content in rep.sections:
            terminalreporter.writer.sep("-", name)
            terminalreporter.writer.line(content)
