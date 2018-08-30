import hashlib
import inspect
import importlib
import os
import re
from fnmatch import fnmatch
import warnings

import matplotlib
import numpy as np
import pytest

import nengo
import nengo.utils.numpy as npext
from nengo.neurons import (Direct, LIF, LIFRate, RectifiedLinear,
                           Sigmoid, SpikingRectifiedLinear)
from nengo.rc import rc
from nengo.utils.compat import ensure_bytes, is_string
from nengo.utils.testing import Analytics, Logger, Plotter


class TestConfig(object):
    """Parameters affecting all Nengo tests.

    These are essentially global variables used by py.test to modify aspects
    of the Nengo tests. We collect them in this class to provide a
    mini namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    test_seed = 0  # changing this will change seeds for all tests
    Simulator = nengo.Simulator
    RefSimulator = nengo.Simulator
    neuron_types = [
        Direct, LIF, LIFRate, RectifiedLinear, Sigmoid, SpikingRectifiedLinear
    ]
    compare_requested = False

    @classmethod
    def is_sim_overridden(cls):
        return cls.Simulator is not nengo.Simulator

    @classmethod
    def is_refsim_overridden(cls):
        return cls.RefSimulator is not nengo.Simulator

    @classmethod
    def is_skipping_frontend_tests(cls):
        return cls.is_sim_overridden() or cls.is_refsim_overridden()


def pytest_configure(config):
    matplotlib.use('agg')
    warnings.simplefilter('always')

    if config.getoption('simulator'):
        TestConfig.Simulator = load_class(config.getoption('simulator')[0])
    if config.getoption('ref_simulator'):
        refsim = config.getoption('ref_simulator')[0]
        TestConfig.RefSimulator = load_class(refsim)

    if config.getoption('neurons'):
        ntypes = config.getoption('neurons')[0].split(',')
        TestConfig.neuron_types = [load_class(n) for n in ntypes]

    if config.getoption('seed_offset'):
        TestConfig.test_seed = config.getoption('seed_offset')[0]

    TestConfig.compare_requested = config.getvalue('compare') is not None


def load_class(fully_qualified_name):
    mod_name, cls_name = fully_qualified_name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


@pytest.fixture(scope="session")
def Simulator(request):
    """The Simulator class being tested.

    Please use this, and not ``nengo.Simulator`` directly. If the test is
    reference simulator specific, then use ``RefSimulator`` below.
    """
    return TestConfig.Simulator


@pytest.fixture(scope="session")
def RefSimulator(request):
    """The reference simulator.

    Please use this if the test is reference simulator specific.
    Other simulators may choose to implement the same API as the
    reference simulator; this allows them to test easily.
    """
    return TestConfig.RefSimulator


def recorder_dirname(request, name):
    """Returns the directory to put test artifacts in.

    Test artifacts produced by Nengo include plots and analytics.

    Note that the return value might be None, which indicates that the
    artifacts should not be saved.
    """
    record = request.config.getvalue(name)
    if is_string(record):
        return record
    elif not record:
        return None

    simulator, nl = TestConfig.RefSimulator, None
    if 'Simulator' in request.funcargnames:
        simulator = request.getfixturevalue('Simulator')
    # 'nl' stands for the non-linearity used in the neuron equation
    if 'nl' in request.funcargnames:
        nl = request.getfixturevalue('nl')
    elif 'nl_nodirect' in request.funcargnames:
        nl = request.getfixturevalue('nl_nodirect')

    dirname = "%s.%s" % (simulator.__module__, name)
    if nl is not None:
        dirname = os.path.join(dirname, nl.__name__)
    return dirname


def parametrize_function_name(request, function_name):
    """Creates a unique name for a test function.

    The unique name accounts for values passed through
    ``pytest.mark.parametrize``.

    This function is used when naming plots saved through the ``plt`` fixture.
    """
    suffixes = []
    if 'parametrize' in request.keywords:
        argnames = request.keywords['parametrize'].args[::2]
        argnames = [x.strip() for names in argnames for x in names.split(',')]
        for name in argnames:
            value = request.getfixturevalue(name)
            if inspect.isclass(value):
                value = value.__name__
            suffixes.append('{}={}'.format(name, value))
    return '+'.join([function_name] + suffixes)


@pytest.fixture
def plt(request):
    """A pyplot-compatible plotting interface.

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
    """An object to store data for analytics.

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
    """A logging.Logger object.

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
    """Generates a unique seed for the given test function.

    The seed should be the same across all machines/platforms.
    """
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


def get_item_name(item):
    """Get a unique backend-independent name for an item (test function)."""
    item_abspath, item_name = str(item.fspath), item.location[2]
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    item_relpath = os.path.relpath(item_abspath, start=nengo_path)
    item_relpath = os.path.join('nengo', item_relpath)
    item_relpath = item_relpath.replace(os.sep, '/')
    return '%s:%s' % (item_relpath, item_name)


@pytest.fixture
def rng(request):
    """A seeded random number generator.

    This should be used in lieu of np.random because we control its seed.
    """
    # add 1 to seed to be different from `seed` fixture
    seed = function_seed(request.function, mod=TestConfig.test_seed + 1)
    return np.random.RandomState(seed)


@pytest.fixture
def seed(request):
    """A seed for seeding Networks.

    This should be used in lieu of an integer seed so that we can ensure that
    tests are not dependent on specific seeds.
    """
    return function_seed(request.function, mod=TestConfig.test_seed)


def pytest_generate_tests(metafunc):
    marks = [
        getattr(pytest.mark, m.name)(*m.args, **m.kwargs)
        for m in getattr(metafunc.function, 'pytestmark', [])]

    def mark_nl(nl):
        if nl is Sigmoid:
            nl = pytest.param(nl, marks=[pytest.mark.filterwarnings(
                'ignore:overflow encountered in exp')] + marks)
        return nl

    if "nl" in metafunc.funcargnames:
        metafunc.parametrize(
            "nl", [mark_nl(nl) for nl in TestConfig.neuron_types])
    if "nl_nodirect" in metafunc.funcargnames:
        nodirect = [mark_nl(n) for n in TestConfig.neuron_types
                    if n is not Direct]
        metafunc.parametrize("nl_nodirect", nodirect)


def pytest_runtest_setup(item):  # noqa: C901
    rc.reload_rc([])
    rc.set('decoder_cache', 'enabled', 'False')
    rc.set('exceptions', 'simplified', 'False')

    if not hasattr(item, 'obj'):
        return  # Occurs for doctests, possibly other weird tests

    conf = item.config
    test_uses_compare = getattr(item.obj, 'compare', None) is not None
    test_uses_sim = 'Simulator' in item.fixturenames
    test_uses_refsim = 'RefSimulator' in item.fixturenames
    tests_frontend = not (test_uses_sim or test_uses_refsim)

    if getattr(item.obj, 'example', None) and not conf.getvalue('noexamples'):
        pytest.skip("examples not requested")
    elif getattr(item.obj, 'slow', None) and not conf.getvalue('slow'):
        pytest.skip("slow tests not requested")
    elif not TestConfig.compare_requested and test_uses_compare:
        pytest.skip("compare tests not requested")
    elif TestConfig.is_skipping_frontend_tests() and tests_frontend:
        pytest.skip("frontend tests not run for alternate backends")
    elif (TestConfig.is_skipping_frontend_tests()
          and test_uses_refsim
          and not TestConfig.is_refsim_overridden()):
        pytest.skip("RefSimulator not overridden")
    elif (TestConfig.is_skipping_frontend_tests()
          and test_uses_sim
          and not TestConfig.is_sim_overridden()):
        pytest.skip("Simulator not overridden")
    elif getattr(item.obj, 'noassertions', None):
        options = []
        for fixture, option in [('analytics', 'analytics'),
                                ('plt', 'plots'),
                                ('logger', 'logs')]:
            if fixture in item.fixturenames and not conf.getvalue(option):
                options.append(option)
        if len(options) > 0:
            pytest.skip("%s not requested" % " and ".join(options))

    if not tests_frontend:
        item_name = get_item_name(item)

        for test, reason in TestConfig.Simulator.unsupported:
            # We add a '*' before test to eliminate the surprise of needing
            # a '*' before the name of a test function.
            if fnmatch(item_name, '*' + test):
                pytest.xfail(reason)


def determine_run_stats(terminalreporter):
    non_runnables = {
        "compare tests not requested",
        "frontend tests not run for alternate backends",
        "RefSimulator not overridden",
        "Simulator not overridden"
    }
    n_ran = 0
    n_runnables = 0
    for key in terminalreporter.stats:
        if key != '':
            if key == 'skipped':
                for report in terminalreporter.stats[key]:
                    reason = report.longrepr[-1]
                    if reason.startswith('Skipped: '):
                        reason = reason[9:]
                    if reason not in non_runnables:
                        n_runnables += 1
            else:
                n = len(terminalreporter.stats[key])
                n_ran += n
                n_runnables += n
    return n_ran, n_runnables


def pytest_terminal_summary(terminalreporter):
    n_ran, n_runnables = determine_run_stats(terminalreporter)
    if n_ran == n_runnables:
        line = "Ran all {} runnable tests.".format(n_ran)
    else:
        line = "Ran {} of {} runnable tests.".format(n_ran, n_runnables)
    terminalreporter.write_sep("=", line, bold=True)
    if TestConfig.compare_requested:
        terminalreporter.writer.line(
            "Non-compare tests run only without --compare option.")
    else:
        terminalreporter.writer.line(
            "Compare tests require --compare option.")
    if TestConfig.is_skipping_frontend_tests():
        terminalreporter.writer.line(
            "Front-end tests only runnable for reference backend.")

    reports = terminalreporter.getreports('passed')
    if not reports or terminalreporter.config.getvalue('compare') is None:
        return
    terminalreporter.write_sep("=", "PASSED")
    for rep in reports:
        for name, content in rep.sections:
            terminalreporter.writer.sep("-", name)
            terminalreporter.writer.line(content)
