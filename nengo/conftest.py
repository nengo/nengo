from fnmatch import fnmatch
import importlib
import os

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # `resource` not available on Windows
import shlex
import sys
import warnings

import pytest
from pytest_allclose import report_rmses

import nengo
from nengo.neurons import (
    Direct,
    LIF,
    LIFRate,
    RectifiedLinear,
    Sigmoid,
    SpikingRectifiedLinear,
)
from nengo.rc import rc


class TestConfig:
    """Parameters affecting all Nengo tests.

    These are essentially global variables used by py.test to modify aspects
    of the Nengo tests. We collect them in this class to provide a
    mini namespace and to avoid using the ``global`` keyword.

    The values below are defaults. The functions in the remainder of this
    module modify these values accordingly.
    """

    Simulator = nengo.Simulator
    RefSimulator = nengo.Simulator
    neuron_types = [
        Direct,
        LIF,
        LIFRate,
        RectifiedLinear,
        Sigmoid,
        SpikingRectifiedLinear,
    ]
    run_unsupported = False

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
    warnings.simplefilter("always")

    if config.getoption("memory") and resource is None:  # pragma: no cover
        raise ValueError("'--memory' option not supported on this platform")

    if config.getoption("simulator"):
        TestConfig.Simulator = load_class(config.getoption("simulator")[0])
    if config.getoption("ref_simulator"):
        refsim = config.getoption("ref_simulator")[0]
        TestConfig.RefSimulator = load_class(refsim)

    if config.getoption("neurons"):
        ntypes = config.getoption("neurons")[0].split(",")
        TestConfig.neuron_types = [load_class(n) for n in ntypes]

    TestConfig.run_unsupported = config.getvalue("unsupported")


def load_class(fully_qualified_name):
    mod_name, cls_name = fully_qualified_name.rsplit(".", 1)
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


def get_item_name(item):
    """Get a unique backend-independent name for an item (test function)."""
    item_path, item_name = str(item.fspath), item.location[2]
    nengo_path = os.path.abspath(os.path.dirname(nengo.__file__))
    if item_path.startswith(nengo_path):
        # if test is within the nengo directory, remove the nengo directory
        # prefix (so that we can move the nengo directory without changing
        # the name)
        item_path = item_path[len(nengo_path) + 1 :]
    item_path = os.path.join("nengo", item_path)
    item_path = item_path.replace(os.sep, "/")
    return "%s:%s" % (item_path, item_name)


def pytest_generate_tests(metafunc):
    marks = [
        getattr(pytest.mark, m.name)(*m.args, **m.kwargs)
        for m in getattr(metafunc.function, "pytestmark", [])
    ]

    def mark_nl(nl):
        if nl is Sigmoid:
            nl = pytest.param(
                nl,
                marks=[pytest.mark.filterwarnings("ignore:overflow encountered in exp")]
                + marks,
            )
        return nl

    if "nl" in metafunc.fixturenames:
        metafunc.parametrize("nl", [mark_nl(nl) for nl in TestConfig.neuron_types])
    if "nl_nodirect" in metafunc.fixturenames:
        nodirect = [mark_nl(n) for n in TestConfig.neuron_types if n is not Direct]
        metafunc.parametrize("nl_nodirect", nodirect)


def pytest_collection_modifyitems(session, config, items):
    if config.getvalue("noexamples"):
        deselect_by_condition(
            lambda item: item.get_closest_marker("example"), items, config
        )
    if not config.getvalue("slow"):
        skip_slow = pytest.mark.skip("slow tests not requested")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)
    if not config.getvalue("spa"):
        deselect_by_condition(
            lambda item: "nengo/spa/" in get_item_name(item), items, config
        )

    uses_sim = lambda item: "Simulator" in item.fixturenames
    uses_refsim = lambda item: "RefSimulator" in item.fixturenames
    if TestConfig.is_skipping_frontend_tests():
        deselect_by_condition(
            lambda item: not (uses_sim(item) or uses_refsim(item)), items, config
        )
        deselect_by_condition(
            lambda item: uses_refsim(item) and not TestConfig.is_refsim_overridden(),
            items,
            config,
        )
        deselect_by_condition(
            lambda item: uses_sim(item) and not TestConfig.is_sim_overridden(),
            items,
            config,
        )


def deselect_by_condition(condition, items, config):
    remaining = []
    deselected = []
    for item in items:
        if condition(item):
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def pytest_report_collectionfinish(config, startdir, items):
    deselect_reasons = ["Nengo core tests collected"]

    if config.getvalue("noexamples"):
        deselect_reasons.append(" example tests deselected (--noexamples passed)")
    if not config.getvalue("slow"):
        deselect_reasons.append(" slow tests skipped (pass --slow to run them)")
    if not config.getvalue("spa"):
        deselect_reasons.append(" spa tests deselected (pass --spa to run them)")

    if TestConfig.is_skipping_frontend_tests():
        deselect_reasons.append(
            " frontend tests deselected because --simulator or "
            "--ref-simulator was passed"
        )
        if not TestConfig.is_refsim_overridden():
            deselect_reasons.append(
                " backend tests for non-reference simulator deselected "
                "because only --ref-simulator was passed"
            )
        if not TestConfig.is_sim_overridden():
            deselect_reasons.append(
                " backend tests for reference simulator deselected "
                "because only --simulator was passed"
            )

    return deselect_reasons


def pytest_runtest_setup(item):
    rc.reload_rc([])
    rc.set("decoder_cache", "enabled", "False")
    rc.set("exceptions", "simplified", "False")
    rc.set("nengo.Simulator", "fail_fast", "True")

    item_name = get_item_name(item)

    # join all the lines and then split (preserving quoted strings)
    unsupported = shlex.split(" ".join(item.config.getini("nengo_test_unsupported")))
    # group pairs (representing testname + reason)
    unsupported = [unsupported[i : i + 2] for i in range(0, len(unsupported), 2)]

    for test, reason in unsupported:
        # wrap square brackets to interpret them literally
        # (see https://docs.python.org/3/library/fnmatch.html)
        test = "".join("[%s]" % c if c in ("[", "]") else c for c in test)

        # We add a '*' before test to eliminate the surprise of needing
        # a '*' before the name of a test function.
        test = "*" + test

        if fnmatch(item_name, test):
            if TestConfig.run_unsupported:
                item.add_marker(pytest.mark.xfail(reason=reason))
            else:
                pytest.skip(reason)


def report_memory_usage(terminalreporter):
    # Calculate memory usage; details at
    # http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/  # noqa, pylint: disable=line-too-long
    rusage_denom = 1024.0
    if sys.platform == "darwin":  # pragma: no cover
        # ... it seems that in OSX the output is in different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    terminalreporter.write_sep("=", "total memory consumed: %.2f MiB" % mem)

    # Ensure we only print once
    terminalreporter.config.option.memory = False


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)
    if resource and terminalreporter.config.option.memory:
        report_memory_usage(terminalreporter)
