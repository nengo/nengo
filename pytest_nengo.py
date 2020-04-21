"""A pytest plugin to support backends running the Nengo test suite.

Unsupported tests
-----------------
The ``nengo_test_unsupported`` option allows you to specify Nengo tests
unsupported by a particular simulator.
This is used if you are writing a backend and want to ignore
tests for functions that your backend currently does not support.

Each line represents one test pattern to skip,
and must be followed by a line containing a string in quotes
denoting the reason for skipping the test(s).

The pattern uses
`Unix filename pattern matching
<https://docs.python.org/3/library/fnmatch.html>`_,
including wildcard characters ``?`` and ``*`` to match one or more characters.
The pattern matches to the test name,
which is the same as the pytest ``nodeid``
seen when calling pytest with the ``-v`` argument.

.. code-block:: ini

   nengo_test_unsupported =
       nengo/tests/test_file_path.py::test_function_name
           "This is a message giving the reason we skip this test"
       nengo/tests/test_file_two.py::test_other_thing
           "This is a test with a multi-line reason for skipping.
           Make sure to use quotes around the whole string (and not inside)."
       nengo/tests/test_file_two.py::test_parametrized_thing[param_value]
           "This skips a parametrized test with a specific parameter value."
"""

from fnmatch import fnmatch
import importlib
import shlex
import sys

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # `resource` not available on Windows

import pytest


def is_sim_overridden(config):
    return config.getini("nengo_simulator") != "nengo.Simulator" or config.getini(
        "nengo_simloader"
    )


def is_nengo_test(item):
    return str(item.fspath.pypkgpath()).endswith("nengo")


def deselect_by_condition(condition, items, config):
    remaining = []
    deselected = []
    for item in items:
        if is_nengo_test(item) and condition(item):
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


def load_class(fully_qualified_name):
    mod_name, cls_name = fully_qualified_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def pytest_configure(config):
    if config.getoption("memory") and resource is None:  # pragma: no cover
        raise ValueError("'--memory' option not supported on this platform")

    config.addinivalue_line("markers", "example: Mark a test as an example.")
    config.addinivalue_line(
        "markers", "slow: Mark a test as slow to skip it per default."
    )


def pytest_addoption(parser):
    parser.addoption(
        "--unsupported",
        action="store_true",
        default=False,
        help="Run (with xfail) tests marked as unsupported by this backend.",
    )
    parser.addoption(
        "--noexamples", action="store_true", default=False, help="Do not run examples"
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="Also run slow tests."
    )
    parser.addoption(
        "--spa", action="store_true", default=False, help="Run deprecated SPA tests"
    )
    group = parser.getgroup("terminal reporting", "reporting", after="general")
    group.addoption(
        "--memory",
        action="store_true",
        default=False,
        help="Show memory consumed by Python after all tests are run "
        "(not available on Windows)",
    )

    parser.addini(
        "nengo_simulator", default="nengo.Simulator", help="The simulator class to test"
    )
    parser.addini(
        "nengo_simloader",
        default=None,
        help="A function that returns the simulator class to test",
    )
    parser.addini(
        "nengo_neurons",
        type="linelist",
        default=[
            "nengo.Direct",
            "nengo.LIF",
            "nengo.LIFRate",
            "nengo.RectifiedLinear",
            "nengo.Sigmoid",
            "nengo.SpikingRectifiedLinear",
            "nengo.Tanh",
            "nengo.tests.test_neurons.SpikingTanh",
        ],
        help="Neuron types under test",
    )
    parser.addini(
        "nengo_test_unsupported",
        type="linelist",
        default=[],
        help="List of unsupported unit tests with reason for exclusion",
    )


def pytest_generate_tests(metafunc):
    marks = [
        getattr(pytest.mark, m.name)(*m.args, **m.kwargs)
        for m in getattr(metafunc.function, "pytestmark", [])
    ]

    def mark_nl(nl):
        if nl.__name__ == "Sigmoid":
            nl = pytest.param(
                nl,
                marks=[pytest.mark.filterwarnings("ignore:overflow encountered in exp")]
                + marks,
            )
        return nl

    neuron_types = [load_class(n) for n in metafunc.config.getini("nengo_neurons")]

    if "nl" in metafunc.fixturenames:
        metafunc.parametrize("nl", [mark_nl(nl) for nl in neuron_types])
    if "nl_nodirect" in metafunc.fixturenames:
        nodirect = [mark_nl(n) for n in neuron_types if n.__name__ != "Direct"]
        metafunc.parametrize("nl_nodirect", nodirect)
    if "nl_positive" in metafunc.fixturenames:
        nodirect = [
            mark_nl(n)
            for n in neuron_types
            if n.__name__ != "Direct" and not n.negative
        ]
        metafunc.parametrize("nl_positive", nodirect)


def pytest_collection_modifyitems(session, config, items):
    uses_sim = lambda item: "Simulator" in item.fixturenames
    if is_sim_overridden(config):
        deselect_by_condition(lambda item: not uses_sim(item), items, config)
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
        deselect_by_condition(lambda item: "spa/tests" in item.nodeid, items, config)


def pytest_report_collectionfinish(config, startdir, items):
    if not any(is_nengo_test(item) for item in items):
        return

    deselect_reasons = ["Nengo core tests collected"]
    if is_sim_overridden(config):
        deselect_reasons.append(
            " frontend tests deselected because simulator is not nengo.Simulator"
        )
    if config.getvalue("noexamples"):
        deselect_reasons.append(" example tests deselected (--noexamples passed)")
    if not config.getvalue("slow"):
        deselect_reasons.append(" slow tests skipped (pass --slow to run them)")
    if not config.getvalue("spa"):
        deselect_reasons.append(" spa tests deselected (pass --spa to run them)")

    return deselect_reasons


def pytest_runtest_setup(item):
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

        if is_nengo_test(item) and fnmatch(item.nodeid, test):
            if item.config.getvalue("unsupported"):
                item.add_marker(pytest.mark.xfail(reason=reason))
            else:
                pytest.skip(reason)


def pytest_terminal_summary(terminalreporter):
    if resource and terminalreporter.config.option.memory:
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


@pytest.fixture(scope="session")
def Simulator(request):
    """The Simulator class being tested.

    Please use this, and not ``nengo.Simulator`` directly.
    """

    if request.config.getini("nengo_simloader"):
        # Note: --simloader takes precedence over --simulator.
        # Some backends might specify both for backwards compatibility reasons.
        SimLoader = load_class(request.config.getini("nengo_simloader"))
        return SimLoader(request)
    else:
        return load_class(request.config.getini("nengo_simulator"))
