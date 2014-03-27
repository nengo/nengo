import pytest
import nengo


def pytest_funcarg__Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return nengo.Simulator


def pytest_generate_tests(metafunc):
    if "nl" in metafunc.funcargnames:
        metafunc.parametrize("nl", [nengo.LIF, nengo.LIFRate, nengo.Direct])
    if "nl_nodirect" in metafunc.funcargnames:
        metafunc.parametrize("nl_nodirect", [nengo.LIF, nengo.LIFRate])


def pytest_addoption(parser):
    parser.addoption('--benchmarks', action='store_true', default=False,
                     help='Also run benchmarking tests')
    parser.addoption('--noexamples', action='store_true', default=False,
                     help='Do not run examples')


def pytest_runtest_setup(item):
    if (getattr(item.obj, 'benchmark', None)
            and not item.config.getvalue('benchmarks')):
        pytest.skip('benchmarks not requested')
    if (getattr(item.obj, 'example', None)
            and item.config.getvalue('noexamples')):
        pytest.skip('examples not requested')
