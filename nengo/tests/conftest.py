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


def pytest_runtest_setup(item):
    # Clear the context before every test so that any Model() creation is
    # treated as a top-level declaration.
    nengo.context.clear()
