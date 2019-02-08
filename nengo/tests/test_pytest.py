import nengo.utils.numpy as npext
from nengo.conftest import TestConfig

pytest_plugins = ["pytester"]


def test_seed_fixture(seed):
    """The seed should be the same on all machines"""
    i = (seed - TestConfig.test_seed) % npext.maxint
    assert i == 1832276344


def test_unsupported(testdir):
    testdir.makeconftest(
        """
        import nengo.conftest

        class MockSimulator(object):
            unsupported = [('*', 'mock simulator')]

        nengo.conftest.TestConfig.Simulator = MockSimulator
        """)
    outcomes = testdir.runpytest(
        "-p", "nengo.tests.options", "--pyargs", "nengo",
    ).parseoutcomes()
    assert outcomes["skipped"] > 350
    assert outcomes["deselected"] > 700
    assert "passed" not in outcomes
    assert "failed" not in outcomes

    outcomes = testdir.runpytest(
        "-p", "nengo.tests.options", "--pyargs", "nengo", "--unsupported",
    ).parseoutcomes()
    assert outcomes["xfailed"] > 350
    assert outcomes["deselected"] > 700
    assert "passed" not in outcomes
    assert "failed" not in outcomes
