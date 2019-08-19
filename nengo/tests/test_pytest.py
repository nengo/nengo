import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("xfail", (True, False))
def test_unsupported(xfail, testdir):
    """Test `nengo_test_unsupported` config option and `--unsupported` arg"""
    # Create a test file with some dummy tests
    testdir.makefile(
        ".py",
        test_file="""
        import pytest

        @pytest.mark.parametrize("param", (True, False))
        def test_unsupported(param):
            print("test_unsupported param=%s ran" % param)
            assert param

        @pytest.mark.parametrize("param", (True, False))
        def test_unsupported_all(param):
            print("test_unsupported_all param=%s ran" % param)
            assert False

        def test_supported():
            print("test_supported ran")
            assert True
        """,
    )

    # Create the .ini file to skip/xfail the failing tests. This will
    # make sure square brackets for parameters just skip that parametrization.
    # We also make sure that both single-line and multiline comments work.
    testdir.makefile(
        ".ini",
        pytest="""
        [pytest]
        nengo_test_unsupported =
            test_file.py:test_unsupported[False]
                "One unsupported param
                with multiline comment"
            test_file.py:test_unsupported_all*
                "Two unsupported params with single-line comment"
        """,
    )

    testdir.makefile(
        ".py",
        conftest="""
        from nengo.conftest import pytest_runtest_setup, pytest_configure
        """,
    )

    args = "-p nengo.tests.options -rsx -sv".split()
    if xfail:
        args.append("--unsupported")
    output = testdir.runpytest_subprocess(*args)

    # ensure that these lines appear somewhere in the output
    output.stdout.fnmatch_lines_random(
        [
            "*One unsupported param with multiline comment",
            "*Two unsupported params with single-line comment",
            "*test_supported ran",
            "*test_unsupported param=True ran",
        ]
    )

    # if `--unsupported`, unsupported tests run and xfail, otherwise they skip
    outcomes = output.parseoutcomes()
    if xfail:
        output.stdout.fnmatch_lines_random(
            [
                "*test_unsupported param=False ran",
                "*test_unsupported_all param=True ran",
                "*test_unsupported_all param=False ran",
            ]
        )
        assert outcomes["xfailed"] == 3
        assert "skipped" not in outcomes
    else:
        assert outcomes["skipped"] == 3
        assert "xfailed" not in outcomes
    assert "failed" not in outcomes
    assert outcomes["passed"] == 2


def test_pyargs(testdir):
    # create a simulator that does not support any tests
    # (we don't actually want to run all the tests)
    testdir.makeconftest(
        """
        import nengo.conftest

        class MockSimulator:
            pass

        nengo.conftest.TestConfig.SimLoader = lambda request: MockSimulator
        """
    )

    # mark all the tests as unsupported
    testdir.makefile(
        ".ini",
        pytest="""
        [pytest]
        nengo_test_unsupported =
            *
                "Using mock simulator"
        """,
    )

    outcomes = testdir.runpytest_subprocess(
        "-p", "nengo.tests.options", "--pyargs", "nengo"
    ).parseoutcomes()

    assert "failed" not in outcomes
    assert "passed" not in outcomes
    assert outcomes["skipped"] > 350
    assert outcomes["deselected"] > 700
