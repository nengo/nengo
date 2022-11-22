import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("xfail", (True, False))
def test_unsupported(xfail, pytester):
    """Test ``nengo_test_unsupported`` config option and ``--unsupported`` arg."""

    # Dir ends with 'nengo' so that `pytest_nengo.is_nengo_test` returns True
    test_dir = pytester.mkpydir("notnengo")

    # Create a test file with some dummy tests, and move it into test_dir
    test_file_path = pytester.makefile(
        ".py",
        test_file="""
        import pytest

        @pytest.mark.parametrize("param", (True, False))
        def test_unsupported(param):
            print(f"test_unsupported param={param} ran")
            assert param

        @pytest.mark.parametrize("param", (True, False))
        def test_unsupported_all(param):
            print(f"test_unsupported_all param={param} ran")
            assert False

        def test_supported():
            print("test_supported ran")
            assert True
        """,
    )
    test_file_path.rename(test_dir / test_file_path.name)

    # Create the .ini file to skip/xfail the failing tests. This will
    # make sure square brackets for parameters just skip that parametrization.
    # We also make sure that both single-line and multiline comments work.
    pytester.makefile(
        ".ini",
        pytest="""
        [pytest]
        nengo_test_unsupported =
            test_file.py::test_unsupported[False]
                "One unsupported param
                with multiline comment"
            test_file.py::test_unsupported_all*
                "Two unsupported params with single-line comment"
        """,
    )

    args = f"{test_dir} -rsx -sv".split()
    if xfail:
        args.append("--unsupported")
    output = pytester.runpytest_inprocess(*args)

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


class MockSimulator:
    """A Simulator that does not support any tests."""


def test_pyargs(pytester):
    # mark all the tests as unsupported
    pytester.makefile(
        ".ini",
        pytest="""
        [pytest]
        nengo_simulator = nengo.tests.test_pytest_nengo.MockSimulator
        nengo_test_unsupported =
            *
                "Using mock simulator"
        """,
    )

    outcomes = pytester.runpytest_inprocess("--pyargs", "nengo").parseoutcomes()

    assert "failed" not in outcomes
    assert "passed" not in outcomes
    assert outcomes["skipped"] > 250
    assert outcomes["deselected"] > 700
