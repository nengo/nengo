"""Nengo-specific pytest options.

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
seen when calling pytest with the ``-v`` argument,
except that only one colon ``:`` is used between file path and function name
rather than the two colons ``::`` used by ``nodeid``.

.. code-block:: ini

   nengo_test_unsupported =
       nengo/tests/test_file_path.py:test_function_name
           "This is a message giving the reason we skip this test"
       nengo/tests/test_file_two.py:test_other_thing
           "This is a test with a multi-line reason for skipping.
           Make sure to use quotes around the whole string (and not inside)."
       nengo/tests/test_file_two.py:test_parametrized_thing[param_value]
           "This skips a parametrized test with a specific parameter value."
"""


def pytest_addoption(parser):
    parser.addoption(
        "--simulator",
        nargs=1,
        type=str,
        default=None,
        help="Specify simulator under test.",
    )
    parser.addoption(
        "--neurons",
        nargs=1,
        type=str,
        default=None,
        help="Neuron types under test (comma separated).",
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
    parser.addoption(
        "--unsupported",
        action="store_true",
        default=False,
        help="Run (with xfail) tests marked as unsupported by this backend.",
    )
    parser.addini(
        "nengo_test_unsupported",
        type="linelist",
        help="List of unsupported unit tests with reason for " "exclusion",
    )

    group = parser.getgroup("terminal reporting", "reporting", after="general")
    group.addoption(
        "--memory",
        action="store_true",
        default=False,
        help="Show memory consumed by Python after all tests are run "
        "(not available on Windows)",
    )
