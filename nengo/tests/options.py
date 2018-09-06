def pytest_addoption(parser):
    parser.addoption('--simulator', nargs=1, type=str, default=None,
                     help='Specify simulator under test.')
    parser.addoption('--ref-simulator', nargs=1, type=str, default=None,
                     help='Specify reference simulator under test.')
    parser.addoption('--neurons', nargs=1, type=str, default=None,
                     help='Neuron types under test (comma separated).')
    parser.addoption(
        '--plots', nargs='?', default=False, const=True,
        help='Save plots (can optionally specify a directory for plots).')
    parser.addoption(
        '--analytics', nargs='?', default=False, const=True,
        help='Save analytics (can optionally specify a directory for data).')
    parser.addoption(
        '--compare', nargs=2,
        help='Compare analytics results (specify directories to compare).')
    parser.addoption(
        '--logs', nargs='?', default=False, const=True,
        help='Save logs (can optionally specify a directory for logs).')
    parser.addoption('--noexamples', action='store_true', default=False,
                     help='Do not run examples')
    parser.addoption(
        '--slow', action='store_true', default=False,
        help='Also run slow tests.')
    parser.addoption('--seed-offset', nargs=1, type=int, default=0,
                     help="Specify offset of the seed values used in tests.")

    parser.addini("nengo_test_tolerances", type="linelist",
                  help="List of 'testname atol=x rtol=y' to override "
                       "tolerances on a test-by-test basis; if there are"
                       "multiple calls to 'allclose' within a single test,"
                       "repeated items with the same testname will be applied"
                       "in sequence")
