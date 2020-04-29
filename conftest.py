from pytest_allclose import report_rmses

from nengo.rc import rc


def pytest_runtest_setup(item):
    rc.reload_rc([])
    rc.set("decoder_cache", "enabled", "False")
    rc.set("exceptions", "simplified", "False")
    rc.set("nengo.Simulator", "fail_fast", "True")
    rc.set("progress", "progress_bar", "False")


def pytest_terminal_summary(terminalreporter):
    report_rmses(terminalreporter)
