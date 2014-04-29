from glob import glob
import os.path

import pytest

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
import _pytest.capture
_pytest.capture.DontReadFromInput.encoding = "utf-8"

from nengo.utils.compat import execfile
from nengo.utils.ipython import export_py, load_notebook


def pytest_generate_tests(metafunc):
    tests = os.path.dirname(os.path.realpath(__file__))
    examples = os.path.realpath(os.path.join(tests, '..', '..', 'examples'))
    examples = glob(examples + '/*.ipynb')

    # if `--optional` is not set, filter out optional notebooks
    ignores = [] if metafunc.config.option.optional else [
        'lorenz_attractor.ipynb']
    examples = [path for path in examples
                if os.path.split(path)[1] not in ignores]

    if "nb_path" in metafunc.funcargnames:
        metafunc.parametrize("nb_path", examples)


@pytest.mark.example
def test_noexceptions(nb_path, tmpdir):
    """Ensure that no cells raise an exception."""
    nb = load_notebook(nb_path)
    pyfile = "%s.py" % str(
        tmpdir.join(os.path.splitext(os.path.basename(nb_path))[0]))
    export_py(nb, pyfile)
    execfile(pyfile, {})


@pytest.mark.example
def test_nooutput(nb_path):
    """Ensure that no cells have output."""
    nb = load_notebook(nb_path)

    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                assert cell.outputs == [], (
                    "Clear all cell outputs in " + nb_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
