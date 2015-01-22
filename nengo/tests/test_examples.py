from glob import glob
import os

import pytest

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
import _pytest.capture
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None

from nengo.utils.paths import examples_dir
from nengo.utils.stdlib import execfile


def pytest_generate_tests(metafunc):
    examples = glob('%s/*.ipynb' % examples_dir)

    # if `--optional` is not set, filter out time-consuming notebooks
    ignores = [] if metafunc.config.option.optional else [
        'inhibitory_gating.ipynb', 'izhikevich.ipynb',
        'learn_communication_channel.ipynb', 'learn_product.ipynb',
        'learn_square.ipynb', 'learn_unsupervised.ipynb',
        'lorenz_attractor.ipynb', 'nef_summary.ipynb', 'network_design.ipynb',
        'network_design_advanced.ipynb', 'question.ipynb',
        'question_control.ipynb', 'question_memory.ipynb',
        'spa_parser.ipynb', 'spa_sequence.ipynb',
        'spa_sequence_routed.ipynb']
    argvalues = [pytest.mark.skipif(os.path.basename(path) in ignores,
                                    reason="Time-consuming")(path)
                 for path in examples]

    if "nb_path" in metafunc.funcargnames:
        metafunc.parametrize("nb_path", argvalues)


@pytest.mark.example
def test_noexceptions(nb_path, tmpdir, plt):
    """Ensure that no cells raise an exception."""
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import export_py, load_notebook
    nb = load_notebook(nb_path)
    pyfile = "%s.py" % (
        tmpdir.join(os.path.splitext(os.path.basename(nb_path))[0]))
    export_py(nb, pyfile)
    execfile(pyfile, {})
    # Note: plt imported but not used to ensure figures are closed
    plt.saveas = None


@pytest.mark.example
def test_nooutput(nb_path):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import load_notebook
    nb = load_notebook(nb_path)

    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                assert cell.outputs == [], (
                    "Clear all cell outputs in " + nb_path)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
