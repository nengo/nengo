import os

import pytest
import _pytest.capture

from nengo.utils.paths import examples_dir
from nengo.utils.stdlib import execfile

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None


too_slow = ['inhibitory_gating',
            'izhikevich',
            'learn_communication_channel',
            'learn_product',
            'learn_square',
            'learn_unsupervised',
            'lorenz_attractor',
            'nef_summary',
            'network_design',
            'network_design_advanced',
            'question',
            'question_control',
            'question_memory',
            'spa_parser',
            'spa_sequence',
            'spa_sequence_routed']

all_examples, slow_examples, fast_examples = [], [], []

for subdir, _, files in os.walk(examples_dir):
    if (os.path.sep + '.') in subdir:
        continue
    files = [f for f in files if f.endswith('.ipynb')]
    examples = [os.path.join(subdir, os.path.splitext(f)[0]) for f in files]
    all_examples.extend(examples)
    slow_examples.extend([e for e, f in zip(examples, files)
                          if os.path.splitext(f)[0] in too_slow])
    fast_examples.extend([e for e, f in zip(examples, files)
                          if os.path.splitext(f)[0] not in too_slow])

# os.walk goes in arbitrary order, so sort after the fact to keep pytest happy
all_examples.sort()
slow_examples.sort()
fast_examples.sort()


def assert_noexceptions(nb_file, tmpdir):
    plt = pytest.importorskip('matplotlib.pyplot')
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import export_py, load_notebook
    nb_path = os.path.join(examples_dir, "%s.ipynb" % nb_file)
    nb = load_notebook(nb_path)
    pyfile = "%s.py" % (
        tmpdir.join(os.path.splitext(os.path.basename(nb_path))[0]))
    export_py(nb, pyfile)
    execfile(pyfile, {})
    plt.close('all')


@pytest.mark.example
@pytest.mark.parametrize('nb_file', fast_examples)
def test_fast_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    assert_noexceptions(nb_file, tmpdir)


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize('nb_file', slow_examples)
def test_slow_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    assert_noexceptions(nb_file, tmpdir)


@pytest.mark.example
@pytest.mark.parametrize('nb_file', all_examples)
def test_nooutput(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import load_notebook

    def check_all(cells):
        for cell in cells:
            if cell.cell_type == 'code':
                assert cell.outputs == [], ("Clear outputs in %s" % nb_path)

    nb_path = os.path.join(examples_dir, "%s.ipynb" % nb_file)
    nb = load_notebook(nb_path)

    assert 'signature' not in nb.metadata, "Remove signature in %s" % nb_path
    if nb.nbformat <= 3:
        for ws in nb.worksheets:
            check_all(ws.cells)
    else:
        check_all(nb.cells)
