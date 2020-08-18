import os

import pytest
import _pytest.capture

from nengo.utils.paths import examples_dir
from nengo.utils.stdlib import execfile

try:
    from nengo.utils.ipython import export_py, iter_cells, load_notebook
except ImportError as err:

    def export_py(err=err, *args, **kwargs):
        raise err

    def load_notebook(err=err, *args, **kwargs):
        raise err


# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None


too_slow = [
    "basal-ganglia",
    "inhibitory-gating",
    "izhikevich",
    "learn-communication-channel",
    "learn-product",
    "learn-square",
    "learn-unsupervised",
    "lorenz-attractor",
    "nef-algorithm",
    "nef-summary",
    "network-design",
    "network-design-advanced",
    "question",
    "question-control",
    "question-memory",
    "spa-parser",
    "spa-sequence",
    "spa-sequence-routed",
]

all_examples, slow_examples, fast_examples = [], [], []

for subdir, _, files in os.walk(examples_dir):
    if (os.path.sep + ".") in subdir:
        continue
    files = [f for f in files if f.endswith(".ipynb")]
    examples = [os.path.join(subdir, os.path.splitext(f)[0]) for f in files]
    all_examples.extend(examples)
    slow_examples.extend(
        [e for e, f in zip(examples, files) if os.path.splitext(f)[0] in too_slow]
    )
    fast_examples.extend(
        [e for e, f in zip(examples, files) if os.path.splitext(f)[0] not in too_slow]
    )

# os.walk goes in arbitrary order, so sort after the fact to keep pytest happy
all_examples.sort()
slow_examples.sort()
fast_examples.sort()


def assert_noexceptions(nb_file, tmpdir):
    plt = pytest.importorskip("matplotlib.pyplot")
    nb_path = os.path.join(examples_dir, "%s.ipynb" % nb_file)
    nb = load_notebook(nb_path)
    pyfile = "%s.py" % (tmpdir.join(os.path.splitext(os.path.basename(nb_path))[0]))
    export_py(nb, pyfile)
    execfile(pyfile, {})
    plt.close("all")


@pytest.mark.example
@pytest.mark.parametrize("nb_file", fast_examples)
@pytest.mark.filterwarnings("ignore:Creating new attribute 'memory_location'")
@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
def test_fast_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    pytest.importorskip("IPython", minversion="3.0")
    pytest.importorskip("jinja2")
    assert_noexceptions(nb_file, tmpdir)


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("nb_file", slow_examples)
def test_slow_noexceptions(nb_file, tmpdir):
    """Ensure that no cells raise an exception."""
    pytest.importorskip("IPython", minversion="3.0")
    pytest.importorskip("jinja2")
    assert_noexceptions(nb_file, tmpdir)


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_no_outputs(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))
    for cell in iter_cells(nb):
        assert cell.outputs == [], "Cell outputs not cleared"
        assert cell.execution_count is None, "Execution count not cleared"


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_version_4(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))
    assert nb.nbformat == 4


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_minimal_metadata(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))

    assert "kernelspec" not in nb.metadata
    assert "signature" not in nb.metadata

    badinfo = (
        "codemirror_mode",
        "file_extension",
        "mimetype",
        "nbconvert_exporter",
        "version",
    )
    for info in badinfo:
        assert info not in nb.metadata.language_info
