import _pytest.capture
import pytest

from nengo.utils.paths import examples_dir
from nengo.utils.stdlib import execfile

try:
    from nengo.utils.ipython import export_py, iter_cells, load_notebook
except ImportError as err:

    def export_py(*args, import_err=err, **kwargs):
        raise import_err

    def load_notebook(*args, import_err=err, **kwargs):
        raise import_err


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
]

files = [
    f for f in examples_dir.glob("**/*.ipynb") if ".ipynb_checkpoints" not in str(f)
]
all_examples = [f.relative_to(examples_dir).with_suffix("") for f in files]
slow_examples = [f for f in all_examples if f.stem in too_slow]
fast_examples = [f for f in all_examples if f.stem not in too_slow]

# glob goes in arbitrary order, so sort after the fact to keep pytest happy
# convert paths to literal strings so that they're displayed nicer in the pytest name
all_examples = sorted(str(e) for e in all_examples)
slow_examples = sorted(str(e) for e in slow_examples)
fast_examples = sorted(str(e) for e in fast_examples)


def assert_noexceptions(nb_file, tmp_path):
    plt = pytest.importorskip("matplotlib.pyplot")
    nb_path = examples_dir / f"{nb_file}.ipynb"
    nb = load_notebook(nb_path)
    pyfile = tmp_path / (nb_path.stem + ".py")
    export_py(nb, pyfile)
    execfile(pyfile, {})
    plt.close("all")


@pytest.mark.example
@pytest.mark.parametrize("nb_file", fast_examples)
@pytest.mark.filterwarnings("ignore:Creating new attribute 'memory_location'")
@pytest.mark.filterwarnings("ignore:Matplotlib is currently using agg")
def test_fast_noexceptions(nb_file, tmp_path):
    """Ensure that no cells raise an exception."""
    pytest.importorskip("IPython", minversion="3.0")
    pytest.importorskip("jinja2")
    assert_noexceptions(nb_file, tmp_path)


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("nb_file", slow_examples)
def test_slow_noexceptions(nb_file, tmp_path):
    """Ensure that no cells raise an exception."""
    pytest.importorskip("IPython", minversion="3.0")
    pytest.importorskip("jinja2")
    assert_noexceptions(nb_file, tmp_path)


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_no_outputs(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(examples_dir / f"{nb_file}.ipynb")
    for cell in iter_cells(nb):
        assert cell.outputs == [], "Cell outputs not cleared"
        assert cell.execution_count is None, "Execution count not cleared"


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_version_4(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(examples_dir / f"{nb_file}.ipynb")
    assert nb.nbformat == 4


@pytest.mark.example
@pytest.mark.parametrize("nb_file", all_examples)
def test_minimal_metadata(nb_file):
    pytest.importorskip("IPython", minversion="3.0")
    nb = load_notebook(examples_dir / f"{nb_file}.ipynb")

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
