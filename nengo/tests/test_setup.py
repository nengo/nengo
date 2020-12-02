# pylint: disable=missing-docstring

import ast
import sys

import pytest

from nengo.utils.paths import install_dir


@pytest.mark.skipif(
    sys.version_info < (3, 8, 0),
    reason="ast.parse `feature_version` added in Python 3.8",
)
@pytest.mark.parametrize("feature_version", [(3, 4), (3, 5)])
def test_setup_compat(feature_version):
    setup_py_path = install_dir / "setup.py"

    assert setup_py_path.exists()
    with setup_py_path.open("r") as fh:
        source = fh.read()

    parsed = ast.parse(source, feature_version=feature_version)
    assert parsed is not None
