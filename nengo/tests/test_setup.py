# pylint: disable=missing-docstring

import ast
import sys

import pytest

from nengo.utils.paths import install_dir

try:
    import tomllib
except ImportError:
    import tomli as tomllib

def test_setup_compat():
    setup_py_path = install_dir / "pyproject.toml"
    with setup_py_path.open("rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["name"] == "nengo"

