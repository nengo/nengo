import pytest

from nengo import spa


def test_deprecation():
    with pytest.warns(DeprecationWarning):
        spa.SPA()
