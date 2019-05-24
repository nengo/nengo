import numpy as np
import pytest

from nengo.builder import Signal
from nengo.builder.operator import SparseDotInc
from nengo.exceptions import BuildError


def test_sparsedotinc_builderror():
    A = Signal(np.ones(2))
    X = Signal(np.ones(2))
    Y = Signal(np.ones(2))

    with pytest.raises(BuildError, match="must be a sparse Signal"):
        SparseDotInc(A, X, Y)
