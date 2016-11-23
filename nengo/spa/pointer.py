import numpy as np

from nengo.exceptions import ValidationError
from nengo.utils.compat import is_integer, is_number, range


class SemanticPointer(object):
    """A Semantic Pointer, based on Holographic Reduced Representations.

    Operators are overloaded so that ``+`` and ``-`` are addition,
    ``*`` is circular convolution, and ``~`` is the inversion operator.
    """

    def __init__(self, data, rng=None):
        if rng is None:
            rng = np.random

        if is_integer(data):
            if data < 1:
                raise ValidationError("Number of dimensions must be a "
                                      "positive int", attr='data', obj=self)

            self.v = rng.randn(data)
            self.v /= np.linalg.norm(self.v)
        else:
            self.v = np.array(data, dtype=float)
            if len(self.v.shape) != 1:
                raise ValidationError("'data' must be a vector", 'data', self)

    def normalized(self):
        nrm = np.linalg.norm(self.v)
        if nrm > 0:
            return SemanticPointer(self.v / nrm)

    def unitary(self):
        """Make the vector unitary."""
        fft_val = np.fft.fft(self.v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
        fft_unit = fft_val / fft_norms
        return SemanticPointer(np.array((np.fft.ifft(fft_unit)).real))

    def copy(self):
        """Return another semantic pointer with the same data."""
        return SemanticPointer(data=self.v)

    def length(self):
        """Return the L2 norm of the vector."""
        return np.linalg.norm(self.v)

    def __len__(self):
        """Return the number of dimensions in the vector."""
        return len(self.v)

    def __str__(self):
        return str(self.v)

    def __add__(self, other):
        return SemanticPointer(data=self.v + other.v)

    def __neg__(self):
        return SemanticPointer(data=-self.v)

    def __sub__(self, other):
        return SemanticPointer(data=self.v - other.v)

    def __mul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If multiplied by a scalar, we do normal multiplication.
        """
        if isinstance(other, SemanticPointer):
            return self.convolve(other)
        elif is_number(other):
            return SemanticPointer(data=self.v * other)
        else:
            raise NotImplementedError(
                "Can only multiply by SemanticPointers or scalars")

    def __rmul__(self, other):
        """Multiplication of two SemanticPointers is circular convolution.

        If mutliplied by a scaler, we do normal multiplication.
        """
        return self.__mul__(other)

    def __invert__(self):
        """Return a reorganized vector that acts as an inverse for convolution.

        This reorganization turns circular convolution into circular
        correlation, meaning that ``A*B*~B`` is approximately ``A``.

        For the vector ``[1, 2, 3, 4, 5]``, the inverse is ``[1, 5, 4, 3, 2]``.
        """
        return SemanticPointer(data=self.v[-np.arange(len(self))])

    def convolve(self, other):
        """Return the circular convolution of two SemanticPointers."""
        x = np.fft.irfft(np.fft.rfft(self.v) * np.fft.rfft(other.v))
        return SemanticPointer(data=x)

    def get_convolution_matrix(self):
        """Return the matrix that does a circular convolution by this vector.

        This should be such that ``A*B == dot(A.get_convolution_matrix, B.v)``.
        """
        D = len(self.v)
        T = []
        for i in range(D):
            T.append([self.v[(i - j) % D] for j in range(D)])
        return np.array(T)

    def dot(self, other):
        """Return the dot product of the two vectors."""
        if isinstance(other, SemanticPointer):
            other = other.v
        return np.dot(self.v, other)

    def compare(self, other):
        """Return the similarity between two SemanticPointers.

        This is the normalized dotproduct, or (equivalently), the cosine of
        the angle between the two vectors.
        """
        if isinstance(other, SemanticPointer):
            other = other.v
        scale = np.linalg.norm(self.v) * np.linalg.norm(other)
        if scale == 0:
            return 0
        return np.dot(self.v, other) / scale

    def distance(self, other):
        """Return a distance measure between the vectors.

        This is ``1-cos(angle)``, so that it is 0 when they are identical, and
        the distance gets larger as the vectors are farther apart.
        """
        return 1 - self.compare(other)

    def mse(self, other):
        """Return the mean-squared-error between two vectors."""
        return np.sum((self - other).v**2) / len(self.v)


class Identity(SemanticPointer):
    def __init__(self, n_dimensions):
        data = np.zeros(n_dimensions)
        data[0] = 1.
        super(Identity, self).__init__(data)


class AbsorbingElement(SemanticPointer):
    def __init__(self, n_dimensions):
        data = np.ones(n_dimensions) / np.sqrt(n_dimensions)
        super(AbsorbingElement, self).__init__(data)


class Zero(SemanticPointer):
    def __init__(self, n_dimensions):
        super(Zero, self).__init__(np.zeros(n_dimensions))
