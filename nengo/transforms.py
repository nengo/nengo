import warnings

import numpy as np

from nengo.base import FrozenObject
from nengo.dists import DistOrArrayParam, Distribution, Uniform
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam,
    EnumParam,
    IntParam,
    NdarrayParam,
    Parameter,
    ShapeParam,
)
from nengo.rc import rc
from nengo.utils.numpy import is_array_like, scipy_sparse


class Transform(FrozenObject):
    """A base class for connection transforms.

    .. versionadded:: 3.0.0
    """

    def sample(self, rng=np.random):
        """Returns concrete weights to implement the specified transform.

        Parameters
        ----------
        rng : `numpy.random.RandomState`, optional
            Random number generator state.

        Returns
        -------
        array_like
            Transform weights
        """
        raise NotImplementedError()

    @property
    def size_in(self):
        """Expected size of input to transform."""
        raise NotImplementedError()

    @property
    def size_out(self):
        """Expected size of output from transform."""
        raise NotImplementedError()


class ChannelShapeParam(ShapeParam):
    """A parameter where the value must be a shape with channels.

    .. versionadded:: 3.0.0
    """

    def coerce(self, transform, shape):  # pylint: disable=arguments-renamed
        if isinstance(shape, ChannelShape):
            if shape.channels_last != transform.channels_last:
                raise ValidationError(
                    f"transform has channels_last={transform.channels_last}, but input "
                    f"shape has channels_last={shape.channels_last}",
                    attr=self.name,
                    obj=transform,
                )
            super().coerce(transform, shape.shape)
        else:
            super().coerce(transform, shape)
            shape = ChannelShape(shape, channels_last=transform.channels_last)
        return shape


class Dense(Transform):
    """A dense matrix transformation between an input and output signal.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    shape : tuple of int
        The shape of the dense matrix: ``(size_out, size_in)``.
    init : `.Distribution` or array_like, optional
        A Distribution used to initialize the transform matrix, or a concrete
        instantiation for the matrix. If the matrix is square we also allow a
        scalar (equivalent to ``np.eye(n) * init``) or a vector (equivalent to
        ``np.diag(init)``) to represent the matrix more compactly.
    """

    shape = ShapeParam("shape", length=2, low=1)
    init = DistOrArrayParam("init")

    def __init__(self, shape, init=1.0):
        super().__init__()

        self.shape = shape

        if is_array_like(init):
            init = np.asarray(init, dtype=rc.float_dtype)

            # check that the shape of init is compatible with the given shape
            # for this transform
            expected_shape = None
            if shape[0] != shape[1]:
                # init must be 2D if transform is not square
                expected_shape = shape
            elif init.ndim == 1:
                expected_shape = (shape[0],)
            elif init.ndim >= 2:
                expected_shape = shape

            if expected_shape is not None and init.shape != expected_shape:
                raise ValidationError(
                    f"Shape of initial value {init.shape} does not match expected "
                    f"shape {expected_shape}",
                    attr="init",
                )

        self.init = init

    @property
    def _argreprs(self):
        return [f"shape={self.shape!r}"]

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            return self.init.sample(*self.shape, rng=rng)

        return self.init

    @property
    def init_shape(self):
        """The shape of the initial value."""
        return self.shape if isinstance(self.init, Distribution) else self.init.shape

    @property
    def size_in(self):
        return self.shape[1]

    @property
    def size_out(self):
        return self.shape[0]


class SparseInitParam(Parameter):
    def coerce(self, instance, value):
        if not (
            isinstance(value, SparseMatrix)
            or (scipy_sparse is not None and isinstance(value, scipy_sparse.spmatrix))
        ):
            raise ValidationError(
                "Must be `nengo.transforms.SparseMatrix` or "
                f"`scipy.sparse.spmatrix`, got '{type(value)}'",
                attr="init",
                obj=instance,
            )
        return super().coerce(instance, value)


class SparseMatrix(FrozenObject):
    """Represents a sparse matrix.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    indices : array_like of int
        An Nx2 array of integers indicating the (row,col) coordinates for the
        N non-zero elements in the matrix.
    data : array_like or `.Distribution`
        An Nx1 array defining the value of the nonzero elements in the matrix
        (corresponding to ``indices``), or a `.Distribution` that will be
        used to initialize the nonzero elements.
    shape : tuple of int
        Shape of the full matrix.
    """

    indices = NdarrayParam("indices", shape=("*", 2), dtype=np.int64)
    data = DistOrArrayParam("data", sample_shape=("*",))
    shape = ShapeParam("shape", length=2)

    def __init__(self, indices, data, shape):
        super().__init__()

        self.indices = indices
        self.shape = shape

        # if data is not a distribution
        if is_array_like(data):
            data = np.asarray(data)

            # convert scalars to vectors
            if data.size == 1:
                data = data.item() * np.ones(self.indices.shape[0], dtype=data.dtype)

            if data.ndim != 1 or data.shape[0] != self.indices.shape[0]:
                raise ValidationError(
                    "Must be a vector of the same length as `indices`",
                    attr="data",
                    obj=self,
                )

        self.data = data
        self._allocated = None
        self._dense = None

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self.indices.shape[0]

    def allocate(self):
        """Return a `scipy.sparse.csr_matrix` or dense matrix equivalent.

        We mark this data as readonly to be consistent with how other
        data associated with signals are allocated. If this allocated
        data is to be modified, it should be copied first.
        """

        if self._allocated is not None:
            return self._allocated

        if scipy_sparse is None:
            warnings.warn(
                "Sparse operations require Scipy, which is not "
                "installed. Using dense matrices instead."
            )
            self._allocated = self.toarray().view()
        else:
            self._allocated = scipy_sparse.csr_matrix(
                (self.data, self.indices.T), shape=self.shape
            )
            self._allocated.data.setflags(write=False)

        return self._allocated

    def sample(self, rng=np.random):
        """Convert `.Distribution` data to fixed array.

        Parameters
        ----------
        rng : `.numpy.random.RandomState`
            Random number generator that will be used when
            sampling distribution.

        Returns
        -------
        matrix : `.SparseMatrix`
            A new `.SparseMatrix` instance with `.Distribution` converted to
            array if ``self.data`` is a `.Distribution`, otherwise simply
            returns ``self``.
        """
        if isinstance(self.data, Distribution):
            return SparseMatrix(
                self.indices,
                self.data.sample(self.indices.shape[0], rng=rng),
                self.shape,
            )
        else:
            return self

    def toarray(self):
        """Return the dense matrix equivalent of this matrix."""

        if self._dense is not None:
            return self._dense

        self._dense = np.zeros(self.shape, dtype=self.dtype)
        self._dense[self.indices[:, 0], self.indices[:, 1]] = self.data
        # Mark as readonly, if the user wants to modify they should copy first
        self._dense.setflags(write=False)
        return self._dense


class Sparse(Transform):
    """A sparse matrix transformation between an input and output signal.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    shape : tuple of int
        The full shape of the sparse matrix: ``(size_out, size_in)``.
    indices : array_like of int
        An Nx2 array of integers indicating the (row,col) coordinates for the
        N non-zero elements in the matrix.
    init : `.Distribution` or array_like, optional
        A Distribution used to initialize the transform matrix, or a concrete
        instantiation for the matrix. If the matrix is square we also allow a
        scalar (equivalent to ``np.eye(n) * init``) or a vector (equivalent to
        ``np.diag(init)``) to represent the matrix more compactly.
    """

    shape = ShapeParam("shape", length=2, low=1)
    init = SparseInitParam("init")

    def __init__(self, shape, indices=None, init=1.0):
        super().__init__()

        self.shape = shape

        if scipy_sparse and isinstance(init, scipy_sparse.spmatrix):
            assert indices is None
            assert init.shape == self.shape
            self.init = init
        elif indices is not None:
            self.init = SparseMatrix(indices, init, shape)
        else:
            raise ValidationError(
                "Either `init` must be a `scipy.sparse.spmatrix`, "
                "or `indices` must be specified.",
                attr="init",
            )

    @property
    def _argreprs(self):
        return [f"shape={self.shape!r}"]

    def sample(self, rng=np.random):
        if scipy_sparse and isinstance(self.init, scipy_sparse.spmatrix):
            return self.init
        else:
            return self.init.sample(rng=rng)

    @property
    def size_in(self):
        return self.shape[1]

    @property
    def size_out(self):
        return self.shape[0]


class Convolution(Transform):
    """An N-dimensional convolutional transform.

    The dimensionality of the convolution is determined by the input shape.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    n_filters : int
        The number of convolutional filters to apply
    input_shape : tuple of int or `.ChannelShape`
        Shape of the input signal to the convolution; e.g.,
        ``(height, width, channels)`` for a 2D convolution with
        ``channels_last=True``.
    kernel_size : tuple of int, optional
        Size of the convolutional kernels (1 element for a 1D convolution,
        2 for a 2D convolution, etc.).
    strides : tuple of int, optional
        Stride of the convolution (1 element for a 1D convolution, 2 for
        a 2D convolution, etc.).
    padding : ``"same"`` or ``"valid"``, optional
        Padding method for input signal. "Valid" means no padding, and
        convolution will only be applied to the fully-overlapping areas of the
        input signal (meaning the output will be smaller). "Same" means that
        the input signal is zero-padded so that the output is the same shape
        as the input.
    channels_last : bool, optional
        If ``True`` (default), the channels are the last dimension in the input
        signal (e.g., a 28x28 image with 3 channels would have shape
        ``(28, 28, 3)``).  ``False`` means that channels are the first
        dimension (e.g., ``(3, 28, 28)``).
    init : `.Distribution` or `~numpy:numpy.ndarray`, optional
        A predefined kernel with shape
        ``kernel_size + (input_channels, n_filters)``, or a ``Distribution``
        that will be used to initialize the kernel.

    Notes
    -----
    As is typical in neural networks, this is technically correlation rather
    than convolution (because the kernel is not flipped).
    """

    n_filters = IntParam("n_filters", low=1)
    input_shape = ChannelShapeParam("input_shape", low=1)
    kernel_size = ShapeParam("kernel_size", low=1)
    strides = ShapeParam("strides", low=1)
    padding = EnumParam("padding", values=("same", "valid"))
    channels_last = BoolParam("channels_last")
    init = DistOrArrayParam("init")

    _param_init_order = ["channels_last", "input_shape"]

    def __init__(
        self,
        n_filters,
        input_shape,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        channels_last=True,
        init=Uniform(-1, 1),
    ):
        super().__init__()

        self.n_filters = n_filters
        self.channels_last = channels_last  # must be set before input_shape
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.init = init

        if len(kernel_size) != self.dimensions:
            raise ValidationError(
                f"Kernel dimensions ({len(kernel_size)}) does not match "
                f"input dimensions ({self.dimensions})",
                attr="kernel_size",
            )
        if len(strides) != self.dimensions:
            raise ValidationError(
                f"Stride dimensions ({len(strides)}) does not match "
                f"input dimensions ({self.dimensions})",
                attr="strides",
            )
        if not isinstance(init, Distribution):
            if init.shape != self.kernel_shape:
                raise ValidationError(
                    f"Kernel shape {init.shape} does not match "
                    f"expected shape {self.kernel_shape}",
                    attr="init",
                )

    @property
    def _argreprs(self):
        argreprs = [
            f"n_filters={self.n_filters!r}",
            f"input_shape={self.input_shape.shape}",
        ]
        if self.kernel_size != (3, 3):
            argreprs.append(f"kernel_size={self.kernel_size!r}")
        if self.strides != (1, 1):
            argreprs.append(f"strides={self.strides!r}")
        if self.padding != "valid":
            argreprs.append(f"padding={self.padding!r}")
        if self.channels_last is not True:
            argreprs.append(f"channels_last={self.channels_last!r}")
        return argreprs

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            # we sample this way so that any variancescaling distribution based
            # on n/d is scaled appropriately
            kernel = [
                self.init.sample(self.input_shape.n_channels, self.n_filters, rng=rng)
                for _ in range(np.prod(self.kernel_size))
            ]
            kernel = np.reshape(kernel, self.kernel_shape)
        else:
            kernel = np.array(self.init, dtype=rc.float_dtype)
        return kernel

    @property
    def kernel_shape(self):
        """Full shape of kernel."""
        return self.kernel_size + (self.input_shape.n_channels, self.n_filters)

    @property
    def size_in(self):
        return self.input_shape.size

    @property
    def size_out(self):
        return self.output_shape.size

    @property
    def dimensions(self):
        """Dimensionality of convolution."""
        return self.input_shape.dimensions

    @property
    def output_shape(self):
        """Output shape after applying convolution to input."""
        output_shape = np.array(self.input_shape.spatial_shape, dtype=rc.float_dtype)
        if self.padding == "valid":
            output_shape -= self.kernel_size
            output_shape += 1
        output_shape /= self.strides
        output_shape = tuple(np.ceil(output_shape).astype(rc.int_dtype))
        output_shape = (
            output_shape + (self.n_filters,)
            if self.channels_last
            else (self.n_filters,) + output_shape
        )

        return ChannelShape(output_shape, channels_last=self.channels_last)


class ChannelShape:
    """Represents shape information with variable channel position.

    .. versionadded:: 3.0.0

    Parameters
    ----------
    shape : tuple of int
        Signal shape
    channels_last : bool, optional
        If True (default), the last item in ``shape`` represents the channels,
        and the rest are spatial dimensions. Otherwise, the first item in
        ``shape`` is the channel dimension.
    """

    def __init__(self, shape, channels_last=True):
        self.shape = tuple(shape)
        self.channels_last = channels_last

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.shape == other.shape
            and self.channels_last == other.channels_last
        )

    def __hash__(self):
        return hash((self.shape, self.channels_last))

    def __repr__(self):
        return (
            f"{type(self).__name__}(shape={self.shape}, "
            f"channels_last={self.channels_last})"
        )

    def __str__(self):
        """Tuple-like string with channel position marked with 'ch'."""
        spatial = [str(s) for s in self.spatial_shape]
        channel = [f"ch={self.n_channels}"]
        return (
            "("
            + ", ".join(spatial + channel if self.channels_last else channel + spatial)
            + ")"
        )

    @property
    def spatial_shape(self):
        """The spatial part of the shape (omitting channels)."""
        if self.channels_last:
            return self.shape[:-1]
        return self.shape[1:]

    @property
    def size(self):
        """The total number of elements in the represented signal."""
        return np.prod(self.shape)

    @property
    def n_channels(self):
        """The number of channels in the represented signal."""
        return self.shape[-1 if self.channels_last else 0]

    @property
    def dimensions(self):
        """The spatial dimensionality of the represented signal."""
        return len(self.shape) - 1


class NoTransform(Transform):
    """Directly pass the signal through without any transform operations.

    .. versionadded:: 3.1.0

    Parameters
    ----------
    size_in : int
        Dimensionality of transform input and output.
    """

    def __init__(self, size_in):
        super().__init__()

        self._size_in = size_in

    def sample(self, rng=np.random):
        """Returns concrete weights to implement the specified transform.

        Parameters
        ----------
        rng : `numpy.random.RandomState`, optional
            Random number generator state.

        Raises
        ------
        TypeError
            There is nothing to sample for NoTransform, so it is an error
            if this is called.
        """
        raise TypeError("Cannot sample a NoTransform")

    @property
    def size_in(self):
        """Expected size of input to transform."""
        return self._size_in

    @property
    def size_out(self):
        """Expected size of output from transform."""
        return self._size_in
