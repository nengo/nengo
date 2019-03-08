import numpy as np

import nengo
from nengo.base import FrozenObject
from nengo.dists import Distribution, DistOrArrayParam
from nengo.exceptions import ValidationError
from nengo.params import ShapeParam, IntParam, EnumParam, BoolParam
from nengo.utils.compat import is_array_like


class Transform(FrozenObject):
    """A base class for connection transforms."""

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
        """Expected size of input to transform"""
        raise NotImplementedError()

    @property
    def size_out(self):
        """Expected size of output from transform"""
        raise NotImplementedError()


class ChannelShapeParam(ShapeParam):
    """A parameter where the value must be a shape with channels."""

    def coerce(self, transform, shape):
        shape = ChannelShape.cast(shape)
        super(ChannelShapeParam, self).coerce(transform, shape.shape)
        return shape


class Dense(Transform):
    """A dense transformation between an input and output signal.

    Parameters
    ----------
    shape : tuple of int
        The shape of the dense matrix: ``(size_out, size_in)``.
    init : `.Distribution` or array_like, optional (Default: 1.0)
        A Distribution used to initialize the transform matrix, or a concrete
        instantiation for the matrix. If the matrix is square we also allow a
        scalar (equivalent to ``np.eye(n) * init``) or a vector (equivalent to
        ``np.diag(init)``) to represent the matrix more compactly.
    """

    shape = ShapeParam("shape", length=2, low=1)
    init = DistOrArrayParam("init")

    def __init__(self, shape, init=1.0):
        super(Dense, self).__init__()

        self.shape = shape

        if is_array_like(init):
            init = np.asarray(init, dtype=np.float64)

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
                    "Shape of initial value %s does not match expected "
                    "shape %s" % (init.shape, expected_shape), attr="init")

        self.init = init

    @property
    def _argreprs(self):
        return ["shape=%r" % (self.shape,)]

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            return self.init.sample(*self.shape, rng=rng)

        return self.init

    @property
    def init_shape(self):
        """The shape of the initial value."""
        return (self.shape if isinstance(self.init, Distribution)
                else self.init.shape)

    @property
    def size_in(self):
        return self.shape[1]

    @property
    def size_out(self):
        return self.shape[0]


class Convolution(Transform):
    """An N-dimensional convolutional transform.

    The dimensionality of the convolution is determined by the input shape.

    Parameters
    ----------
    n_filters : int
        The number of convolutional filters to apply
    input_shape : tuple of int or `.ChannelShape`
        Shape of the input signal to the convolution; e.g.,
        ``(height, width, channels)`` for a 2D convolution with
        ``channels_last=True``.
    kernel_size : tuple of int, optional (Default: (3, 3))
        Size of the convolutional kernels (1 element for a 1D convolution,
        2 for a 2D convolution, etc.).
    strides : tuple of int, optional (Default: (1, 1))
        Stride of the convolution (1 element for a 1D convolution, 2 for
        a 2D convolution, etc.).
    padding : ``"same"`` or ``"valid"``, optional (Default: "valid")
        Padding method for input signal. "Valid" means no padding, and
        convolution will only be applied to the fully-overlapping areas of the
        input signal (meaning the output will be smaller). "Same" means that
        the input signal is zero-padded so that the output is the same shape
        as the input.
    channels_last : bool, optional (Default: True)
        If ``True`` (default), the channels are the last dimension in the input
        signal (e.g., a 28x28 image with 3 channels would have shape
        ``(28, 28, 3)``).  ``False`` means that channels are the first
        dimension (e.g., ``(3, 28, 28)``).
    init : `.Distribution` or `~numpy:numpy.ndarray`, optional \
           (Default: Uniform(-1, 1))
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

    def __init__(self, n_filters, input_shape, kernel_size=(3, 3),
                 strides=(1, 1), padding="valid", channels_last=True,
                 init=nengo.dists.Uniform(-1, 1)):
        super(Convolution, self).__init__()

        self.n_filters = n_filters
        self.input_shape = ChannelShape.cast(input_shape,
                                             channels_last=channels_last)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.init = init

        if len(kernel_size) != self.dimensions:
            raise ValidationError(
                "Kernel dimensions (%d) do not match input dimensions (%d)"
                % (len(kernel_size), self.dimensions), attr="kernel_size")
        if len(strides) != self.dimensions:
            raise ValidationError(
                "Stride dimensions (%d) do not match input dimensions (%d)"
                % (len(strides), self.dimensions), attr="strides")
        if not isinstance(init, Distribution):
            if init.shape != self.kernel_shape:
                raise ValidationError(
                    "Kernel shape %s does not match expected shape %s"
                    % (init.shape, self.kernel_shape), attr="init")

    @property
    def _argreprs(self):
        argreprs = ["n_filters=%r" % (self.n_filters,),
                    "input_shape=%s" % (self.input_shape,)]
        if self.kernel_size != (3, 3):
            argreprs.append("kernel_size=%r" % (self.kernel_size,))
        if self.strides != (1, 1):
            argreprs.append("strides=%r" % (self.strides,))
        if self.padding != 'valid':
            argreprs.append("padding=%r" % (self.padding,))
        # if self.channels_last is not True:
        #     argreprs.append("channels_last=%r" % (self.channels_last,))
        return argreprs

    @property
    def channels_last(self):
        return self.input_shape.channels_last

    @property
    def dimensions(self):
        """Dimensionality of convolution."""
        return self.input_shape.dimensions

    @property
    def kernel_shape(self):
        """Full shape of kernel."""
        return self.kernel_size + (self.input_shape.n_channels, self.n_filters)

    @property
    def output_shape(self):
        """Output shape after applying convolution to input."""
        output_shape = np.array(
            self.input_shape.spatial_shape, dtype=np.float64)
        if self.padding == "valid":
            output_shape -= self.kernel_size
            output_shape += 1
        output_shape /= self.strides
        output_shape = tuple(np.ceil(output_shape).astype(np.int64))
        output_shape = (output_shape + (self.n_filters,) if self.channels_last
                        else (self.n_filters,) + output_shape)

        return ChannelShape(output_shape, channels_last=self.channels_last)

    @property
    def size_in(self):
        return self.input_shape.size

    @property
    def size_out(self):
        return self.output_shape.size

    def sample(self, rng=np.random):
        if isinstance(self.init, Distribution):
            # we sample this way so that any variancescaling distribution based
            # on n/d is scaled appropriately
            kernel = [
                self.init.sample(
                    self.input_shape.n_channels, self.n_filters, rng=rng)
                for _ in range(np.prod(self.kernel_size))
            ]
            kernel = np.reshape(kernel, self.kernel_shape)
        else:
            kernel = np.array(self.init)
        return kernel


class ChannelShape(object):
    """Represents shape information with variable channel position.

    Parameters
    ----------
    shape : tuple of int
        Signal shape
    channels_last : bool, optional (Default: True)
        If True (default), the last item in ``shape`` represents the channels,
        and the rest are spatial dimensions. Otherwise, the first item in
        ``shape`` is the channel dimension.
    """

    @staticmethod
    def cast(shape, channels_last=None):
        if isinstance(shape, ChannelShape):
            if (channels_last is not None
                    and shape.channels_last != channels_last):
                raise ValidationError(
                    "requested channels_last=%s, but shape already has "
                    "channels_last=%s" % (channels_last, shape.channels_last),
                    attr='channels_last', obj=shape)
            return shape
        else:
            return ChannelShape(shape, channels_last=channels_last)

    def __init__(self, shape, channels_last=True):
        self.shape = tuple(shape)
        self.channels_last = channels_last

    def __repr__(self):
        return "%s(shape=%s, channels_last=%s)" % (
            type(self).__name__, self.shape, self.channels_last)

    def __str__(self):
        """Tuple-like string with channel position marked with "ch"."""
        spatial = [str(s) for s in self.spatial_shape]
        channel = ["ch=%d" % self.n_channels]
        return "(%s)" % ", ".join(spatial + channel if self.channels_last else
                                  channel + spatial)

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
