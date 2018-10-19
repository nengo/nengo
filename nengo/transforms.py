import numpy as np

import nengo
from nengo.base import FrozenObject
from nengo.dists import Distribution
from nengo.exceptions import ValidationError
from nengo.utils.compat import range


def get_transform(transform, shape, rng=np.random):
    """Convenience function to sample a transform or return samples.

    Use this function in situations where you accept an argument that could
    be a transform, or could be an ``array_like`` of samples.

    Parameters
    ----------
    transform : `.Transform` or `.Distribution` or array_like
        Source of the transform to be returned.
    shape : tuple of int
        Desired shape of transform
    rng : `numpy.random.RandomState`, optional (Default: np.random)
        Random number generator.

    Returns
    -------
    samples : array_like
        Sampled array
    """

    if isinstance(transform, Transform):
        return transform.sample(rng=rng)
    elif isinstance(transform, Distribution):
        assert len(shape) == 2
        return transform.sample(*shape, rng=rng)
    else:
        assert (transform.shape == ()
                or transform.shape == (shape[0],)
                or transform.shape == shape)
        return np.array(transform)


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


class Convolution(Transform):
    """An N-dimensional convolutional transform.

    The dimensionality of the convolution is determined by the input shape.

    Parameters
    ----------
    n_filters : int
        The number of convolutional filters to apply
    input_shape : tuple of int or `.ConvShape`
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
    kernel : `.Distribution` or `~numpy:numpy.ndarray`, optional \
             (Default: Uniform(-1, 1))
        A predefined kernel with shape
        ``kernel_size + (input_channels, n_filters)``, or a ``Distribution``
        that will be used to initialize the kernel.

    Notes
    -----
    As is typical in neural networks, this is technically correlation rather
    than convolution (because the kernel is not flipped).
    """

    def __init__(self, n_filters, input_shape, kernel_size=(3, 3),
                 strides=(1, 1), padding="valid", channels_last=True,
                 kernel=nengo.dists.Uniform(-1, 1)):
        super(Convolution, self).__init__()

        if isinstance(input_shape, ConvShape):
            assert input_shape.channels_last == channels_last
        else:
            input_shape = ConvShape(input_shape, channels_last=channels_last)

        self.n_filters = n_filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.channels_last = channels_last
        self.kernel = kernel
        self.dimensions = len(kernel_size)

        if len(kernel_size) != input_shape.dimensions:
            raise ValidationError(
                "Kernel dimensions (%d) do not match input dimensions (%d)"
                % (len(kernel_size), input_shape.dimensions),
                attr="kernel_size")
        if len(strides) != input_shape.dimensions:
            raise ValidationError(
                "Stride dimensions (%d) do not match input dimensions (%d)"
                % (len(strides), input_shape.dimensions), attr="strides")
        if not isinstance(kernel, Distribution):
            if kernel.shape != self.kernel_shape:
                raise ValidationError(
                    "Kernel shape %s does not match expected shape %s"
                    % (kernel.shape, self.kernel_shape), attr="kernel")

        # compute output shape
        output_shape = np.array(input_shape.spatial_shape, dtype=np.float64)
        if self.padding == "valid":
            output_shape -= kernel_size
            output_shape += 1
        output_shape /= strides
        output_shape = tuple(np.ceil(output_shape).astype(np.int64))
        output_shape = (output_shape + (n_filters,) if channels_last
                        else (n_filters,) + output_shape)

        self.output_shape = ConvShape(
            output_shape, channels_last=channels_last)

        self._paramdict = {
            "n_filters": n_filters,
            "input_shape": self.input_shape,
            "kernel_size": self.kernel_size,
            "strides": self.strides
        }

    @property
    def kernel_shape(self):
        return self.kernel_size + (self.input_shape.n_channels, self.n_filters)

    def sample(self, rng=np.random):
        if isinstance(self.kernel, Distribution):
            # we sample this way so that any variancescaling distribution based
            # on n/d is scaled appropriately
            kernel = [
                self.kernel.sample(
                    self.input_shape.n_channels, self.n_filters, rng=rng)
                for _ in range(np.prod(self.kernel_size))
            ]
            kernel = np.reshape(kernel, self.kernel_shape)
        else:
            kernel = np.array(self.kernel)
        return kernel

    @property
    def size_in(self):
        return self.input_shape.size

    @property
    def size_out(self):
        return self.output_shape.size


class ConvShape(object):
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
    def __init__(self, shape, channels_last=True):
        self.shape = tuple(shape)
        self.channels_last = channels_last

    def __str__(self):
        return "%s(shape=%s, ch_last=%d)" % (
            type(self).__name__, self.shape, self.channels_last)

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
