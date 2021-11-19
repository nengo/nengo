import inspect
import warnings

import numpy as np

import nengo.utils.numpy as npext
from nengo.base import NengoObject, ObjView
from nengo.exceptions import ValidationError
from nengo.params import Default, IntParam, Parameter
from nengo.processes import Process
from nengo.rc import rc
from nengo.utils.numpy import is_array_like
from nengo.utils.stdlib import checked_call


class OutputParam(Parameter):
    def __init__(self, name, default, optional=True, readonly=False):
        assert optional  # None has meaning (passthrough node)
        super().__init__(name, default, optional, readonly)

    def _fn_args_validation_error(self, output, attr, node):
        n_args = 2 if node.size_in > 0 else 1
        msg = (
            f"output function '{output}' is expected to accept exactly {n_args} "
            f"argument"
        )
        msg += (
            " (time, as a float)"
            if n_args == 1
            else "s (time, as a float and data, as a NumPy array)"
        )
        return ValidationError(msg, attr=attr, obj=node)

    def check_ndarray(self, node, output):
        if len(output.shape) > 1:
            raise ValidationError(
                f"Node output must be a vector (got shape {output.shape})",
                attr=self.name,
                obj=node,
            )
        if node.size_in != 0:
            raise ValidationError(
                "output must be callable if size_in != 0", attr=self.name, obj=node
            )
        if node.size_out is not None and node.size_out != output.size:
            raise ValidationError(
                f"Size of Node output ({output.size}) does not match "
                f"size_out ({node.size_out})",
                attr=self.name,
                obj=node,
            )

    def coerce(self, node, output):  # pylint: disable=arguments-renamed
        output = super().coerce(node, output)

        size_in_set = node.size_in is not None
        node.size_in = node.size_in if size_in_set else 0

        # --- Validate and set the new size_out
        if output is None:
            if node.size_out is not None:
                warnings.warn(
                    "'Node.size_out' is being overwritten with "
                    "'Node.size_in' since 'Node.output=None'"
                )
            node.size_out = node.size_in
        elif isinstance(output, Process):
            if not size_in_set:
                node.size_in = output.default_size_in
            if node.size_out is None:
                node.size_out = output.default_size_out
        elif callable(output):
            self.check_callable_args_list(node, output)
            # We trust user's size_out if set, because calling output
            # may have unintended consequences (e.g., network communication)
            if node.size_out is None:
                node.size_out = self.check_callable_output(node, output)
        elif is_array_like(output):
            # Make into correctly shaped numpy array before validation
            output = npext.array(output, min_dims=1, copy=False, dtype=rc.float_dtype)
            self.check_ndarray(node, output)
            if not np.all(np.isfinite(output)):
                raise ValidationError(
                    "Output value must be finite.", attr=self.name, obj=node
                )
            node.size_out = output.size
        else:
            raise ValidationError(
                f"Invalid node output type '{type(output).__name__}'",
                attr=self.name,
                obj=node,
            )

        return output

    def check_callable_output(self, node, output):
        t, x = 0.0, np.zeros(node.size_in)
        args = (t, x) if node.size_in > 0 else (t,)
        result, invoked = checked_call(output, *args)
        if not invoked:
            raise self._fn_args_validation_error(output, self.name, node)
        if result is not None:
            result = np.asarray(result)
            if len(result.shape) > 1:
                raise ValidationError(
                    f"Node output must be a vector (got shape {result.shape})",
                    attr=self.name,
                    obj=node,
                )
        # return callable output size
        return 0 if result is None else result.size

    def check_callable_args_list(self, node, output):
        # not all callables provide an argspec, such as numpy
        try:
            func_argspec = inspect.getfullargspec(output)
        except (TypeError, ValueError):
            pass
        else:
            args_len = len(func_argspec.args)
            if inspect.ismethod(output) or not inspect.isroutine(output):
                # don't count self as an argument
                args_len -= 1

            defaults_len = 0
            if func_argspec.defaults is not None:
                defaults_len = len(func_argspec.defaults)

            required_len = args_len - defaults_len
            expected_len = 2 if node.size_in > 0 else 1

            if func_argspec.varargs:
                args_len = max(expected_len, args_len)

            if not required_len <= expected_len <= args_len:
                raise self._fn_args_validation_error(output, self.name, node)


class Node(NengoObject):
    """Provide non-neural inputs to Nengo objects and process outputs.

    Nodes can accept input, and perform arbitrary computations
    for the purpose of controlling a Nengo simulation.
    Nodes are typically not part of a brain model per se,
    but serve to summarize the assumptions being made
    about sensory data or other environment variables
    that cannot be generated by a brain model alone.

    Nodes can also be used to test models by providing specific input signals
    to parts of the model, and can simplify the input/output interface of a
    `~nengo.Network` when used as a relay to/from its internal
    ensembles (see `~nengo.networks.EnsembleArray` for an example).

    Parameters
    ----------
    output : callable, array_like, or None
        Function that transforms the Node inputs into outputs,
        a constant output value, or None to transmit signals unchanged.
    size_in : int, optional
        The number of dimensions of the input data parameter.
    size_out : int, optional
        The size of the output signal. If None, it will be determined
        based on the values of ``output`` and ``size_in``.
    label : str, optional
        A name for the node. Used for debugging and visualization.
    seed : int, optional
        The seed used for random number generation.
        Note: no aspects of the node are random, so currently setting
        this seed has no effect.

    Attributes
    ----------
    label : str
        The name of the node.
    output : callable, array_like, or None
        The given output.
    size_in : int
        The number of dimensions for incoming connection.
    size_out : int
        The number of output dimensions.
    """

    probeable = ("output",)

    output = OutputParam("output", default=None)
    size_in = IntParam("size_in", default=None, low=0, optional=True)
    size_out = IntParam("size_out", default=None, low=0, optional=True)

    def __init__(
        self,
        output=Default,
        size_in=Default,
        size_out=Default,
        label=Default,
        seed=Default,
    ):
        if not (seed is Default or seed is None):
            raise NotImplementedError("Changing the seed of a node has no effect")
        super().__init__(label=label, seed=seed)

        self.size_in = size_in
        self.size_out = size_out
        self.output = output  # Must be set after size_out; may modify size_out

    def __getitem__(self, key):
        return ObjView(self, key)

    def __len__(self):
        return self.size_out
