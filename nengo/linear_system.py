import numpy as np

from nengo.base import Process
from nengo.exceptions import ValidationError
from nengo.params import BoolParam, EnumParam, NdarrayParam
from nengo.rc import rc
from nengo._vendor.scipy.signal import (
    cont2discrete,
    ss2tf,
    tf2ss,
    tf2zpk,
    zpk2ss,
)


def _as2d(M, last=True, dtype=np.float64):
    if M is None:
        return M

    M = np.asarray(M, dtype=dtype)
    if M.ndim < 2:
        return M.reshape((1, -1)) if last else M.reshape((-1, 1))
    else:
        return M


class LinearSystem(Process):
    """General linear time-invariant (LTI) system.

    Parameters
    ----------
    sys : list or tuple
        A list or tuple representing a system in one of the following forms, depending
        on the length:
          - length 2: (numerator, denominator) transfer function form
          - length 3: (zero, pole, gain) zpk form
          - length 4: (A, B, C, D) state-space form
    x0 : array_like
        Initial values for the system state. The first dimension must equal the
        ``state_size``.
    analog : boolean, optional
        Whether `sys` is in analog (continuous-time) or discrete form.
    method : string
        The method to use for discretization (if ``analog`` is True). See
        `scipy.signal.cont2discrete` for information about the options.

    Attributes
    ----------
    analog : boolean
        Whether the system is described in analog (continuous-time) or discrete form.
        Analog coefficients will be converted to discrete using the timestep ``dt``.
    method : string
        The method to use for discretization (if ``analog`` is True). See
        `scipy.signal.cont2discrete` for information about the options.
    """

    A = NdarrayParam("A", shape=("*", "*"))
    B = NdarrayParam("B", shape=("*", "*"))
    C = NdarrayParam("C", shape=("*", "*"))
    D = NdarrayParam("D", shape=("*", "*"))
    x0 = NdarrayParam("x0", shape=("...",))
    analog = BoolParam("analog")
    method = EnumParam(
        "method", values=("bilinear", "euler", "backward_diff", "zoh", "impulse")
    )

    @classmethod
    def _find_size(cls, matrices, index):
        for M in matrices:
            if M is not None and M.size > 0:
                return M.shape[index]

        return 0

    def __init__(self, sys, analog=True, method="zoh", x0=0, default_dt=0.001):
        self._CombineClass = LinearSystem

        self._init_sys = sys  # for `argreprs`
        self._tf = None
        self._zpk = None

        dtype = LinearSystem.A.dtype
        if len(sys) == 2:
            A, B, C, D = tf2ss(*sys)
        elif len(sys) == 3:
            A, B, C, D = zpk2ss(*sys)
        elif len(sys) == 4:
            A, B, C, D = sys
        else:
            raise ValidationError(
                "Must be a tuple in (num, den), (z, p, k), or (A, B, C, D) form "
                "(the received length %d is not 2, 3, or 4)" % len(sys),
                "sys",
                obj=self,
            )

        A, B, C, D = [_as2d(M, last=False) for M in (A, B, C, D)]
        self._input_size = self._find_size([B, D], index=1)
        self._output_size = self._find_size([C, D], index=0)
        state_size = self._find_size([A, B], index=0)
        Ashape = (state_size, state_size)
        Bshape = (state_size, self.input_size)
        Cshape = (self.output_size, state_size)
        Dshape = (self.output_size, self.input_size)
        self.A = np.zeros(Ashape, dtype=dtype) if A is None or A.size == 0 else A
        self.B = np.zeros(Bshape, dtype=dtype) if B is None or B.size == 0 else B
        self.C = np.zeros(Cshape, dtype=dtype) if C is None or C.size == 0 else C
        self.D = np.zeros(Dshape, dtype=dtype) if D is None or D.size == 0 else D
        self.x0 = x0

        super().__init__(
            default_size_in=self.input_size,
            default_size_out=self.output_size,
            default_dt=default_dt,
        )
        self.analog = analog
        self.method = method

        # check system
        if self.A.ndim != 2 or self.A.shape != Ashape:
            raise ValidationError("Must be a square matrix", "A", obj=self)
        if self.B.ndim != 2 or self.B.shape != Bshape:
            raise ValidationError("Must be a (%d, %d) matrix" % Bshape, "B", obj=self)
        if self.C.ndim != 2 or self.C.shape != Cshape:
            raise ValidationError("Must be a (%d, %d) matrix" % Cshape, "C", obj=self)
        if self.D.ndim != 2 or self.D.shape != Dshape:
            raise ValidationError("Must be a (%d, %d) matrix" % Dshape, "D", obj=self)

        if self.x0.ndim > 0 and self.x0.shape[-1] != self.state_size:
            raise ValidationError(
                "Last dimension (%d) must equal state size (%d)"
                % (self.x0.shape[-1], self.state_size),
                "x0",
                obj=self,
            )

    @property
    def sys(self):
        """The system in the format used at initialization"""
        return self._init_sys

    @property
    def ss(self):
        """The system in state-space (A, B, C, D) form."""
        return self.A, self.B, self.C, self.D

    @property
    def tf(self):
        """The system in transfer function (numerator, denominator) form."""
        if self._tf is None:
            self._tf = ss2tf(*self.ss)
        return self._tf

    @property
    def zpk(self):
        """The system in zpk (zero, pole, gain) form."""
        if self._zpk is None:
            self._zpk = tf2zpk(*self.tf)
        return self._zpk

    @property
    def state_size(self):
        """The size of the system state"""
        return self.A.shape[0]

    @property
    def input_size(self):
        """The size of the system input"""
        return self._input_size

    @property
    def output_size(self):
        """The size of the system output"""
        return self._output_size

    def _expand_x0(self, x0=None, shape=(), broadcast=False):
        x0 = np.asarray(x0) if x0 is not None else self.x0
        full_shape = shape + (self.state_size,)
        if self.state_size == 0:
            return np.zeros(full_shape, dtype=x0.dtype)

        if x0.ndim > 0 and (
            x0.ndim > len(full_shape) or x0.shape != full_shape[-x0.ndim :]
        ):
            raise ValidationError(
                "Initial state shape %s does not match requested shape %s"
                % (x0.shape, full_shape),
                "x0",
                obj=self,
            )

        return x0 if broadcast else x0 * np.ones(full_shape, dtype=x0.dtype)

    def combine(self, other):
        """Combine in series with another LinearSystem."""
        cls = self._CombineClass
        if not isinstance(other, cls):
            raise ValidationError(
                "Can only combine with other %s" % cls.__name__, attr="other", obj=self,
            )
        if self.analog != other.analog:
            raise ValidationError(
                "Cannot combine analog and digital systems", attr="other", obj=self,
            )

        if self.input_size != other.output_size:
            raise ValidationError(
                "Input size (%d) must match output size of other system (%d)"
                % (self.input_size, other.output_size),
                attr="other",
                obj=self,
            )

        input_size = other.input_size
        s1 = other.state_size
        state_size = self.state_size + other.state_size
        output_size = self.output_size
        A = np.zeros((state_size, state_size), dtype=self.A.dtype)
        B = np.zeros((state_size, input_size), dtype=self.B.dtype)
        C = np.zeros((output_size, state_size), dtype=self.C.dtype)
        D = np.zeros((output_size, input_size), dtype=self.D.dtype)

        # Let y = y2, x = [x1, x2], u = u1, where 1 == other and 2 == self
        # x2 = A2.x2 + B2.u2 = A2.x2 + B2.C1.x1 + B2.D1.u1
        # y2 = C2.x2 + D2.u2 = C2.x2 + D2.C1.x1 + D2.D1.u1
        A[:s1, :s1] = other.A
        A[s1:, :s1] = self.B.dot(other.C)
        A[s1:, s1:] = self.A
        B[:s1, :] = other.B
        B[s1:, :] = self.B.dot(other.D)
        C[:, :s1] = self.D.dot(other.C)
        C[:, s1:] = self.C
        D[:, :] = self.D.dot(other.D)

        x0 = 0
        if state_size > 0:
            x0shape = (
                ()
                if self.x0.ndim <= 1 and other.x0.ndim <= 1
                else self.x0.shape[:-1]
                if self.x0.ndim > other.x0.ndim
                else other.x0.shape[:-1]
            )
            x0a = self._expand_x0(shape=x0shape)
            x0b = other._expand_x0(shape=x0shape)
            x0 = np.concatenate([x0b, x0a], axis=-1)

        return cls(
            sys=(A, B, C, D),
            analog=self.analog,
            method=self.method,
            x0=x0,
            default_dt=self.default_dt,
        )

    def discrete_ss(self, dt):
        """Returns the discrete system state-space representation.

        Note that if the system is ``not analog``, the stored (A, B, C, D) is returned.
        This function cannot change the ``dt`` of an already discrete system.
        """
        A, B, C, D = self.ss

        # discretize (if len(A) == 0, filter is stateless and already discrete)
        if self.analog and len(A) > 0:
            A, B, C, D, _ = cont2discrete((A, B, C, D), dt, method=self.method)

        return A, B, C, D

    def discretize(self, dt):
        """Returns an equivalent discrete linear system.

        Note that if the system is ``not analog``, a copy of the system is returned.
        This function cannot change the ``dt`` of an already discrete system.
        """
        return self._CombineClass(
            self.discrete_ss(dt),
            analog=False,
            method=self.method,
            x0=self.x0,
            default_dt=self.default_dt,
        )

    def make_state(self, shape_in, shape_out, dt, dtype=None, x0=None):
        dtype = rc.float_dtype if dtype is None else np.dtype(dtype)
        if dtype.kind != "f":
            raise ValidationError(
                "Only float data types are supported (got %s). Please cast "
                "your data to a float type." % dtype,
                attr="dtype",
                obj=self,
            )

        # create state memory variable X
        assert shape_in[-1] == self.input_size
        assert shape_out[-1] == self.output_size
        assert shape_in[:-1] == shape_out[:-1]
        X = np.zeros(shape_out[:-1] + (self.state_size,), dtype=dtype)
        X[:] = self._expand_x0(x0, shape=shape_out[:-1], broadcast=True)
        return {"X": X}

    def make_step(self, shape_in, shape_out, dt, rng, state):  # noqa: C901
        assert state is not None

        X = state["X"]
        A, B, C, D = [M.astype(X.dtype) for M in self.discrete_ss(dt)]
        AT, BT, CT, DT = A.T, B.T, C.T, D.T
        has_A = A.size > 0 and (A != 0).any()
        has_B = B.size > 0 and (B != 0).any()
        has_C = C.size > 0 and (C != 0).any()
        has_D = D.size > 0 and (D != 0).any()
        has_output = self.output_size > 0

        if self.input_size > 0:

            def step_linearsystem(t, u, A=A, B=B, C=C, D=D, X=X):
                if has_C and has_D:
                    Y = X.dot(CT) + u.dot(DT)
                elif has_C:
                    Y = X.dot(CT)
                elif has_D:
                    Y = u.dot(DT)
                else:
                    Y = np.zeros(shape_out) if has_output else None

                if has_A and has_B:
                    X[...] = X.dot(AT) + u.dot(BT)
                elif has_A:
                    X[...] = X.dot(AT)
                elif has_B:
                    X[...] = u.dot(BT)
                else:
                    X[...] = 0

                return Y

        else:

            def step_linearsystem(t, A=A, B=B, C=C, D=D, X=X):
                if has_C:
                    Y = X.dot(CT)
                else:
                    Y = np.zeros(shape_out) if has_output else None

                if has_A:
                    X[...] = X.dot(AT)
                else:
                    X[...] = 0

                return Y

        return step_linearsystem
