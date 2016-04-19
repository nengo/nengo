"""Abstract syntax trees for SPA actions.

SPA actions are parsed into an abstract syntax tree (AST) representing the
structure of the action expressions. In the first step, this syntax tree is
independent of the actual model; it only represents the syntactic information.

For the construction of the AST the Python eval function is used (by modifiying
the name lookup in the globals dictionary). Because of this all Python code
not involving identifiers will be statically evaluated before insertion into
the syntax tree (e.g., '2 * 3 + 1' will be inserted as ``Scalar(7)``).

Each node in the syntax tree will evaluate to a specific type. The most
important types are ``TScalar`` for expressions evaluating to a scalar and
:class:`TVocabulary` for expressions evaluating to a semantic pointer. This
latter type describes the vocabulary the semantic pointer belongs to and
different vocabularies give different types. This ensures that only semantic
pointers of a matching vocabulary are assigned.

To determine the type of each node an actual model has to be provided. This is
because names of semantic pointers are not associated with a vocabulary and it
needs to be inferred from some actual SPA module for which we have to be able
to resolve the names of those modules. There are a few basic rules for this
type inference:

1. If something with unknown vocabulary is assigned to a module, that module's
   vocabulary provides the type.
2. If a binary operation has an operand with unknown vocabulary it is
   determined from the other operand.

Once all the types have been determined, the AST can be used to construct
Nengo objects to perform the operations represented with the AST. In this
process each node in the syntax tree can create :class:`Artifact`s. These
give generated Nengo objects to be connected to the appropriate places
including the transform that should be used for that. This is necessary
because at the time most objects are constructed we only know this constructed
object and the transform, but not what it is supposed to connect to. So the
final connection will be done by some other node in the syntax tree.

To avoid confusion with the normal Nengo build process, we use the term
'construct' here.


In the following we provide the grammar of SPA actions for reference:

``
Scalar: <any Python expression evaluating to a single number != 0>
Symbol: <valid Python identifier starting with a capital letter>
Zero: '0'
Module: <valid Python identifier> | <valid Python identifier> '.' Module
Source: S'(' Source ')' | Scalar | Symbol | Zero | Module | BinaryOperation |
        UnaryOperation | DotProduct
BinaryOperation: Product | Sum | Difference
Product: Source '*' Source
Sum: Source '+' Source
Difference: Source '-' Source
UnaryOperation: ApproxInverse | Negative
ApproxInverse: '~' Source
Negative: '-' Source
DotProduct: 'dot(' Source ',' Source ')'
Sink: <valid Python identifier> | <valid Python identifier> '.' Sink
Effect: Sink '=' Source
Effects: Effect | Effect ',' Effects
Action: Source '-->' Effects

Note that `Difference` ``a - b`` will be represented as `a + (-b)` in the AST.
"""

import warnings

import numpy as np

import nengo
from nengo.spa.pointer import SemanticPointer
from nengo.exceptions import SpaModuleError, SpaParseError, SpaTypeError


class ConstructionContext(object):
    """Context in which SPA actions are constructed.

    This primarily provides the SPA modules used to construct certain
    components. All attributes except `root_module` may be ``None`` if these
    are not provided in the current construction context.

    Attributes
    ----------
    root_module : :class:`spa.Module`
        The root module the encapsulated all of the constructed structures.
    cortical : :class:`spa.Cortical`
        Module to manage cortical rules.
    bg : :class:`spa.BasalGanglia`
        Module to manage the basal ganglia part of action selection.
    thalamus : :class:`spa.Thalamus`
        Module to manage the thalamus part of action selection.
    sink : :class:`Sink`
        Node in the AST where some result will be send to.
    """
    __slots__ = ['root_module', 'cortical', 'bg', 'thalamus', 'sink']

    def __init__(
            self, root_module, cortical=None, bg=None, thalamus=None,
            sink=None):
        self.root_module = root_module
        self.cortical = cortical
        self.bg = bg
        self.thalamus = thalamus
        self.sink = sink

    def subcontext_with_sink(self, sink):
        return self.__class__(
            root_module=self.root_module, cortical=self.cortical, bg=self.bg,
            thalamus=self.thalamus, sink=sink)

    @property
    def sink_module(self):
        return self.root_module.get_module(self.sink.name, strip_output=True)

    @property
    def sink_input(self):
        return self.root_module.get_module_input(self.sink.name)


class Type(object):
    """Describes a type.

    Each part of the AST evaluates to some type.
    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.name == other.name

    def __ne__(self, other):
        return not self == other


TAction = Type('TAction')
TScalar = Type('TScalar')
TEffect = Type('TEffect')
TEffects = Type('TEffects')


class TVocabulary(Type):
    """Each vocabulary is treated as its own type.

    All vocabulary types constitute a type class.
    """
    def __init__(self, vocab):
        super(TVocabulary, self).__init__('TVocabulary')
        self.vocab = vocab

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.vocab)

    def __str__(self):
        return '{}<{}>'.format(self.name, self.vocab)

    def __eq__(self, other):
        return (super(TVocabulary, self).__eq__(other) and
                self.vocab is other.vocab)


class Artifact(object):
    """Stores information about Nengo objects constructed from SPA actions.

    This deals with the problem that when we construct the Nengo object we
    have the object itself and know the transform for the outgoing connection,
    but we do not know what to connect to yet. Thus, this class allows to store
    and pass around that information until we know what to connect to.

    Attributes
    ----------
    nengo_source : :class:`nengo.NengoObject`
        Some constructed Nengo object that allows outgoing connections.
    transform : array-like
        Transform to be applied to the outgoing connection from the
        `nengo_source`.
    """
    def __init__(self, nengo_source, transform=1):
        self.nengo_source = nengo_source
        self.transform = transform

    def add_transform(self, tr):
        return Artifact(self.nengo_source, np.dot(tr, self.transform))


def infer_vocab(root_module, *nodes):
    """Return the first vocabulary type that can be inferred for one of the
    `nodes`.

    If the context that an operation is embedded in does not provide a
    vocabulary type, it might be possible to infer it from one of the nodes
    in the operation.

    If no vocabulary type can be inferred, ``None`` will be returned.

    Note that this function calls ``infer_types`` on a subset or all `nodes`
    which has side effects!
    """
    for node in nodes:
        try:
            node.infer_types(root_module, None)
            if isinstance(node.type, TVocabulary):
                return node.type
        except SpaTypeError:
            pass
    return None


def construct_bias(value, context):
    """Constructs a bias node (if not existent) and a transform to `value`."""
    with context.root_module:
        if not hasattr(context.root_module, 'bias'):
            context.root_module.bias = nengo.Node([1], label="bias")
    if isinstance(value, SemanticPointer):
        value = value.v
    transform = np.array([value]).T
    return [Artifact(context.root_module.bias, transform=transform)]


def value_to_transform(value):
    if isinstance(value, SemanticPointer):
        value = np.atleast_2d(value.v).T
    return np.asarray(value)


def ensure_node(obj):
    """Return `obj` as :class:`Node`.

    Leaves `obj` unchanged if it is already an instance of :class:`Node`.
    """
    if not isinstance(obj, Node):
        if obj is 0:
            obj = Zero()
        else:
            obj = Scalar(obj)
    return obj


class Node(object):
    """Abstract class for a node in the AST.

    Attributes
    ----------
    fixed : bool
        Indicates if the value of this node is static or changes with time.
    type : :class:`Type`
        Type that this node evaluates to. This will be set to ``None`` until
        the type inference was run.
    """
    def __init__(self, fixed):
        self.fixed = fixed
        self.type = None

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.__dict__ == other.__dict__)

    def infer_types(self, root_module, context_type):
        """Run type inference on this node and its children.

        Will raise a :class:`nengo.exceptions.SpaTypeError` if invalid,
        non-matching, or undefined types are discovered.

        This function is idempotent.

        Parameters
        ----------
        root_module : :class:`spa.Module`
            The root_module used to resolve names.
        context_type : :class:`Type`
            The type of the context of this node. Allows to infer the type
            from the context if the node has no definitive type on its own.
        """
        raise NotImplementedError()

    def construct(self, context):
        """Construct network components for this node and its children.

        Parameters
        ----------
        context : :class:`ConstructionContext`
            The context in which the network components are constructed.

        Returns
        -------
        list of :class:`Articfact`
            The constructed objects with transforms that should be connected to
            the objects that continue the processing.
        """
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate the value of this node statically.

        This can only be done for ``fixed`` nodes, otherwise a ``ValueError``
        will be raised.
        """
        raise NotImplementedError()


class Source(Node):
    """Abstract base class of all AST nodes that can provide some output value.
    """
    def __invert__(self):
        return ApproxInverse(self)

    def __neg__(self):
        return Negative(self)

    def __add__(self, other):
        return Sum(self, other)

    def __radd__(self, other):
        return Sum(other, self)

    def __sub__(self, other):
        return Sum(self, Negative(other))

    def __rsub__(self, other):
        return Sum(other, Negative(self))

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        return Product(other, self)

    def infer_types(self, root_module, context_type):
        raise NotImplementedError()

    def construct(self, context):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()


class Scalar(Source):
    """A fixed scalar."""
    def __init__(self, value):
        super(Scalar, self).__init__(fixed=True)
        self.value = value
        self.type = TScalar

    def infer_types(self, root_module, context_type):
        pass

    def construct(self, context):
        return construct_bias(self.value, context)

    def evaluate(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Symbol(Source):
    """A fixed semantic pointer identified by its name (`key`).

    The `key` has to start with a capatial letter.
    """
    def __init__(self, key):
        super(Symbol, self).__init__(fixed=True)
        self.validate(key)
        self.key = key

    def validate(self, key):
        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")

    def infer_types(self, root_module, context_type):
        if not isinstance(context_type, TVocabulary):
            raise SpaTypeError("Invalid type.")
        self.type = context_type
        # Make sure that key has been added to vocab after type inference to
        # make translate operations as deterministic as possible (it could
        # differ if at a later point another key would be added).
        self.type.vocab.parse(self.key)

    def construct(self, context):
        value = self.type.vocab[self.key].v
        return construct_bias(value, context)

    def evaluate(self):
        return self.type.vocab[self.key]

    def __str__(self):
        return self.key


class Zero(Source):
    """Zero which can act as scalar or zero vector."""

    def __init__(self):
        super(Zero, self).__init__(fixed=True)

    def infer_types(self, root_module, context_type):
        if context_type is None:
            self.type = TScalar
        elif context_type == TScalar or isinstance(context_type, TVocabulary):
            self.type = context_type
        else:
            raise SpaTypeError("Invalid type.")

    def construct(self, context):
        return []

    def evaluate(self):
        if self.type == TScalar:
            return 0
        else:
            return SemanticPointer(np.zeros(self.type.vocab.dimensions))

    def __str__(self):
        return "0"


class Module(Source):
    """A SPA module or module output identified by its name.

    This will provide potentially time varying input. This class is not used
    for modules that act as sink.
    """
    def __init__(self, name):
        super(Module, self).__init__(fixed=False)
        self.name = name

    def infer_types(self, root_module, context_type):
        vocab = root_module.get_module_output(self.name)[1]
        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def construct(self, context):
        return [Artifact(context.root_module.get_module_output(self.name)[0])]

    def evaluate(self):
        raise ValueError("Module cannot be statically evaluated.")

    def __getattr__(self, name):
        return Module(self.name + '.' + name)

    def __str__(self):
        return self.name


class BinaryNode(Source):
    """Base class for binary operators.

    Attributes
    ----------
    lhs : :class:`Node`
        Left-hand side
    rhs : :class:`Node`
        Right-hand side
    """
    def __init__(self, lhs, rhs):
        lhs = ensure_node(lhs)
        rhs = ensure_node(rhs)

        super(BinaryNode, self).__init__(fixed=lhs.fixed and rhs.fixed)
        self.lhs = lhs
        self.rhs = rhs

    def infer_types(self, root_module, context_type):
        raise NotImplementedError()

    def construct(self, context):
        raise NotImplementedError()

    def _connect_binary_operation(self, context, net):
        with context.root_module:
            for artifact in self.lhs.construct(context):
                nengo.Connection(
                    artifact.nengo_source, net.A, transform=artifact.transform,
                    synapse=context.root_module.synapse)
            for artifact in self.rhs.construct(context):
                nengo.Connection(
                    artifact.nengo_source, net.B, transform=artifact.transform,
                    synapse=context.root_module.synapse)


class BinaryOperation(BinaryNode):
    """Base class for binary operators.

    Attributes
    ----------
    lhs : :class:`Node`
        Left-hand side
    rhs : :class:`Node`
        Right-hand side
    operator : str
        String representation of the operator.
    """
    def __init__(self, lhs, rhs, operator):
        super(BinaryOperation, self).__init__(lhs, rhs)
        self.operator = operator

    def infer_types(self, root_module, context_type):
        if context_type is None:
            context_type = infer_vocab(root_module, self.lhs, self.rhs)

        self.lhs.infer_types(root_module, context_type)
        self.rhs.infer_types(root_module, context_type)

        if self.lhs.type == self.rhs.type:
            self.type = self.lhs.type
        elif self.lhs.type == TScalar:
            self.type = self.rhs.type
        elif self.rhs.type == TScalar:
            self.type = self.lhs.type
        else:
            raise SpaTypeError(
                "Incompatible types {} and {} in operation '{}'.".format(
                    self.lhs.type, self.rhs.type, self))

    def construct(self, context):
        raise NotImplementedError()

    def __str__(self):
        return '({} {} {})'.format(self.lhs, self.operator, self.rhs)


class Product(BinaryOperation):
    def __init__(self, lhs, rhs):
        super(Product, self).__init__(lhs, rhs, '*')

    def construct(self, context):
        if self.fixed:
            return construct_bias(self.evaluate(), context)

        if self.lhs.fixed:
            tr = self.lhs.evaluate()
            artifacts = self.rhs.construct(context)
        if self.rhs.fixed:
            tr = self.rhs.evaluate()
            artifacts = self.lhs.construct(context)

        is_binding = (isinstance(self.lhs.type, TVocabulary) and
                      isinstance(self.rhs.type, TVocabulary))

        if self.lhs.fixed or self.rhs.fixed:
            if is_binding:
                tr = tr.get_convolution_matrix()
            else:
                tr = value_to_transform(tr)
            return [x.add_transform(tr) for x in artifacts]

        with context.root_module:
            if is_binding:
                net = nengo.networks.CircularConvolution(
                    context.root_module.cconv_neurons,
                    self.type.vocab.dimensions,
                    net=nengo.Network(label='cconv'))
            elif self.lhs.type == TScalar and self.rhs.type == TScalar:
                net = nengo.networks.Product(
                    context.root_module.product_neurons, 1,
                    net=nengo.Network(label='product'))
            else:
                raise NotImplementedError(
                    "Dynamic scaling of semantic pointer not implemented.")

        self._connect_binary_operation(context, net)
        return [Artifact(net.output)]

    def evaluate(self):
        return self.lhs.evaluate() * self.rhs.evaluate()


class Sum(BinaryOperation):
    def __init__(self, lhs, rhs):
        super(Sum, self).__init__(lhs, rhs, '+')

    def construct(self, context):
        if self.fixed:
            return construct_bias(self.evaluate(), context)

        return (self.lhs.construct(context) +
                self.rhs.construct(context))

    def evaluate(self):
        return self.lhs.evaluate() + self.rhs.evaluate()


class UnaryOperation(Source):
    """Base class for unary operators.

    Attributes
    ----------
    source : :class:`Node`
        Node the operator is applied to.
    operator : str
        String representation of the operator.
    """
    def __init__(self, source, operator):
        source = ensure_node(source)
        super(UnaryOperation, self).__init__(fixed=source.fixed)
        self.source = source
        self.operator = operator

    def infer_types(self, root_module, context_type):
        self.source.infer_types(root_module, context_type)
        self.type = self.source.type

    def construct(self, context):
        raise NotImplementedError()

    def __str__(self):
        return self.operator + str(self.source)


class ApproxInverse(UnaryOperation):
    def __init__(self, source):
        super(ApproxInverse, self).__init__(source, '~')

    def infer_types(self, root_module, context_type):
        super(ApproxInverse, self).infer_types(root_module, context_type)
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Cannot apply approximate inverse to '{}' which is not of "
                "type TVocabulary, but {}.".format(self.source, self.type))

    def construct(self, context):
        if self.fixed:
            return construct_bias(self.evaluate(), context)

        d = self.type.vocab.dimensions
        tr = np.eye(d)[-np.arange(d)]
        return [x.add_transform(tr) for x in self.source.construct(context)]

    def evaluate(self):
        return ~self.source.evaluate()


class Negative(UnaryOperation):
    def __init__(self, source):
        super(Negative, self).__init__(source, '-')

    def construct(self, context):
        if self.fixed:
            return construct_bias(self.evaluate(), context)
        return [x.add_transform(-1) for x in self.source.construct(context)]

    def evaluate(self):
        return -self.source.evaluate()


class DotProduct(BinaryNode):
    def __init__(self, lhs, rhs):
        super(DotProduct, self).__init__(lhs, rhs)
        self.type = TScalar

    def infer_types(self, root_module, context_type):
        if context_type is None:
            context_type = infer_vocab(root_module, self.lhs, self.rhs)
        self.lhs.infer_types(root_module, context_type)
        self.rhs.infer_types(root_module, context_type)
        if not isinstance(self.lhs.type, TVocabulary):
            raise SpaTypeError(
                "First argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.lhs.type))
        if not isinstance(self.rhs.type, TVocabulary):
            raise SpaTypeError(
                "Second argument of dot product '{}' is not of type "
                "TVocabulary, but {}.".format(self, self.rhs.type))
        if self.lhs.type.vocab is not self.rhs.type.vocab:
            raise SpaTypeError(
                "Incompatible types {} and {} in dot product '{}'.".format(
                    self.lhs.type, self.rhs.type, self))

    def construct(self, context):
        if self.fixed:
            return construct_bias(self.evaluate(), context)

        if self.lhs.fixed:
            tr = value_to_transform(self.lhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.rhs.construct(context)]
        if self.rhs.fixed:
            tr = value_to_transform(self.rhs.evaluate()).T
            return [x.add_transform(tr)
                    for x in self.lhs.construct(context)]

        d = self.lhs.type.vocab.dimensions
        assert self.rhs.type.vocab.dimensions == d
        net = nengo.networks.Product(
            context.root_module.product_neurons, d,
            net=nengo.Network(label='dot product'))
        self._connect_binary_operation(context, net)
        return [Artifact(
            net.output, nengo.networks.product.dot_product_transform(d))]

    def evaluate(self):
        return np.dot(self.lhs.evaluate(), self.rhs.evaluate())

    def __str__(self):
        return 'dot({}, {})'.format(self.lhs, self.rhs)


class Effect(Node):
    """Assignment of an expression to a SPA module.

    Attributes
    ----------
    sink : :class:`Sink`
        Module that acts as sink.
    source : :class:`Source`
        Source of information to be fed to the sink.
    channeled : bool
        Indicates if information should be passed through an additional
        (inhibitable) channel between the source and sink.
    channel : :class:`nengo.networks.EnsembleArray`
        The channel that was constructed for this effect. Will initially be
        ``None`` and will only be constructed if `channeled` is ``True``.
    """
    def __init__(self, sink, source, channeled=False):
        source = ensure_node(source)
        super(Effect, self).__init__(fixed=source.fixed)
        self.type = TEffect
        self.sink = ensure_node(sink)
        self.source = ensure_node(source)
        self.channeled = channeled
        self.channel = None

    def infer_types(self, root_module, context_type):
        if context_type is not None:
            raise ValueError("Effect only allows a context_type of None.")
        self.sink.infer_types(root_module, None)
        self.source.infer_types(root_module, self.sink.type)
        if self.sink.type != self.source.type:
            raise SpaTypeError("Cannot assign {} to {} in '{}'".format(
                self.source.type, self.sink.type, self))

    def construct(self, context):
        assert context.sink is None
        context = context.subcontext_with_sink(self.sink)

        if self.channeled and self.fixed:
            return []  # Will be implemented in transform from thalamus

        if self.channeled:
            self.channel = context.thalamus.construct_channel(
                context.sink_module, context.sink_input)
            target = self.channel.input
            connect_fn = context.thalamus.connect
        else:
            target = context.sink_input[0]
            connect_fn = context.cortical.connect

        artifacts = self.source.construct(context)
        for artifact in artifacts:
            connect_fn(
                artifact.nengo_source, target, transform=artifact.transform)
        return []

    def evaluate(self):
        raise ValueError("Effects cannot be statically evaluated.")

    def __str__(self):
        return '{} = {}'.format(self.sink, self.source)


class Effects(Node):
    """Multiple effects."""
    def __init__(self, *effects):
        super(Effects, self).__init__(fixed=all(e.fixed for e in effects))
        self.type = TEffects
        self.effects = effects

    def infer_types(self, root_module, context_type):
        for e in self.effects:
            e.infer_types(root_module, context_type)

    def construct(self, context):
        for effect in self.effects:
            effect.construct(context)
        return []

    def evaluate(self):
        raise ValueError("Effects cannot be statically evaluated.")

    def __str__(self):
        return ', '.join(str(e) for e in self.effects)


class Sink(Node):
    """SPA module that acts as sink identified by its name."""

    def __init__(self, name):
        super(Sink, self).__init__(fixed=False)
        self.name = name

    def infer_types(self, root_module, context_type):
        vocab = root_module.get_module_input(self.name)[1]
        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def construct(self, context):
        return []

    def evaluate(self):
        raise ValueError("Sinks cannot be statically evaluated.")

    def __str__(self):
        return self.name


class Action(Node):
    """A conditional SPA action.

    Attributes
    ----------
    index : int
        Numerical index of the action.
    condition : :class:`Node`
        Condition for the action's effects to become active.
    effets : :class:`Node`
        Effects when the condition is met.
    name : str
        Name of the action.
    """
    def __init__(self, condition, effects, index=0, name=None):
        super(Action, self).__init__(fixed=False)
        self.type = TAction
        self.index = index
        self.condition = ensure_node(condition)
        self.effects = ensure_node(effects)
        self.name = name

    @property
    def effect(self):
        warnings.warn(DeprecationWarning("Use the effects attribute instead."))
        return self.effects

    def infer_types(self, root_module, context_type):
        if isinstance(self.condition, Node):
            self.condition.infer_types(root_module, context_type)
            if self.condition.type != TScalar:
                raise SpaTypeError(
                    "Condition '{}' does not evaluate to a scalar.".format(
                        self.condition))

        self.effects.infer_types(root_module, None)

    def construct(self, context):
        if context.bg is None or context.thalamus is None:
            raise SpaModuleError(
                "Conditional actions require basal ganglia and thalamus.")

        # construct bg utility
        condition_artifacts = self.condition.construct(context)

        # construct effects
        self.effects.construct(context)

        # construct thalamus gate
        if not self.effects.fixed:
            context.thalamus.construct_gate(self.index)

        for artifact in condition_artifacts:
            context.bg.connect_input(
                artifact.nengo_source, artifact.transform, self.index)

        # connect up
        for effect in self.effects.effects:
            if effect.fixed:
                sink = context.root_module.get_module_input(
                    effect.sink.name)[0]
                tr = value_to_transform(effect.source.evaluate())
                context.thalamus.connect_fixed(self.index, sink, transform=tr)
            else:
                context.thalamus.connect_gate(self.index, effect.channel)

        return []

    def evaluate(self):
        raise NotImplementedError("Cannot evaluate conditional actions.")

    def __str__(self):
        return '{} --> {}'.format(self.condition, self.effects)
