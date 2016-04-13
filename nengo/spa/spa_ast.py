from nengo.exceptions import SpaModuleError, SpaParseError, SpaTypeError


class Type(object):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.name)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.name == other.name


TAction = Type('TAction')
TScalar = Type('TScalar')
TEffect = Type('TEffect')


class TVocabulary(Type):
    def __init__(self, vocab):
        super(TVocabulary, self).__init__('TVocabulary')
        self.vocab = vocab

    def __repr__(self):
        return '{}({!r}, {!r})'.format(
            self.__class__.__name__, self.name, self.vocab)

    def __eq__(self, other):
        return (super(TVocabulary, self).__eq__(other) and
                self.vocab is other.vocab)


class Node(object):
    def __init__(self):
        self.type = None

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.__dict__ == other.__dict__)

    def infer_types(self, model, context_type):
        raise NotImplementedError()


class Source(Node):
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

    def infer_types(self, model, context_type):
        raise NotImplementedError()


class Symbol(Source):
    def __init__(self, key):
        super(Symbol, self).__init__()
        self.validate(key)
        self.key = key

    def validate(self, key):
        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")

    def infer_types(self, model, context_type):
        if not isinstance(context_type, TVocabulary):
            raise SpaTypeError("Invalid type.")
        self.type = context_type

    def __str__(self):
        return self.key


class Module(Source):
    def __init__(self, name):
        super(Module, self).__init__()
        self.name = name

    def infer_types(self, root_module, context_type):
        vocab = root_module.get_module_output(self.name)[1]
        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def __getattr__(self, name):
        return Module(self.name + '.' + name)

    def __str__(self):
        return self.name


class BinaryOperation(Source):
    def __init__(self, lhs, rhs, operator):
        super(BinaryOperation, self).__init__()
        self.lhs = lhs
        self.rhs = rhs
        self.operator = operator

    def infer_types(self, model, context_type):
        if hasattr(self.lhs, 'infer_types'):
            self.lhs.infer_types(model, context_type)
            lhs_type = self.lhs.type
        else:
            lhs_type = TScalar
        if hasattr(self.rhs, 'infer_types'):
            self.rhs.infer_types(model, context_type)
            rhs_type = self.rhs.type
        else:
            rhs_type = TScalar

        if lhs_type == rhs_type:
            self.type = lhs_type
        elif lhs_type == TScalar:
            self.type = rhs_type
        elif rhs_type == TScalar:
            self.type = lhs_type
        else:
            raise SpaTypeError("Incompatible types in multiply.")

    def __str__(self):
        return '({} {} {})'.format(self.lhs, self.operator, self.rhs)


class Product(BinaryOperation):
    def __init__(self, lhs, rhs):
        super(Product, self).__init__(lhs, rhs, '*')


class Sum(BinaryOperation):
    def __init__(self, lhs, rhs):
        super(Sum, self).__init__(lhs, rhs, '+')


class UnaryOperation(Source):
    def __init__(self, source, symbol):
        super(UnaryOperation, self).__init__()
        self.source = source
        self.symbol = symbol

    def infer_types(self, model, context_type):
        self.source.infer_types(model, context_type)
        self.type = self.source.type

    def __str__(self):
        return self.symbol + str(self.source)


class ApproxInverse(UnaryOperation):
    def __init__(self, source):
        super(ApproxInverse, self).__init__(source, '~')

    def infer_types(self, model, context_type):
        super(ApproxInverse, self).infer_types(model, context_type)
        if not isinstance(self.type, TVocabulary):
            raise SpaTypeError(
                "Can apply approximate inverse only to semantic pointers.")


class Negative(UnaryOperation):
    def __init__(self, source):
        super(Negative, self).__init__(source, '-')


class DotProduct(Source):
    def __init__(self, lhs, rhs):
        super(DotProduct, self).__init__()
        self.type = TScalar
        self.lhs = lhs
        self.rhs = rhs

    def infer_types(self, model, context_type):
        self.lhs.infer_types(model, context_type)
        self.rhs.infer_types(model, context_type)
        if self.lhs.type.vocab is not self.rhs.type.vocab:
            raise SpaTypeError(
                "Dot product of pointers from different vocabularies.")

    def __str__(self):
        return 'dot({}, {})'.format(self.lhs, self.rhs)


class Effect(Node):
    def __init__(self, sink, source):
        super(Effect, self).__init__()
        self.type = TEffect
        self.sink = sink
        self.source = source

    def infer_types(self, model, context_type):
        if context_type is None:
            raise ValueError("Effect only allows a context_type of None.")
        self.sink.infer_types(model, None)
        self.source.infer_types(model, self.sink.type)

    def __str__(self):
        return '{} = {}'.format(self.sink, self.source)


class Sink(Node):
    def __init__(self, name):
        super(Sink, self).__init__()
        self.name = name

    def infer_types(self, model, context_type):
        try:
            vocab = model.get_module_input(self.name)[1]
        except SpaModuleError:
            vocab = model.get_module_input(self.name + '.default')[1]

        if vocab is None:
            self.type = TScalar
        else:
            self.type = TVocabulary(vocab)

    def __str__(self):
        return self.name


class Action(Node):
    def __init__(self, condition, effect):
        super(Action, self).__init__()
        self.type = TAction
        self.condition = condition
        self.effect = effect

    def infer_types(self, model, context_type):
        self.condition.infer_types(model, context_type)
        self.effect.infer_types(model, None)

        if self.condition.type != TScalar:
            raise SpaTypeError("Condition has to evaluate to a scalar.")

    def __str__(self):
        return '{} --> {}'.format(self.condition, self.effect)
