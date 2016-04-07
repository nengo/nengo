"""Syntactic parsing of the subexpressions of all action expressions."""

from nengo.exceptions import SpaParseError
from nengo.utils.compat import is_number


class Symbol(object):
    """A set of semantic pointer symbols and associated math

    This is an abstract semantic pointer (not associated with a particular
    vocabulary or dimension).  It is just meant for keeping track of the
    desired manipulations until such time as it is parsed with a particular
    Vocabulary.

    Its contents are a single string, and this string is manipulated via
    standard mathematical operators (+ - * ~) for SemanticPointers.  The
    result will always be able to be passed to a Vocabulary's parse()
    method to get a valid SemanticPointer.

    This is used by the spa.Action parsing system.
    """

    def __init__(self, symbol):
        self.symbol = symbol

    def __add__(self, other):
        if is_number(other):
            other = Symbol('%g' % other)
        if isinstance(other, Symbol):
            return Symbol('(%s + %s)' % (self.symbol, other.symbol))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Symbol):
            return Symbol('(%s - %s)' % (self.symbol, other.symbol))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Symbol):
            if other.symbol == '1':
                return self
            if self.symbol == '1':
                return other
            return Symbol('(%s * %s)' % (self.symbol, other.symbol))
        if is_number(other):
            if other == 1:
                return self
            if self.symbol == '1':
                return Symbol('%g' % other)
            return Symbol('(%s * %g)' % (self.symbol, other))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __invert__(self):
        if self.symbol.startswith('~'):
            return Symbol(self.symbol[1:])
        else:
            return Symbol('~%s' % self.symbol)

    def __neg__(self):
        if self.symbol.startswith('-'):
            return Symbol(self.symbol[1:])
        else:
            return Symbol('-%s' % self.symbol)

    def __str__(self):
        return str(self.symbol)


class Source(object):
    """A particular source of a vector for the Action system.

    This will always refer to a particular named output from a spa.Module.
    It also tracks a single Symbol which represents a desired transformation
    from that output.  For example, Source('vision')*Symbol('VISION') will
    result in a Source object for 'vision', but with transform set to the
    Symbol('VISION').

    This is used by the spa.Action parsing system.
    """

    def __init__(
            self, name, transform=Symbol('1'), inverted=False, module=None):
        self.name = name            # the name of the module output
        self.module = module
        self.transform = transform  # the Symbol for the transformation
        self.inverted = inverted

    def __getattr__(self, name):
        if self.module is None:
            raise AttributeError('{0!r} has no submodules.'.format(self.name))
        return Source(
            '{}.{}'.format(self.name, name),
            module=self.module.get_module(name))

    def __invert__(self):
        if self.transform.symbol != '1':
            raise SpaParseError(
                "You can only invert sources without transforms")
        return Source(self.name, self.transform, not self.inverted)

    def __mul__(self, other):
        if isinstance(other, Source):
            return Convolution(self, other)
        elif is_number(other) or isinstance(other, Symbol):
            return Source(self.name, self.transform*other, self.inverted)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Source(self.name, -self.transform, self.inverted)

    def __add__(self, other):
        return Summation([self]).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __str__(self):
        if self.transform.symbol == '1':
            trans_text = ""
        else:
            trans_text = "%s * " % self.transform
        if self.inverted:
            trans_text += "~"
        return "%s%s" % (trans_text, self.name)


class DotProduct(object):
    """The dot product of a Source and a Source or a Source and a Symbol.

    This represents a similarity measure for computing the utility of
    and action.  It also maintains a scaling factor on the result,
    so that the 0.5 in "0.5*DotProduct(Source('vision'), Symbol('DOG'))"
    can be correctly tracked.

    This class is meant to be used with an eval-based parsing system in the
    Condition class, so that the above DotProduct can also be created with
    "0.5*dot(vision,DOG)".
    """

    def __init__(self, item1, item2, scale=1.0):
        if isinstance(item1, (int, float)):
            item1 = Symbol(item1)
        if isinstance(item2, (int, float)):
            item2 = Symbol(item2)
        if not isinstance(item1, (Source, Symbol)):
            raise SpaParseError("The first item in the dot product is not a "
                                "semantic pointer or a spa.Module output.")
        if not isinstance(item2, (Source, Symbol)):
            raise SpaParseError("The second item in the dot product is not a "
                                "semantic pointer or a spa.Module output.")
        if not isinstance(item1, Source) and not isinstance(item2, Source):
            raise SpaParseError("One of the two terms for the dot product "
                                "must be a spa.Module output.")
        self.item1 = item1
        self.item2 = item2
        self.scale = float(scale)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DotProduct(self.item1, self.item2, self.scale * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__mul__(1.0 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        return DotProduct(self.item1, self.item2, -self.scale)

    def __add__(self, other):
        return Summation([self]).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __str__(self):
        if self.scale == 1:
            scale_text = ""
        elif self.scale == -1:
            scale_text = "-"
        else:
            scale_text = "%g * " % self.scale
        return "%sdot(%s, %s)" % (scale_text, self.item1, self.item2)


class Convolution(object):
    """The convolution of two sources together."""

    def __init__(self, source1, source2, transform=Symbol('1')):
        self.source1 = source1
        self.source2 = source2
        self.transform = transform

    def __mul__(self, other):
        if isinstance(other, (Symbol, int, float)):
            return Convolution(
                self.source1, self.source2, self.transform * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return Convolution(self.source1, self.source2, -self.transform)

    def __add__(self, other):
        return Summation([self]).__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __str__(self):
        return "((%s) * (%s)) * %s" % (
            self.source1, self.source2, self.transform)


class Summation(object):
    """A summation over all subexpressions."""

    def __init__(self, items):
        self.items = items

    def __mul__(self, other):
        return Summation([x*other for x in self.items])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__mul__(1.0 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        if isinstance(other,
                      (int, float, Source, Symbol, DotProduct, Convolution)):
            return Summation(self.items + [other])
        if isinstance(other, Summation):
            return Summation(self.items + other.items)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return Summation([-x for x in self.items])

    def __str__(self):
        return " + ".join(str(v) for v in self.items)
