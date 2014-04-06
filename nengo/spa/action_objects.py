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
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Symbol):
            return Symbol('(%s - %s)' % (self.symbol, other.symbol))
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Symbol):
            if other.symbol == '1':
                return self
            if self.symbol == '1':
                return other
            return Symbol('(%s * %s)' % (self.symbol, other.symbol))
        elif is_number(other):
            if other == 1:
                return self
            if self.symbol == '1':
                return Symbol('%g' % other)
            return Symbol('(%s * %g)' % (self.symbol, other))
        else:
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
        return self.symbol


class Source(object):
    """A particular source of a vector for the Action system.

    This will always refer to a particular named output from a spa.Module.
    It also tracks a single Symbol which represents a desired transformation
    from that output.  For example, Source('vision')*Symbol('VISION') will
    result in a Source object for 'vision', but with transform set to the
    Symbol('VISION').

    This is used by the spa.Action parsing system.
    """
    def __init__(self, name, transform=Symbol('1')):
        self.name = name            # the name of the module output
        self.transform = transform  # the Symbol for the transformation

    def __mul__(self, other):
        if is_number(other) or isinstance(other, Symbol):
            return self.__class__(self.name, transform=self.transform*other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__class__(self.name, transform=-self.transform)

    def __str__(self):
        if self.transform.symbol == '1':
            trans_text = ''
        else:
            trans_text = '%s * ' % self.transform
        return '%s%s' % (trans_text, self.name)
