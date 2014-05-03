from __future__ import print_function

import re

from nengo.spa.action_objects import Symbol, Source
from nengo.utils.compat import is_number, iteritems


class SourceWithAddition(Source):
    """A Source that handles addition with other Sources and Symbols.

    This is needed because action effects can do "motor = vision + memory + A"
    but we do not support this for conditions, as it is not clear what
    "dot(vision + memory + B, A)" should mean.
    """
    def __init__(self, name, transform=Symbol('1'), inverted=False):
        super(SourceWithAddition, self).__init__(name, transform)
        self.inverted = inverted

    def __add__(self, other):
        if is_number(other):
            other = Symbol('%g' % other)
        if isinstance(other, (Symbol, Source, CombinedSource)):
            return VectorList([self, other])
        else:
            return NotImplemented

    def __invert__(self):
        if self.transform.symbol != '1':
            raise TypeError('You can only invert sources without transforms')
        return SourceWithAddition(self.name, self.transform, not self.inverted)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return SourceWithAddition(self.name, transform=-self.transform,
                                  inverted=self.inverted)

    def __mul__(self, other):
        if isinstance(other, SourceWithAddition):
            return CombinedSource(self, other)
        elif isinstance(other, (Symbol, int, float)):
            return SourceWithAddition(self.name,
                                      transform=self.transform * other,
                                      inverted=self.inverted)
        else:
            return NotImplemented

    def __str__(self):
        if self.transform.symbol == '1':
            trans_text = ''
        else:
            trans_text = '%s * ' % self.transform
        if self.inverted:
            trans_text += '~'
        return '%s%s' % (trans_text, self.name)


class CombinedSource(object):
    """The convolution of two sources together"""
    def __init__(self, source1, source2, transform=Symbol('1')):
        self.source1 = source1
        self.source2 = source2
        self.transform = transform

    def __mul__(self, other):
        if is_number(other) or isinstance(other, Symbol):
            return CombinedSource(self.source1, self.source2,
                                  self.transform * other)
        else:
            return NotImplemented

    def __rmul__(self, other):
        if is_number(other) or isinstance(other, Symbol):
            return CombinedSource(self.source1, self.source2,
                                  self.transform * other)
        else:
            return NotImplemented

    def __neg__(self):
        return CombinedSource(self.source1, self.source2, -self.transform)

    def __add__(self, other):
        if is_number(other):
            other = Symbol('%g' % other)
        if isinstance(other, (Symbol, Source, CombinedSource)):
            return VectorList([self, other])
        else:
            return NotImplemented

    def __invert__(self):
        if self.symbol.startswith('~'):
            return Symbol(self.symbol[1:])
        else:
            return Symbol('~%s' % self.symbol)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __str__(self):
        return '((%s) * (%s)) * %s' % (
            self.source1, self.source2, self.transform)


class VectorList(object):
    """A list of Symbols and Sources.

    Used to handle multiple effects on the same module input, such as
    "motor = vision + A"
    """
    def __init__(self, items):
        self.items = items

    def __mul__(self, other):
        return VectorList([x*other for x in self.items])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__mul__(1.0/other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        if is_number(other):
            other = Symbol('%g' % other)
        if isinstance(other, (Symbol, Source, CombinedSource)):
            return VectorList(self.items + [other])
        elif isinstance(other, VectorList):
            return VectorList(self.items + other.items)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return VectorList([-x for x in self.items])

    def __str__(self):
        return ' + '.join([str(v) for v in self.items])


class Effect(object):
    """Parses an Action effect given a set of module outputs.

    Parameters
    ----------
    sources : list of strings
        The names of the module outputs that can be used as part of the
        effect (i.e. the sources of vectors that can build up the effects)
    effect: string
        The action to implement.  This is a set of assignment statements
        which can be parsed into a VectorList.

    The following are valid effects:
        "motor=A"
        "motor=A*B, memory=vision+DOG"
        "motor=0.5*(memory*A + vision*B)"
    """
    def __init__(self, sources, effect):
        self.objects = {}     # the list of known terms

        # make terms for all the known module outputs
        for name in sources:
            self.objects[name] = SourceWithAddition(name)

        # the effect would be parsed as "dict(%s)" % effect so that it
        # will correctly handle the commas and naming of the module inputs
        # the effects go to (such as "motor" in "motor=A").  However,
        # that would cause a naming conflict if someone has a module
        # called 'dict'.  So we use '__effect_dictionary' instead of 'dict'.
        self.objects['__effect_dictionary'] = dict

        self.validate_string(effect)
        # do the parsing
        self.effect = eval('__effect_dictionary(%s)' % effect, {}, self)

        # ensure that all the results are VectorLists
        for k, v in iteritems(self.effect):
            if is_number(v):
                v = Symbol('%g' % v)
            if isinstance(v, (Symbol, Source, CombinedSource)):
                self.effect[k] = VectorList([v])
            assert isinstance(self.effect[k], VectorList)

    def __getitem__(self, key):
        # this gets used by the eval in the constructor to create new
        # terms as needed
        item = self.objects.get(key, None)
        if item is None:
            if not key[0].isupper():
                raise KeyError('Semantic pointers must begin with a capital')
            item = Symbol(key)
            self.objects[key] = item
        return item

    def __str__(self):
        return ', '.join(['%s=%s' % x for x in iteritems(self.effect)])

    def validate_string(self, text):
        m = re.search('~[^a-zA-Z]', text)
        if m is not None:
            raise TypeError('~ is only permitted before names (e.g., DOG) or '
                            'modules (e.g., vision)')
