import inspect
import re

import numpy as np

from nengo.utils.compat import is_number, is_string, iteritems


class Input(object):
    def __init__(self, name, obj, vocab, invert=False):
        self.name = name
        self.obj = obj
        self.vocab = vocab
        self.invert = invert

    def __mul__(self, other):
        if is_string(other):
            return TransformedInput(
                self.name, self.obj, self.vocab, self.invert, other)
        elif isinstance(other, Input):
            return ConvolvedInput(
                self.name, self.obj, self.vocab, self.invert, other)
        elif is_number(other):
            return TransformedInput(
                self.name, self.obj, self.vocab, self.invert, '%g' % other)
        else:
            raise ValueError('Rule error: multiplication of an Input ("%s") '
                             'by unknown term ("%s")' % (self.name, other))

    def __invert__(self):
        return Input(self.name, self.obj, self.vocab, not self.invert)


class TransformedInput(object):
    def __init__(self, name, obj, vocab, invert, transform):
        self.name = name
        self.obj = obj
        self.vocab = vocab
        self.invert = invert
        self.transform = transform

    def __invert__(self):
        return TransformedInput(
            self.name, self.obj, self.vocab, not self.invert, self.transform)


class ConvolvedInput(object):
    def __init__(self, name, obj, vocab, invert, convolve):
        self.name = name
        self.obj = obj
        self.vocab = vocab
        self.invert = invert
        self.convolve = convolve

    def __invert__(self):
        return ConvolvedInput(
            self.name, self.obj, self.vocab, not self.invert, self.convolve)


class Output(object):
    def __init__(self, name, obj, vocab):
        self.name = name
        self.obj = obj
        self.vocab = vocab


class Rule(object):
    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.matches = {}
        self.effects_direct = {}
        self.effects_route = []

        code = inspect.getsource(function)
        m = re.match(r'[^(]+\([^(]*\):', code)
        code = 'if True:' + code[m.end():]
        self.rule = compile(code, '<production-%s>' % self.name, 'exec')

    def match(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception('invalid match in rule "%s"' % self.name)
        for k, v in iteritems(kwargs):
            if k not in self.inputs:
                raise Exception('No module named "%s" found for match in '
                                'rule "%s"' % (k, self.name))
            assert k not in self.matches
            self.matches[k] = v

    def effect(self, *args, **kwargs):
        if len(args) > 0:
            raise Exception('invalid effect in rule "%s"' % self.name)
        for k, v in iteritems(kwargs):
            if k not in self.outputs:
                raise KeyError('No module named "%s" found for effect in '
                               'rule "%s"' % (k, self.name))
            if is_string(v):
                assert k not in self.effects_direct
                self.effects_direct[k] = v
            elif isinstance(v, (Input, TransformedInput, ConvolvedInput)):
                self.effects_route.append((self.outputs[k], v))
            else:
                raise Exception('Unknown effect "%s=%s" found in rule "%s"'
                                % (k, v, self.name))

    def process(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        globals = dict(match=self.match, effect=self.effect)
        globals.update(inputs)

        eval(self.rule, {}, globals)


class Rules(object):
    def __init__(self, rules):
        self.rules = []
        for name, func in inspect.getmembers(rules):
            if callable(func):
                if not name.startswith('__'):
                    self.rules.append(Rule(name, func))

    @property
    def count(self):
        return len(self.rules)

    def process(self, spa):
        self.inputs = {}
        self.outputs = {}
        for name, m in iteritems(spa._modules):
            for label, (obj, vocab) in iteritems(m.outputs):
                n = name
                if label != 'default':
                    n += '_' + label
                self.inputs[n] = Input(n, obj, vocab)
            for label, (obj, vocab) in iteritems(m.inputs):
                n = name
                if label != 'default':
                    n += '_' + label
                self.outputs[n] = Output(n, obj, vocab)

        for rule in self.rules:
            rule.process(self.inputs, self.outputs)

    def get_inputs(self):
        inputs = {}

        for name, input in iteritems(self.inputs):
            transform = []
            assert input.vocab is not None

            for rule in self.rules:
                if name in rule.matches:
                    row = input.vocab.parse(rule.matches[name]).v
                else:
                    row = [0] * input.vocab.dimensions
                transform.append(row)
            transform = np.array(transform)
            if np.count_nonzero(transform) > 0:
                inputs[input.obj] = transform
        return inputs

    def get_outputs_direct(self):
        outputs = {}

        for name, output in iteritems(self.outputs):
            transform = []
            assert output.vocab is not None

            for rule in self.rules:
                if name in rule.effects_direct:
                    row = output.vocab.parse(rule.effects_direct[name]).v
                else:
                    row = [0] * output.vocab.dimensions
                transform.append(row)
            transform = np.array(transform)
            if np.count_nonzero(transform) > 0:
                outputs[output.obj] = transform.T
        return outputs

    def get_outputs_route(self):
        for i, rule in enumerate(self.rules):
            for route in rule.effects_route:
                yield i, route

    @property
    def names(self):
        return [r.name for r in self.rules]
