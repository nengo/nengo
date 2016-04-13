"""Expressions and Effects used to define all Actions."""

import re
import warnings
from collections import OrderedDict

from nengo.exceptions import SpaModuleError, SpaParseError
from nengo.spa import action_objects as ao
import nengo.spa.spa_ast
from nengo.spa.spa_ast import DotProduct, Module, Sink, Symbol
from nengo.utils.compat import iteritems


class Expression(object):
    """Parses an Action expression given a set of module outputs.

    Parameters
    ----------
    module : :class:`nengo.spa.Module`
        The SPA module used to look up names.
    expression : str
        The expression to evaluate. This either defines the utility of the
        action, or a value from an effect's assignment, given the state
        information from the module outputs. The simplest expression is
        ``"1"`` and they can get more complex, such as
        ``"0.5*(dot(vision, DOG) + dot(memory, CAT*MOUSE)*3 - 1)"``.
    """

    def __init__(self, module, expression):
        self.module = module
        self.objects = {}   # the list of known terms

        # handle the term 'dot(a, b)' to mean DotProduct(a, b)
        self.objects['dot'] = ao.DotProduct

        # use Python's eval to do the parsing of expressions for us
        self.validate_string(expression)
        sanitized_exp = ' '.join(expression.split('\n'))
        try:
            self.expression = eval(sanitized_exp, {}, self)
        except NameError as e:
            raise SpaParseError("Unknown module in expression '%s': %s" %
                                (expression, e))
        except TypeError as e:
            raise SpaParseError("Invalid operator in expression '%s': %s" %
                                (expression, e))

        # normalize the result to a summation
        if not isinstance(self.expression, ao.Summation):
            self.expression = ao.Summation([self.expression])

    def validate_string(self, text):
        m = re.search('~[^a-zA-Z]', text)
        if m is not None:
            raise SpaParseError("~ is only permitted before names (e.g., DOG) "
                                "or modules (e.g., vision): %s" % text)

    def __getitem__(self, key):
        # this gets used by the eval in the constructor to create new
        # terms as needed
        if key == '__tracebackhide__':  # gives better tracebacks in py.test
            return False
        if key in self.objects:
            return self.objects[key]
        else:
            try:
                return ao.Namespace(key, module=self.module.get_module(key))
            except SpaModuleError:
                if not key[0].isupper():
                    raise SpaParseError(
                        "Semantic pointers must begin with a capital "
                        "letter.")
                item = ao.Symbol(key)
                self.objects[key] = item
                return item

    def __str__(self):
        return str(self.expression)


class Effect(object):
    """Parses an action effect given a set of module outputs.

    The following, in an `.Action` string, are valid effects::

        "motor=A"
        "motor=A*B, memory=vision+DOG"
        "motor=0.5*(memory*A + vision*B)"

    Parameters
    ----------
    module : :class:`nengo.spa.Module`
        The SPA module used to look up names.
    effect: str
        The action to implement. This is a set of assignment statements
        which can be parsed into a `.VectorList`.
    """

    def __init__(self, module, effect):
        self.effect = OrderedDict()
        # Splits by ',' and separates into lvalue=rvalue. We cannot simply use
        # split, because the rvalue may contain commas in the case of dot(*,*).
        # However, *? is lazy, and * is greedy, making this regex work.
        for lvalue, rvalue in re.findall("(.*?)=([^=]*)(?:,|$)", effect):
            sink = lvalue.strip()
            try:
                module.get_module_input(sink)
            except SpaModuleError:
                raise SpaParseError(
                    "Left-hand module '%s' from effect '%s=%s' "
                    "is not defined." %
                    (lvalue, lvalue, rvalue))
            if sink in self.effect:
                raise SpaParseError(
                    "Left-hand module '%s' from effect '%s=%s' "
                    "is assigned to multiple times in '%s'." %
                    (lvalue, lvalue, rvalue, effect))
            self.effect[sink] = Expression(module, rvalue)

    def __str__(self):
        return ", ".join("%s=%s" % x for x in iteritems(self.effect))


class Parser(object):
    builtins = {'dot': DotProduct}

    def parse_action(self, action):
        try:
            condition, effect = action.split('-->', 1)
        except ValueError:
            raise SpaParseError("Not an action, '-->' missing.")
        return nengo.spa.spa_ast.Action(
            self.parse_expr(condition), self.parse_effect(effect))

    def parse_effect(self, effect):
        try:
            sink, source = effect.split('=', 1)
        except ValueError:
            raise SpaParseError("Not an effect; assignment missing")
        return nengo.spa.spa_ast.Effect(
            Sink(sink.strip()), self.parse_expr(source))

    def parse_expr(self, expr):
        return eval(expr, {}, self)

    def __getitem__(self, key):
        if key == '__tracebackhide__':  # gives better tracebacks in py.test
            return False
        if key in self.builtins:
            return self.builtins[key]
        if key[0].isupper():
            return Symbol(key)
        else:
            return Module(key)


class Action(object):
    """A single action.

    Consists of a conditional `.Expression` (optional) and an `.Effect`.

    Parameters
    ----------
    module : :class:`nengo.spa.Module`
        The SPA module used to look up names.
    action : str
        A string defining the action.  If ``'-->'`` is in the string, this
        is used as a marker to split the string into condition and effect.
        Otherwise it is treated as having no condition and just effect.
    name : str
        The name of this action.
    """

    def __init__(self, module, action, name):
        self.name = name
        if '-->' in action:
            condition, effect = action.split('-->', 1)
            self.condition = Expression(module, condition)
            self.effect = Effect(module, effect)
        else:
            self.condition = None
            self.effect = Effect(module, action)

    def __str__(self):
        return "<Action %s:\n  %s\n --> %s\n>" % (
            self.name, self.condition, self.effect)


class Actions(object):
    """A collection of Action objects.

    The ``*args`` and ``**kwargs`` are treated as unnamed and named actions,
    respectively. The list of actions are only generated once
    `~.Actions.process` is called, since it needs access to the list of
    module inputs and outputs from the SPA object. The ``**kwargs`` are sorted
    alphabetically before being processed.
    """

    def __init__(self, *args, **kwargs):
        self.actions = None
        self.args = args
        self.kwargs = kwargs

    def add(self, *args, **kwargs):
        if self.actions is not None:
            warnings.warn("The actions currently being added must be processed"
                          " either by spa.BasalGanglia or spa.Cortical"
                          " to be added to the model.")

        self.args += args
        self.kwargs.update(kwargs)

    @property
    def count(self):
        """Return the number of actions."""
        return len(self.args) + len(self.kwargs)

    def process(self, module):
        """Parse the actions and generate the list of Action objects."""
        self.actions = []

        sorted_kwargs = sorted(self.kwargs.items())

        for action in self.args:
            self.actions.append(Action(module, action, name=None))
        for name, action in sorted_kwargs:
            self.actions.append(Action(module, action, name=name))
