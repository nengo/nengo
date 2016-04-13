"""Parsing of SPA actions."""
from nengo.exceptions import SpaParseError
from nengo.spa.spa_ast import (
    Action, DotProduct, Effect, Effects, Module, Sink, Symbol)


class Parser(object):
    """Parser for SPA actions."""

    builtins = {'dot': DotProduct}

    def parse_action(self, action, index=0, name=None, strict=True):
        """Parse an SPA action.

        Parameters
        ----------
        action : str
            Action to parse.
        index : int, optional
            Index of the action for identification by basal ganglia and
            thalamus.
        name : str, optional
            Name of the action.
        strict : bool, optional
            If ``True`` only actual conditional actions are allowed and an
            exception will be raised for anything else. If ``False``, allows
            also the parsing of effects without the conditional part.

        Returns
        -------
        :class:`spa_ast.Action` or :class:`spa_ast.Effects`
        """
        try:
            condition, effects = action.split('-->', 1)
        except ValueError:
            if strict:
                raise SpaParseError("Not an action, '-->' missing.")
            else:
                return self.parse_effects(action, channeled=False)
        else:
            return Action(
                self.parse_expr(condition),
                self.parse_effects(effects, channeled=True), index)

    def parse_effects(self, effects, channeled=False):
        """Pares SPA effects.

        Parameters
        ----------
        effects : str
            Effects to parse.
        channeled : bool, optional
            Whether the effects should be passed through channels when
            constructed.

        Returns
        -------
        :class:`spa_ast.Effects`
        """
        parsed = []
        symbol_stack = []
        start = 0
        for i, c in enumerate(effects):
            top = symbol_stack[-1] if len(symbol_stack) > 0 else None
            if top == '\\':  # escaped character, ignore
                symbol_stack.pop()
            elif top is not None and top in '\'"':  # in a string
                if c == '\\':  # escape
                    symbol_stack.append(c)
                elif c == top:  # end string
                    symbol_stack.pop()
            else:
                if c in '\'"':  # start string
                    symbol_stack.append(c)
                elif c in '([':  # start parens/brackets
                    symbol_stack.append(c)
                elif c in ')]':  # end parens/brackets
                    if (top == '(' and c != ')') or (top == '[' and c != ']'):
                        raise SpaParseError("Parenthesis mismatch.")
                    symbol_stack.pop()
                elif c == ',' and len(symbol_stack) == 0:  # effect delimiter
                    parsed.append(effects[start:i])
                    start = i + 1
        parsed.append(effects[start:])

        if len(symbol_stack) != 0:
            top = symbol_stack.pop()
            if top in '([':
                raise SpaParseError("Parenthesis mismatch.")
            elif top in '\'"':
                raise SpaParseError("Unclosed string.")
            else:
                raise SpaParseError("Unmatched: " + top)

        return Effects(*[self.parse_effect(effect, channeled=channeled)
                         for effect in parsed])

    def parse_effect(self, effect, channeled=False):
        """Parse single SPA effect.

        Parameters
        ----------
        effect : str
            Effect to parse.
        channeled : bool, optional
            Whether the effect should be passed through a channel when
            constructed.

        Returns
        -------
        :class:`spa_ast.Effect`
        """
        try:
            sink, source = effect.split('=', 1)
        except ValueError:
            raise SpaParseError("Not an effect; assignment missing")
        return Effect(
            Sink(sink.strip()), self.parse_expr(source), channeled=channeled)

    def parse_expr(self, expr):
        """Parse an SPA expression.

        Parameters
        ----------
        expr : str
            Expression to parse.

        Returns
        -------
        :class:`spa_ast.Source`
        """
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
        self.construction_context = None

    def add(self, *args, **kwargs):
        self.args += args
        self.kwargs.update(kwargs)
        self._process_new_actions(*args, **kwargs)

    @property
    def count(self):
        """Return the number of actions."""
        return len(self.args) + len(self.kwargs)

    def process(self):
        """Parse the actions and generate the list of Action objects."""
        self.actions = []
        self._process_new_actions(*self.args, **self.kwargs)

    def _process_new_actions(self, *args, **kwargs):
        sorted_kwargs = sorted(kwargs.items())

        parser = Parser()
        for i, action in enumerate(args):
            self.actions.append(parser.parse_action(action, i, strict=False))
        for i, (name, action) in enumerate(sorted_kwargs, start=self.count):
            self.actions.append(parser.parse_action(
                action, i, name=name, strict=False))

        for action in self.actions:
            action.infer_types(self.construction_context.root_module, None)
        # Infer types for all actions before doing any construction, so that
        # all semantic pointers are added to the respective vocabularies so
        # that the translate transform are identical.
        for action in self.actions:
            action.construct(self.construction_context)
