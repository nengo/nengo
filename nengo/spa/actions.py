"""Parsing of SPA actions."""

import warnings

from nengo.config import Config
from nengo.exceptions import SpaParseError
from nengo.spa.basalganglia import BasalGanglia
from nengo.spa.cortical import Cortical
from nengo.spa.thalamus import Thalamus
from nengo.spa.spa_ast import (
    Action, ConstructionContext, DotProduct, Effect, Effects, Module,
    Reinterpret, Sink, Symbol, Translate)


class Parser(object):
    """Parser for SPA actions.

    Parameters
    ----------
    vocabs : dict
        Vocabularies to make accessible in the parsed action rules.
    """

    builtins = {
        'dot': DotProduct,
        'reinterpret': Reinterpret,
        'translate': Translate}

    def __init__(self, vocabs=None):
        if vocabs is None:
            vocabs = {}
        self.vocabs = vocabs

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
                self.parse_effects(effects, channeled=True), index, name=name)

    def parse_effects(self, effects, channeled=False):  # noqa: C901
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
        if key in self.vocabs:
            return self.vocabs[key]
        if key in self.builtins:
            return self.builtins[key]
        if key[0].isupper():
            return Symbol(key)
        else:
            return Module(key)


class Actions(Config):
    """A collection of Action objects.

    The *args and **kwargs are treated as unnamed and named Actions,
    respectively.  The list of actions are only generated once process()
    is called, since it needs access to the list of module inputs and
    outputs from the SPA object. The **kwargs are sorted alphabetically before
    being processed.

    The keyword argument `vocabs` is special in that it provides a dictionary
    mapping names to vocabularies. The vocabularies can then be used with those
    names in the action rules.
    """

    def __init__(self, *args, **kwargs):
        super(Actions, self).__init__(Cortical, BasalGanglia, Thalamus)

        self.actions = []
        self.effects = []

        self.args = args
        self.kwargs = kwargs
        if 'vocabs' in kwargs:
            self.vocabs = self.kwargs.pop('vocabs')
        else:
            self.vocabs = None
        self.construction_context = None

        self.parse(self.vocabs, *self.args, **self.kwargs)

    def parse(self, vocabs, *args, **kwargs):
        sorted_kwargs = sorted(kwargs.items())

        parser = Parser(vocabs=vocabs)
        for action in args:
            self._parse_and_add(parser, action)
        for name, action in sorted_kwargs:
            self._parse_and_add(parser, action, name=name)

    def add(self, *args, **kwargs):
        if 'vocabs' in kwargs:
            vocabs = self.kwargs.pop('vocabs')
            self.vocabs.update(vocabs)
        else:
            vocabs = None
        self.args += args
        self.kwargs.update(kwargs)
        self.parse(vocabs, *args, **kwargs)

    @property
    def count(self):
        """Return the number of actions."""
        warnings.warn(DeprecationWarning(
            "Use len(Actions.actions) or len(Actions.effects)."))
        return len(self.args) + len(self.kwargs)

    def process(self):
        """Parse the actions and generate the list of Action objects."""
        self.process_new_actions(self.vocabs, *self.args, **self.kwargs)

    def process_new_actions(self, vocabs=None, *args, **kwargs):
        with self.construction_context.root_module:
            for action in self.effects + self.actions:
                action.infer_types(self.construction_context.root_module, None)
            # Infer types for all actions before doing any construction, so
            # that all semantic pointers are added to the respective
            # vocabularies so that the translate transform are identical.
            for action in self.effects + self.actions:
                action.construct(self.construction_context)

    def _parse_and_add(self, parser, action, name=None):
        ast = parser.parse_action(
            action, len(self.actions), strict=False, name=name)
        if isinstance(ast, Effects):
            self.effects.append(ast)
        else:
            self.actions.append(ast)

    def build(self, root_module, cortical=None, bg=None, thalamus=None):
        needs_cortical = len(self.effects) > 0
        needs_bg = len(self.actions) > 0

        with root_module, self:
            if needs_cortical and cortical is None:
                cortical = Cortical()
                root_module.cortical = cortical
            if needs_bg and bg is None:
                bg = BasalGanglia(action_count=len(self.actions))
                root_module.bg = bg
            if needs_bg and thalamus is None:
                thalamus = Thalamus(bg)
                root_module.thalamus = thalamus

        self.construction_context = ConstructionContext(
            root_module, cortical=cortical, bg=bg, thalamus=thalamus)
        with root_module:
            for action in self.effects + self.actions:
                action.infer_types(root_module, None)
            # Infer types for all actions before doing any construction, so
            # that # all semantic pointers are added to the respective
            # vocabularies so that the translate transform are identical.
            for action in self.effects + self.actions:
                action.construct(self.construction_context)
