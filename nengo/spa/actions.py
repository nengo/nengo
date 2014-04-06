from nengo.spa.action_condition import Condition
from nengo.spa.action_effect import Effect
from nengo.utils.compat import iteritems


class Action(object):
    """A single action.

    Consists of a Condition and an Effect.  The Condition is optional.

    Parameters
    ----------
    sources : list of strings
        The names of valid sources of information (SPA module outputs)
    sinks : list of string
        The names of valid places to send information (SPA module inputs)
    action : string
        A string defining the action.  If '-->' is in the string, this
        is used as a marker to split the string into Condition and Effect.
        Otherwise it is treated as having no Condition and just Effect.
    name : string
        The name of this action
    """
    def __init__(self, sources, sinks, action, name):
        self.name = name
        if name is None:
            name = action    # only used for raised Exceptions below

        try:
            if '-->' in action:
                condition, effect = action.split('-->', 1)
                self.condition = Condition(sources, condition)
                self.effect = Effect(sources, effect)
            else:
                self.condition = None
                self.effect = Effect(sources, action)
        except NameError as e:
            raise NameError('Unknown module referenced in action "%s": %s' %
                            (name, e))
        except TypeError as e:
            raise TypeError('Invalid operator in action "%s": %s' %
                            (name, e))

        for sink_name in self.effect.effect.keys():
            if sink_name not in sinks:
                raise NameError(('Unknown module effects in action "%s": ' +
                                 " name '%s' is not defined") %
                                (name, sink_name))

    def __str__(self):
        return '<Action %s:\n  Condition: %s\n  Effect: %s\n>' % (
            self.name, self.condition, self.effect)


class Actions(object):
    """A collection of Action objects.

    The *args and **kwargs are treated as unnamed and named Actions,
    respectively.  The list of actions are only generated once process()
    is called, since it needs access to the list of module inputs and
    outputs from the SPA object.
    """
    def __init__(self, *args, **kwargs):
        self.actions = None
        self.args = args
        self.kwargs = kwargs

    @property
    def count(self):
        """Return the number of actions."""
        return len(self.args) + len(self.kwargs)

    def process(self, spa):
        """Parse the actions and generate the list of Action objects."""
        self.actions = []

        sources = list(spa.get_module_outputs())
        sinks = list(spa.get_module_inputs())

        for action in self.args:
            self.actions.append(Action(sources, sinks, action, name=None))
        for name, action in iteritems(self.kwargs):
            self.actions.append(Action(sources, sinks, action, name=name))
