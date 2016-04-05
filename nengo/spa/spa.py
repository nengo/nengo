import numpy as np

from nengo.spa.vocab import VocabularyMap
from nengo.spa.module import Module


class SPA(Module):
    """Base class for SPA models.

    This expands the standard `.Network` system to support structured
    connections that use Semantic Pointers and associated vocabularies
    in their definitions.

    To build a SPA model, you can either use ``with`` or create a subclass
    of this SPA class.

    If you use the ``with`` statement, any attribute added to the SPA network
    will be accessible for SPA connections.

    If you chose to create a subclass, any `~.spa.module.Module` object that
    is assigned to a member variable will automatically be accessible by the
    SPA connection system.

    As an example, the following code will build three modules
    (two buffers and a memory) that can be referred to as ``a``, ``b``,
    and ``c``, respectively.

    First, the example with a ``with`` statement::

        example = spa.Spa()

        with example:
            example.a = spa.Buffer(dimensions=8)
            example.b = spa.Buffer(dimensions=16)
            example.c = spa.Memory(dimensions=8)

    Now, the example with a subclass::

        class Example(spa.SPA):
            def __init__(self):
                with self:
                    self.a = spa.Buffer(dimensions=8)
                    self.b = spa.Buffer(dimensions=16)
                    self.c = spa.Memory(dimensions=8)

    These names can be used by special modules that are aware of these
    names. As an example, the `.Cortical` module allows you to form connections
    between these modules in ways that are aware of semantic pointers::

        with example:
            example.a = spa.Buffer(dimensions=8)
            example.b = spa.Buffer(dimensions=16)
            example.c = spa.Memory(dimensions=8)
            example.cortical = spa.Cortical(spa.Actions(
                    'b=a*CAT', 'c=b*~CAT'))

    For complex cognitive control, the key modules are the `.spa.BasalGanglia`
    and the `.spa.Thalamus`. Together, these allow us to define complex actions
    using the `.spa.Action` syntax::

        class SequenceExample(spa.SPA):
            def __init__(self):
                self.state = spa.Memory(dimensions=32)

                actions = spa.Actions('dot(state, A) --> state=B',
                                      'dot(state, B) --> state=C',
                                      'dot(state, C) --> state=D',
                                      'dot(state, D) --> state=E',
                                      'dot(state, E) --> state=A')

                self.bg = spa.BasalGanglia(actions=actions)
                self.thal = spa.Thalamus(self.bg)
    """

    def __init__(self, label=None, seed=None, add_to_container=None,
                 vocabs=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = None
        vocabs = [] if vocabs is None else vocabs
        super(SPA, self).__init__(
            label, seed, add_to_container, VocabularyMap(vocabs, rng))

    def get_default_vocab(self, dimensions):
        """Return a Vocabulary with the desired dimensions.

        This will create a new default Vocabulary if one doesn't exist.
        """
        return self.vocabs.get_or_create(dimensions)
