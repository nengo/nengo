from collections import Mapping
import re
import warnings

import numpy as np

import nengo
from nengo.exceptions import NengoWarning, SpaParseError, ValidationError
from nengo.spa import pointer
from nengo.spa.pointer import Identity
from nengo.utils.compat import is_number, is_integer, range


valid_sp_regex = re.compile('[A-Z][_a-zA-Z0-9]*')


class Vocabulary(object):
    """A collection of semantic pointers, each with their own text label.

    The Vocabulary can also act as a dictionary, with keys as the names
    of the semantic pointers and values as the `.SemanticPointer` objects
    themselves. If it is asked for a pointer that does not exist, one will
    be automatically created.

    Parameters
    -----------
    dimensions : int
        Number of dimensions for each semantic pointer.
    strict : bool, optional (Default: True)
        TODO
    max_similarity : float, optional (Default: 0.1)
        When randomly generating pointers, ensure that the cosine of the
        angle between the new pointer and all existing pointers is less
        than this amount. If the system is unable to find such a pointer
        after 100 tries, a warning message is printed.
    rng : `numpy.random.RandomState`, optional (Default: None)
        The random number generator to use to create new vectors.

    Attributes
    ----------
    keys : list of strings
        The names of all known semantic pointers (e.g., ``['A', 'B', 'C']``).
    vectors : ndarray
        All of the semantic pointer values in a matrix, in the same order
        as in ``keys``.
    """

    def __init__(self, dimensions, strict=True, max_similarity=0.1, rng=None):

        if not is_integer(dimensions) or dimensions < 1:
            raise ValidationError("dimensions must be a positive integer",
                                  attr='dimensions', obj=self)
        self.dimensions = dimensions
        self.strict = strict
        self.max_similarity = max_similarity
        self.pointers = {}
        self.keys = []
        self._vectors = np.zeros((0, dimensions), dtype=float)
        self.rng = rng
        self.parent = None

    @property
    def vectors(self):
        v = self._vectors.view()
        v.setflags(write=False)
        return v

    def __str__(self):
        return '{}-dimensional vocab at 0x{:x}'.format(
            self.dimensions, id(self))

    def create_pointer(self, attempts=100, transform=None):
        """Create a new semantic pointer.

        This will take into account the max_similarity
        parameter from self. If a pointer satisfying max_similarity
        is not generated after the specified number of attempts, the
        candidate pointer with lowest maximum cosine with all existing
        pointers is returned.
        """
        if len(self) == 0:
            best_p = pointer.SemanticPointer(self.dimensions, rng=self.rng)
        else:
            best_p = None
            best_sim = np.inf
            for _ in range(attempts):
                p = pointer.SemanticPointer(self.dimensions, rng=self.rng)
                if transform is not None:
                    p = eval('p.' + transform, {'p': p}, self)
                p_sim = np.max(np.dot(self._vectors, p.v))
                if p_sim < best_sim:
                    best_p = p
                    best_sim = p_sim
                    if p_sim < self.max_similarity:
                        break
            else:
                warnings.warn(
                    'Could not create a semantic pointer with '
                    'max_similarity=%1.2f (D=%d, M=%d)'
                    % (self.max_similarity, self.dimensions,
                       len(self.pointers)))
        return best_p

    def __contains__(self, key):
        return key in self.keys

    def __len__(self):
        return len(self._vectors)

    def __getitem__(self, key):
        """Return the semantic pointer with the requested name."""
        if not self.strict and key not in self:
            self.add(key, self.create_pointer())
        return self.pointers[key]

    def add(self, key, p):
        """Add a new semantic pointer to the vocabulary.

        The pointer value can be a `.SemanticPointer` or a vector.
        """
        if not valid_sp_regex.match(key):
            raise SpaParseError(
                "Invalid Semantic Pointer name. Valid names are valid "
                "Python 2 identifiers beginning with a capital letter.")
        if not isinstance(p, pointer.SemanticPointer):
            p = pointer.SemanticPointer(p)

        if key in self.pointers:
            raise ValidationError("The semantic pointer %r already exists"
                                  % key, attr='pointers', obj=self)

        self.pointers[key] = p
        self.keys.append(key)
        self._vectors = np.vstack([self._vectors, p.v])

    def populate(self, pointers):
        for p_expr in pointers.split(','):
            assign_split = p_expr.split('=', 1)
            modifier_split = p_expr.split('.', 1)
            if len(assign_split) > 1:
                name, value_expr = assign_split
                value = eval(value_expr.strip(), {}, self)
            elif len(modifier_split) > 1:
                name = modifier_split[0]
                value = self.create_pointer(transform=modifier_split[1])
            else:
                name = p_expr
                value = self.create_pointer()
            self.add(name.strip(), value)

    def parse(self, text):
        """Evaluate a text string and return the corresponding SemanticPointer.

        This uses the Python ``eval()`` function, so any Python operators that
        have been defined for SemanticPointers are valid (``+``, ``-``, ``*``,
        ``~``, ``()``). Valid semantic pointer terms must start
        with a capital letter.

        If the expression returns a scalar (int or float), a scaled version
        of the identity SemanticPointer will be returned.
        """

        # The following line does everything.  Note that self is being
        # passed in as the locals dictionary, and thanks to the __getitem__
        # implementation, this will automatically create new semantic
        # pointers as needed.
        try:
            value = eval(text, {}, self)
        except NameError:
            raise SpaParseError(
                "Semantic pointers must start with a capital letter.")

        if is_number(value):
            value *= Identity(self.dimensions)
        elif not isinstance(value, pointer.SemanticPointer):
            raise SpaParseError(
                "The result of parsing '%s' is not a SemanticPointer." % text)
        return value

    def parse_n(self, *texts):
        return [self.parse(t) for t in texts]

    def dot(self, v):
        """Returns the dot product with all terms in the Vocabulary.

        Input parameter can either be a `.SemanticPointer` or a vector.
        """
        if isinstance(v, pointer.SemanticPointer):
            v = v.v
        return np.dot(self._vectors, v)

    def transform_to(self, other, populate=None, keys=None):
        """Create a linear transform from one Vocabulary to another.

        This is simply the sum of the outer products of the corresponding
        terms in each Vocabulary.

        Parameters
        ----------
        other : Vocabulary
            The other vocabulary to translate into.
        keys : list, optional (Default: None)
            If None, any term that exists in just one of the Vocabularies
            will be created in the other Vocabulary and included. Otherwise,
            the transformation will only consider terms in this list. Any
            terms in this list that do not exist in the Vocabularies will
            be created.
        """
        # If the parent vocabs of self and other are the same, then no
        # transform is needed between the two vocabularies, so return an
        # identity matrix.
        my_parent = self if self.parent is None else self.parent
        other_parent = other if other.parent is None else other.parent

        if my_parent is other_parent:
            return np.eye(self.dimensions)
        else:
            if keys is None:
                keys = self.keys

            t = np.zeros((other.dimensions, self.dimensions), dtype=float)
            for k in keys:
                if k not in other:
                    if populate is None:
                        warnings.warn(NengoWarning(
                            "The transform_to source vocabulary has keys not "
                            "existent in the target vocabulary. These will "
                            "be ignored. Use the `populate=False` keyword "
                            "argument to silence this warning or "
                            "`populate=True` to automatically add missing "
                            "keys to the target vocabulary."))
                    elif populate:
                        other.populate(k)
                    else:
                        continue
                a = self[k].v
                b = other[k].v
                t += np.outer(b, a)
            return t

    def create_subset(self, keys):
        """Returns the subset of this vocabulary.

        Creates and returns a subset of the current vocabulary that contains
        all the semantic pointers found in keys.

        Parameters
        ----------
        keys : list
            List of semantic pointer names to be copied over to the
            new vocabulary.
        """
        # Make new Vocabulary object
        subset = Vocabulary(self.dimensions, self.strict, self.max_similarity,
                            self.rng)

        # Copy over the new keys
        for key in keys:
            subset.add(key, self.pointers[key])

        # Assign the parent
        if self.parent is not None:
            subset.parent = self.parent
        else:
            subset.parent = self

        return subset


class VocabularyMap(Mapping):
    """Maps dimensionalities to corresponding vocabularies."""
    def __init__(self, vocabs=None, rng=None):
        if vocabs is None:
            vocabs = []
        self.rng = rng

        self._vocabs = {}
        try:
            for vo in vocabs:
                self.add(vo)
        except (AttributeError, TypeError):
            raise ValueError(
                "The `vocabs` argument requires a list of Vocabulary "
                "instances or `None`.")

    def add(self, vocab):
        if vocab.dimensions in self._vocabs:
            warnings.warn("Duplicate vocabularies with dimension %d. "
                          "Using the last entry in the vocab list with "
                          "that dimensionality." % (vocab.dimensions))
        self._vocabs[vocab.dimensions] = vocab

    def __delitem__(self, dimensions):
        del self._vocabs[dimensions]

    def discard(self, vocab):
        if isinstance(vocab, int):
            del self._vocabs[vocab]
        elif self._vocabs.get(vocab.dimensions, None) is vocab:
            del self._vocabs[vocab.dimensions]

    def __getitem__(self, dimensions):
        return self._vocabs[dimensions]

    def get_or_create(self, dimensions):
        if dimensions not in self._vocabs:
            self._vocabs[dimensions] = Vocabulary(
                dimensions, strict=False, rng=self.rng)
        return self._vocabs[dimensions]

    def __iter__(self):
        return iter(self._vocabs)

    def __len__(self):
        return len(self._vocabs)

    def __contains__(self, vocab):
        if isinstance(vocab, int):
            return vocab in self._vocabs
        else:
            return (vocab.dimensions in self._vocabs and
                    self._vocabs[vocab.dimensions] is vocab)


class VocabularyMapParam(nengo.params.Parameter):
    """Can be a mapping from dimensions to vocabularies."""

    def validate(self, instance, vocab_set):
        super(VocabularyMapParam, self).validate(instance, vocab_set)

        if vocab_set is not None and not isinstance(vocab_set, VocabularyMap):
            try:
                VocabularyMap(vocab_set)
            except ValueError:
                raise ValidationError(
                    "Must be of type 'VocabularyMap' or compatible "
                    "(got type %r)."
                    % type(vocab_set).__name__, attr=self.name, obj=instance)

        return vocab_set

    def __set__(self, instance, value):
        if not isinstance(value, VocabularyMap):
            value = VocabularyMap(value)
        super(VocabularyMapParam, self).__set__(instance, value)


class VocabularyOrDimParam(nengo.params.Parameter):
    """Can be a vocabulary or integer denoting a dimensionality."""

    def validate(self, instance, value):
        super(VocabularyOrDimParam, self).validate(instance, value)

        if value is not None:
            if is_integer(value):
                if value < 1:
                    raise ValidationError(
                        "Vocabulary dimensionality must be at least 1.",
                        attr=self.name, obj=instance)
            elif not isinstance(value, Vocabulary):
                raise ValidationError(
                    "Must be of type 'Vocabulary' or an integer (got type %r)."
                    % type(value).__name__, attr=self.name, obj=instance)

    def __set__(self, instance, value):
        if is_integer(value):
            value = instance.vocabs.get_or_create(value)
        super(VocabularyOrDimParam, self).__set__(instance, value)
