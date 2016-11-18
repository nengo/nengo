import warnings

import numpy as np

from nengo.exceptions import ReadonlyError, SpaParseError, ValidationError
from nengo.params import Parameter
from nengo.spa import pointer
from nengo.utils.compat import is_iterable, is_number, is_integer, range


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
    randomize : bool, optional (Default: True)
        Whether to randomly generate pointers. If False, the semantic
        pointers will be ``[1, 0, 0, ...], [0, 1, 0, ...], [0, 0, 1, ...]``
        and so on.
    unitary : bool or list, optional (Default: False)
        If True, all generated pointers will be unitary. If a list of
        strings, any pointer whose name is in the list will be forced to be
        unitary when created.
    max_similarity : float, optional (Default: 0.1)
        When randomly generating pointers, ensure that the cosine of the
        angle between the new pointer and all existing pointers is less
        than this amount. If the system is unable to find such a pointer
        after 100 tries, a warning message is printed.
    include_pairs : bool, optional (Default: False)
        Whether to keep track of all pairs of pointers as well. This
        is helpful for determining if a vector is similar to ``A*B`` (in
        addition to being similar to ``A`` or ``B``), but exponentially
        increases the processing time.
    rng : `numpy.random.RandomState`, optional (Default: None)
        The random number generator to use to create new vectors.

    Attributes
    ----------
    include_pairs : bool
        Whether to keep track of all pairs of pointers as well. This
        is helpful for determining if a vector is similar to ``A*B`` (in
        addition to being similar to ``A`` or ``B``), but exponentially
        increases the processing time.
    key_pairs : list
        The names of all pairs of semantic pointers
        (e.g., ``['A*B', 'A*C', 'B*C']``).
    keys : list of strings
        The names of all known semantic pointers (e.g., ``['A', 'B', 'C']``).
    vector_pairs : ndarray
        The values for each pair of semantic pointers, convolved together,
        in the same order as in ``key_pairs``.
    vectors : ndarray
        All of the semantic pointer values in a matrix, in the same order
        as in ``keys``.
    """

    def __init__(self, dimensions, randomize=True, unitary=False,
                 max_similarity=0.1, include_pairs=False, rng=None):

        if not is_integer(dimensions) or dimensions < 1:
            raise ValidationError("dimensions must be a positive integer",
                                  attr='dimensions', obj=self)
        self.dimensions = dimensions
        self.randomize = randomize
        self.unitary = unitary
        self.max_similarity = max_similarity
        self.pointers = {}
        self.keys = []
        self.key_pairs = None
        self.vectors = np.zeros((0, dimensions), dtype=float)
        self.vector_pairs = None
        self._include_pairs = None
        self.include_pairs = include_pairs
        self._identity = None
        self.rng = rng
        self.readonly = False
        self.parent = None

    def create_pointer(self, attempts=100, unitary=False):
        """Create a new semantic pointer.

        This will take into account the randomize and max_similarity
        parameters from self. If a pointer satisfying max_similarity
        is not generated after the specified number of attempts, the
        candidate pointer with lowest maximum cosine with all existing
        pointers is returned.
        """
        if self.randomize:
            if self.vectors.shape[0] == 0:
                p = pointer.SemanticPointer(self.dimensions, rng=self.rng)
            else:
                p_sim = np.inf
                for _ in range(attempts):
                    pp = pointer.SemanticPointer(self.dimensions, rng=self.rng)
                    pp_sim = max(np.dot(self.vectors, pp.v))
                    if pp_sim < p_sim:
                        p = pp
                        p_sim = pp_sim
                        if p_sim < self.max_similarity:
                            break
                else:
                    warnings.warn(
                        'Could not create a semantic pointer with '
                        'max_similarity=%1.2f (D=%d, M=%d)'
                        % (self.max_similarity,
                           self.dimensions,
                           len(self.pointers)))

            # Check and make vector unitary if needed
            if unitary:
                p.make_unitary()
        else:
            index = len(self.pointers)
            if index >= self.dimensions:
                raise ValidationError(
                    "Tried to make more semantic pointers than "
                    "dimensions with non-randomized Vocabulary",
                    attr='dimensions', obj=self)
            p = pointer.SemanticPointer(np.eye(self.dimensions)[index])
        return p

    def __getitem__(self, key):
        """Return the semantic pointer with the requested name.

        If one does not exist, automatically create one. The key must be
        a valid semantic pointer name, which is any Python identifier starting
        with a capital letter.
        """
        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")
        value = self.pointers.get(key, None)
        if value is None:
            if is_iterable(self.unitary):
                unitary = key in self.unitary
            else:
                unitary = self.unitary
            value = self.create_pointer(unitary=unitary)
            self.add(key, value)
        return value

    def add(self, key, p):
        """Add a new semantic pointer to the vocabulary.

        The pointer value can be a `.SemanticPointer` or a vector.
        """
        if self.readonly:
            raise ReadonlyError(attr='Vocabulary',
                                msg="Cannot add semantic pointer '%s' to "
                                    "read-only vocabulary." % key)

        if not key[0].isupper():
            raise SpaParseError(
                "Semantic pointers must begin with a capital letter.")
        if not isinstance(p, pointer.SemanticPointer):
            p = pointer.SemanticPointer(p)

        if key in self.pointers:
            raise ValidationError("The semantic pointer %r already exists"
                                  % key, attr='pointers', obj=self)

        self.pointers[key] = p
        self.keys.append(key)
        self.vectors = np.vstack([self.vectors, p.v])

        # Generate vector pairs
        if self.include_pairs and len(self.keys) > 1:
            for k in self.keys[:-1]:
                self.key_pairs.append('%s*%s' % (k, key))
                v = (self.pointers[k] * p).v
                self.vector_pairs = np.vstack([self.vector_pairs, v])

    @property
    def include_pairs(self):
        return self._include_pairs

    @include_pairs.setter
    def include_pairs(self, value):
        """Adjusts whether key pairs are kept track of by the vocabulary.

        If this is turned on, we need to compute all the pairs of terms
        already existing.
        """
        if value == self._include_pairs:
            return
        self._include_pairs = value
        if self._include_pairs:
            self.key_pairs = []
            self.vector_pairs = np.zeros((0, self.dimensions), dtype=float)
            for i in range(1, len(self.keys)):
                for k in self.keys[:i]:
                    key = self.keys[i]
                    self.key_pairs.append('%s*%s' % (k, key))
                    v = (self.pointers[k] * self.pointers[key]).v
                    self.vector_pairs = np.vstack((self.vector_pairs, v))
        else:
            self.key_pairs = None
            self.vector_pairs = None

    def parse(self, text):
        """Evaluate a text string and return the corresponding SemanticPointer.

        This uses the Python ``eval()`` function, so any Python operators that
        have been defined for SemanticPointers are valid (``+``, ``-``, ``*``,
        ``~``, ``()``). Any terms do not exist in the vocabulary will be
        automatically generated. Valid semantic pointer terms must start
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
            value = value * self.identity
        if not isinstance(value, pointer.SemanticPointer):
            raise SpaParseError(
                "The result of parsing '%s' is not a SemanticPointer" % text)
        return value

    @property
    def identity(self):
        """Return the identity vector."""
        if self._identity is None:
            v = np.zeros(self.dimensions)
            v[0] = 1
            self._identity = pointer.SemanticPointer(v)
        return self._identity

    def text(self, v, minimum_count=1, maximum_count=None,  # noqa: C901
             threshold=0.1, join=';', terms=None, normalize=False):
        """Return a human-readable text version of the provided vector.

        This is meant to give a quick text version of a vector for display
        purposes. To do this, compute the dot product between the vector
        and all the terms in the vocabulary. The top few vectors are
        chosen for inclusion in the text. It will try to only return
        terms with a match above the threshold, but will always return
        at least minimum_count and at most maximum_count terms. Terms
        are sorted from most to least similar.

        Parameters
        ----------
        v : SemanticPointer or ndarray
            The vector to convert into text.
        minimum_count : int, optional (Default: 1)
            Always return at least this many terms in the text.
        maximum_count : int, optional (Default: None)
            Never return more than this many terms in the text.
            If None, all terms will be returned.
        threshold : float, optional (Default: 0.1)
            How small a similarity for a term to be ignored.
        join : str, optional (Default: ';')
            The text separator to use between terms.
        terms : list, optional (Default: None)
            Only consider terms in this list of strings.
        normalize : bool, optional (Default: False)
            Whether to normalize the vector before computing similarity.
        """
        if isinstance(v, pointer.SemanticPointer):
            v = v.v
        else:
            v = np.array(v, dtype='float')

        if normalize:
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm

        m = np.dot(self.vectors, v)
        matches = [(mm, self.keys[i]) for i, mm in enumerate(m)]
        if self.include_pairs:
            m2 = np.dot(self.vector_pairs, v)
            matches2 = [(mm2, self.key_pairs[i]) for i, mm2 in enumerate(m2)]
            matches.extend(matches2)
        if terms is not None:
            # TODO: handle the terms parameter more efficiently, so we don't
            # compute a whole bunch of dot products and then throw them out
            matches = [mm for mm in matches if mm[1] in terms]
        matches.sort()
        matches.reverse()

        r = []
        for m in matches:
            if minimum_count is not None and len(r) < minimum_count:
                r.append(m)
            elif maximum_count is not None and len(r) == maximum_count:
                break
            elif threshold is None or m[0] > threshold:
                r.append(m)
            else:
                break

        return join.join(['%0.2f%s' % (sim, key) for (sim, key) in r])

    def dot(self, v):
        """Returns the dot product with all terms in the Vocabulary.

        Input parameter can either be a `.SemanticPointer` or a vector.
        """
        if isinstance(v, pointer.SemanticPointer):
            v = v.v
        return np.dot(self.vectors, v)

    def dot_pairs(self, v):
        """Returns the dot product with all pairs of terms in the Vocabulary.

        Input parameter can either be a `.SemanticPointer` or a vector.
        """
        if not self.include_pairs:
            raise ValidationError(
                "'include_pairs' must be True to call dot_pairs",
                attr='include_pairs', obj=self)

        if isinstance(v, pointer.SemanticPointer):
            v = v.v
        return np.dot(self.vector_pairs, v)

    def transform_to(self, other, keys=None):
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
                if self.readonly and other.readonly:
                    keys = [k for k in self.keys if k in other.keys]
                elif self.readonly:
                    keys = list(self.keys)
                elif other.readonly:
                    keys = list(other.keys)
                else:
                    keys = list(self.keys)
                    keys.extend([k for k in other.keys if k not in self.keys])

            t = np.zeros((other.dimensions, self.dimensions), dtype=float)
            for k in keys:
                a = self[k].v
                b = other[k].v
                t += np.outer(b, a)
            return t

    def prob_cleanup(self, similarity, vocab_size, steps=10000):
        """Estimate the chance of successful cleanup.

        This returns the chance that, out of vocab_size randomly chosen
        vectors, at least one of them will be closer to a particular
        vector than the value given by compare. To use this, compare
        your noisy vector with the ideal vector, pass that value in as
        the similarity parameter, and set ``vocab_size`` to be the number of
        competing vectors.

        The steps parameter sets the accuracy of the approximate integral
        needed to compute this.

        The basic principle used here is that the probability of two random
        vectors in a D-dimensional space being a given angle apart is
        proportional to ``sin(angle)**(D-2)``.  So we integrate this value
        to get a probability of one vector being farther away than the
        desired angle, and then raise that to vocab_size to get the
        probability that all of them are farther away.
        """

        # TODO: test for numerical stability.  We are taking a number
        # slightly below 1 and raising it to a large exponent, so there's
        # lots of room for rounding errors.

        angle = np.arccos(similarity)

        x = np.linspace(0, np.pi, steps)
        y = np.sin(x) ** (self.dimensions-2)

        total = np.sum(y)
        too_close = np.sum(y[:int(angle * steps / np.pi)])

        perror1 = too_close / total

        pcorrect = (1 - perror1) ** vocab_size
        return pcorrect

    def extend(self, keys, unitary=False):
        """Extends the vocabulary with additional keys.

        Creates and adds the semantic pointers listed in keys to the
        vocabulary.

        Parameters
        ----------
        keys : list
            List of semantic pointer names to be added to the vocabulary.
        unitary : bool or list, optional (Default: False)
            If True, all generated pointers will be unitary. If a list of
            strings, any pointer whose name is on the list will be forced to
            be unitary when created.
        """
        if is_iterable(unitary):
            if is_iterable(self.unitary):
                self.unitary.extend(unitary)
            else:
                self.unitary = list(unitary)
        elif unitary:
            if is_iterable(self.unitary):
                self.unitary.extend(keys)
            else:
                self.unitary = list(keys)

        for key in keys:
            if key not in self.keys:
                self[key]

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
        subset = Vocabulary(self.dimensions,
                            self.randomize,
                            self.unitary,
                            self.max_similarity,
                            self.include_pairs,
                            self.rng)

        # Copy over the new keys
        for key in keys:
            subset.add(key, self.pointers[key])

        # Assign the parent
        if self.parent is not None:
            subset.parent = self.parent
        else:
            subset.parent = self

        # Make the subset read only
        subset.readonly = True

        return subset


class VocabularyParam(Parameter):
    """Can be a Vocabulary."""

    def coerce(self, instance, vocab):
        self.check_type(instance, vocab, Vocabulary)
        return super(VocabularyParam, self).coerce(instance, vocab)
