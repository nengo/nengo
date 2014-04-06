import warnings

import numpy as np

from nengo.spa import pointer
from nengo.utils.compat import is_iterable, is_number, is_integer, range


class Vocabulary(object):
    """A collection of semantic pointers, each with their own text label.

    The Vocabulary can also act as a dictionary, with keys as the names
    of the semantic pointers and values as the SemanticPointer objects
    themselves.  If it is asked for a pointer that does not exist, one will
    be automatically created.

    Parameters
    -----------
    dimensions : int
        Number of dimensions for each semantic pointer
    randomize : bool, optional
        Whether to randomly generate pointers.  If False, the semantic
        pointers will be [1,0,0,...], [0,1,0,...], [0,0,1,...] and so on.
    unitary : bool or list of strings, optional
        If True, all generated pointers to be unitary.  If a list of
        names, any pointer whose name is on the list will be forced to be
        unitary when created.
    max_similarity : float, optional
        When randomly generating pointers, ensure that the cosine of the
        angle between the new pointer and all existing pointers is less
        than this amount.  If the system is unable to find such a pointer
        after 100 tries, a warning message is printed.
    include_pairs : bool, optional
        Whether to keep track of all pairs of pointers as well.  This
        is helpful for determining if a vector is similar to A*B (in
        addition to being similar to A or B), but exponentially increases
        the processing time.
    rng : numpy.random.RandomState, optional
        The random number generator to use to create new vectors

    Attributes
    ----------
    keys : list of strings
        The names of all known semantic pointers ('A', 'B', 'C')
    key_pairs : list of strings
        The names of all pairs of semantic pointers ('A*B', 'A*C', 'B*C')
    vectors : numpy.array
        All of the semantic pointer values in a matrix, in the same order
        as in keys
    vector_pairs : numpy.array
        The values for each pair of semantic pointers, convolved together,
        in the same order as in key_pairs
    include_pairs : bool
        Whether to keep track of all pairs of pointers as well.  This
        is helpful for determining if a vector is similar to A*B (in
        addition to being similar to A or B), but exponentially increases
        the processing time.
    identity : SemanticPointer
        The identity vector for this dimensionality [1, 0, 0, 0, ...]

    """

    def __init__(self, dimensions, randomize=True, unitary=False,
                 max_similarity=0.1, include_pairs=False, rng=None):

        if not is_integer(dimensions):
            raise TypeError('dimensions must be an integer')
        if dimensions < 1:
            raise ValueError('dimensions must be positive')
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

    def create_pointer(self, attempts=100, unitary=False):
        """Create a new semantic pointer.

        This will take into account the randomize and max_similarity
        parameters from self.
        """
        if self.randomize:
            count = 0
            p = pointer.SemanticPointer(self.dimensions, rng=self.rng)
            if self.vectors.shape[0] > 0:
                while count < 100:
                    similarity = np.dot(self.vectors, p.v)
                    if max(similarity) < self.max_similarity:
                        break
                    p = pointer.SemanticPointer(self.dimensions, rng=self.rng)
                    count += 1
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
                raise IndexError('Tried to make more semantic pointers than' +
                                 ' dimensions with non-randomized Vocabulary')
            p = pointer.SemanticPointer(np.eye(self.dimensions)[index])
        return p

    def __getitem__(self, key):
        """Return the semantic pointer with the requested name.

        If one does not exist, automatically create one.  The key must be
        a valid semantic pointer name, which is any Python identifier starting
        with a capital letter.
        """
        if not key[0].isupper():
            raise KeyError('Semantic pointers must begin with a capital')
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

        The pointer value can be a SemanticPointer or a vector.
        """
        if not key[0].isupper():
            raise KeyError('Semantic pointers must begin with a capital')
        if not isinstance(p, pointer.SemanticPointer):
            p = pointer.SemanticPointer(p)

        if key in self.pointers:
            raise KeyError("The semantic pointer '%s' already exists" % key)

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
        """Adjusts whether key pairs are kept track of by the Vocabulary.

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

        This uses the Python eval() function, so any Python operators that
        have been defined for SemanticPointers are valid (+, -, *, ~, ()).
        Any terms do not exist in the vocabulary will be automatically
        generated.  Valid semantic pointer terms must start with a capital
        letter.

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
            raise KeyError('Semantic pointers must start with a capital')

        if is_number(value):
            value = value * self.identity
        if not isinstance(value, pointer.SemanticPointer):
            raise TypeError('The result of "%s" was not a SemanticPointer' %
                            text)
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
        purposes.  To do this, compute the dot product between the vector
        and all the terms in the vocabulary.  The top few vectors are
        chosen for inclusion in the text.  It will try to only return
        terms with a match above the threshold, but will always return
        at least minimum_count and at most maximum_count terms.  Terms
        are sorted from most to least similar.

        Parameters
        ----------
        v : SemanticPointer or array
            The vector to convert into text
        minimum_count : int, optional
            Always return at least this many terms in the text
        maximum_count : int, optional
            Never return more than this many terms in the text
        threshold : float, optional
            How small a similarity for a term to be ignored
        join : string, optional
            The text separator to use between terms
        terms : list of strings, optional
            Only consider terms in this list
        normalize : bool, optional
            Whether to normalize the vector before computing similarity

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

        Input parameter can either be a SemanticPointer or a vector.
        """
        if isinstance(v, pointer.SemanticPointer):
            v = v.v
        return np.dot(self.vectors, v)

    def dot_pairs(self, v):
        """Returns the dot product with all pairs of terms in the Vocabulary.

        Input parameter can either be a SemanticPointer or a vector.
        """
        if not self.include_pairs:
            raise Exception('include_pairs must be True to call dot_pairs')
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
            The other vocabulary to translate into
        keys : list of strings
            If None, any term that exists in just one of the Vocabularies
            will be created in the other Vocabulary and included.  Otherwise,
            the transformation will only consider terms in this list.  Any
            terms in this list that do not exist in the Vocabularies will
            be created.
        """
        if keys is None:
            keys = list(self.keys)
            for k in other.keys:
                if k not in keys:
                    keys.append(k)

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
        vector than the value given by compare.  To use this, compare
        your noisy vector with the ideal vector, pass that value in as
        the similarity parameter, and set vocab_size to be the number of
        competing vectors.

        The steps parameter sets the accuracy of the approximate integral
        needed to compute this.

        The basic principle used here is that the probability of two random
        vectors in a D-dimensional space being a given angle apart is
        proportional to sin(angle)**(D-2).  So we integrate this value
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
