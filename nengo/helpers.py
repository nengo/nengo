def encoders():
    @staticmethod
    def _process_encoders(encoders, neurons, dims, n_ensembles):
        if encoders is None:
            encoders = [None for _ in xrange(n_ensembles)]
        elif len(encoders) == dims:
            if np.asarray(encoders).ndim == 1:
                encoders = [np.array(encoders) for _ in xrange(n_ensembles)]
        elif len(encoders) == neurons:
            if len(encoders[0]) != dims:
                msg = ("len(encoders[0]) should match dimensions_per_ensemble. "
                       "Currently %d, %d" % (len(encoders[0]) != dims))
                raise core.ShapeMismatch(msg)
            encoders = [np.array(encoders) for _ in xrange(n_ensembles)]
        elif len(encoders) != n_ensembles:
            msg = ("len(encoders) should match n_ensembles. "
                   "Currently %d, %d" % (len(encoders) != n_ensembles))
            raise core.ShapeMismatch(msg)
        return encoders


def transform(pre_dims, post_dims,
              weight=1.0, index_pre=None, index_post=None):
    """Helper function used to create a ``pre_dims`` by ``post_dims``
    linear transformation matrix.

    Parameters
    ----------
    pre_dims, post_dims : int
        The numbers of presynaptic and postsynaptic dimensions.
    weight : float, optional
        The weight value to use in the transform.

        All values in the transform are either 0 or ``weight``.

        **Default**: 1.0
    index_pre, index_post : iterable of int
        Determines which values are non-zero, and indicates which
        dimensions of the pre-synaptic ensemble should be routed to which
        dimensions of the post-synaptic ensemble.

    Returns
    -------
    transform : 2D matrix of floats
        A two-dimensional transform matrix performing the requested routing.

    Examples
    --------

      # Sends the first two dims of pre to the first two dims of post
      >>> gen_transform(pre_dims=2, post_dims=3,
                        index_pre=[0, 1], index_post=[0, 1])
      [[1, 0], [0, 1], [0, 0]]

    """
    t = [[0 for pre in xrange(pre_dims)] for post in xrange(post_dims)]
    if index_pre is None:
        index_pre = range(pre_dims)
    elif isinstance(index_pre, int):
        index_pre = [index_pre]

    if index_post is None:
        index_post = range(post_dims)
    elif isinstance(index_post, int):
        index_post = [index_post]

    for i in xrange(min(len(index_pre), len(index_post))):  # was max
        pre = index_pre[i]  # [i % len(index_pre)]
        post = index_post[i]  # [i % len(index_post)]
        t[post][pre] = weight
    return t


def weights(pre_neurons, post_neurons, function):
    """Helper function used to create a ``pre_neurons`` by ``post_neurons``
    connection weight matrix.

    Parameters
    ----------
    pre_neurons, post_neurons : int
        The numbers of presynaptic and postsynaptic neurons.
    function : function
        A function that generates weights.

        If it accepts no arguments, it will be called to
        generate each individual weight (useful
        to great random weights, for example).
        If it accepts two arguments, it will be given the
        ``pre`` and ``post`` index in the weight matrix.

    Returns
    -------
    weights : 2D matrix of floats
        A two-dimensional connection weight matrix.

    Examples
    --------

      >>> gen_weights(2, 2, random.random)
      [[0.6281625119511959, 0.48560016153108376], [0.9639779858394248, 0.4768136917985597]]

      >>> def product(pre, post):
      ...     return pre * post
      >>> gen_weights(3, 3, product)
      [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

    """
    argspec = inspect.getargspec(func)
    if len(argspec[0]) == 0:
        return [[func() for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]
    elif len(argspec[0]) == 2:
        return [[func(pre, post) for pre in xrange(pre_neurons)
                 for post in xrange(post_neurons)]]


# def compute_transform(dim_pre, dim_post, array_size_post, array_size_pre,
#         weight=1, index_pre=None, index_post=None, transform=None):
#     """Helper function used by :func:`nef.Network.connect()` to create
#     the `dim_post` by `dim_pre` transform matrix.

#     Values are either 0 or *weight*. *index_pre* and *index_post*
#     are used to determine which values are non-zero, and indicate
#     which dimensions of the pre-synaptic ensemble should be routed
#     to which dimensions of the post-synaptic ensemble.

#     :param int dim_pre: first dimension of transform matrix
#     :param int dim_post: second dimension of transform matrix
#     :param int array_size: size of the network arrayvv
#     :param float weight: the non-zero value to put into the matrix
#     :param index_pre: the indexes of the pre-synaptic dimensions to use
#     :type index_pre: list of integers or a single integer
#     :param index_post:
#         the indexes of the post-synaptic dimensions to use
#     :type index_post: list of integers or a single integer
#     :returns:
#         a two-dimensional transform matrix performing
#         the requested routing

#     """

#     all_pre = dim_pre * array_size_pre

#     if transform is None:
#         # create a matrix of zeros
#         transform = [[0] * all_pre for i in range(dim_post * array_size_post)]

#         # default index_pre/post lists set up *weight* value
#         # on diagonal of transform

#         # if dim_post * array_size_post != all_pre,
#         # then values wrap around when edge hit
#         if index_pre is None:
#             index_pre = range(all_pre)
#         elif isinstance(index_pre, int):
#             index_pre = [index_pre]
#         if index_post is None:
#             index_post = range(dim_post * array_size_post)
#         elif isinstance(index_post, int):
#             index_post = [index_post]

#         for i in range(max(len(index_pre), len(index_post))):
#             pre = index_pre[i % len(index_pre)]
#             post = index_post[i % len(index_post)]
#             transform[post][pre] = weight

#     transform = np.asarray(transform)

#     # reformulate to account for post.array_size_post
#     if transform.shape == (dim_post * array_size_post, all_pre):
#         rval = np.zeros((array_size_pre, dim_pre, array_size_post, dim_post))
#         for i in range(array_size_post):
#             for j in range(dim_post):
#                 rval[:, :, i, j] = transform[i * dim_post + j].reshape(
#                         array_size_pre, dim_pre)

#         transform = rval
#     else:
#         raise NotImplementedError()

#     rval = np.asarray(transform)
#     return rval


# def sample_unit_signal(dimensions, num_samples, rng):
#     """Generate sample points uniformly distributed within the sphere.

#     Returns float array of sample points: dimensions x num_samples

#     """
#     samples = rng.randn(num_samples, dimensions)

#     # normalize magnitude of sampled points to be of unit length
#     norm = np.sum(samples * samples, axis=1)
#     samples /= np.sqrt(norm)[:, None]

#     # generate magnitudes for vectors from uniform distribution
#     scale = rng.rand(num_samples, 1) ** (1.0 / dimensions)

#     # scale sample points
#     samples *= scale

#     return samples.T


# def filter_coefs(pstc, dt):
#     """
#     Use like: fcoef, tcoef = filter_coefs(pstc=pstc, dt=dt)
#         transform(tcoef, a, b)
#         filter(fcoef, b, b)
#     """
#     pstc = max(pstc, dt)
#     decay = np.exp(-dt / pstc)
#     return decay, (1.0 - decay)
