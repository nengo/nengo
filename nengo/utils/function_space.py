from __future__ import absolute_import

import nengo
import numpy as np
import nengo.utils.numpy as npext


class Function(nengo.dists.Distribution):
    """ A distribution that generates samples from a given function
    space using the distributions specified for each parameter. """

    def __init__(self, function, superimpose=1, **distributions):
        self.function = function
        self.distributions = distributions
        self.superimpose = superimpose
        super(Function, self).__init__()

    def sample(self, n, d=None, rng=np.random):
        # TODO: an error occurs if there are fewer samples than dimensions
        n_samples = n * self.superimpose

        kwargs = {}
        for k, v in self.distributions.items():
            if isinstance(v, nengo.dists.Distribution):
                kwargs[k] = v.sample(n_samples, rng=rng)
            else:
                kwargs[k] = np.tile(v, n_samples)
        values = []
        index = 0
        for i in range(n):
            total = []
            for j in range(self.superimpose):
                args = {k: v[index] for k, v in kwargs.items()}
                total.append(self.function(**args))
                index += 1
            values.append(np.sum(total, axis=0))
        return np.vstack(values)


class FunctionSpaceDistribution(nengo.dists.Distribution):
    """ Projects data samples generated in the given function
    space into samples of weights over the basis function."""

    def __init__(self, function_space, data):
        self.fs = function_space
        self.data = data
        super(FunctionSpaceDistribution, self).__init__()

    def sample(self, n, d=None, rng=np.random):
        if isinstance(self.data, nengo.dists.Distribution):
            data = self.data.sample(n, rng=rng)
        else:
            data = self.data
        return np.dot(data, self.fs.basis)


class Combined(nengo.dists.Distribution):
    """ A distribution that allows both FunctionSpaceDistributions
    to be used for some dimensions and regular Nengo distributions
    to be used for others."""

    def __init__(self, distributions, dimensions, weighting=None,
                 weights=None, normalize_weights=True):
        if weights is None:
            weights = np.ones(len(distributions))
        if normalize_weights:
            weights = weights / np.linalg.norm(weights)

        self.weighting = weighting
        self.weights = weights
        self.distributions = distributions
        self.dimensions = dimensions
        self.n_dimensions = sum(dimensions)
        super(Combined, self).__init__()

    def sample(self, n, d=None, rng=np.random):
        assert d == self.n_dimensions

        if self.weighting == 'dimensions':
            # sample in d-dimensional space, but project down to
            # len(dist)-dimensional space for the weights to bias
            # the projection so that each distribution is projected
            # from a proportional number of dimensions
            samples = rng.randn(n, d)
            samples /= npext.norm(samples, axis=1, keepdims=True)
            weights = np.zeros((n, len(distributions)))
            start = 0
            for i, end in enumerate(np.cumsum(self.dimensions)):
                weights[:,i] = npext.norm(samples[:,start:end], axis=1, keepdims=False)
                start = end
        elif self.weighting == 'distributions':
            samples = rng.randn(n, len(self.distributions))
            samples /= npext.norm(samples, axis=1, keepdims=True)
            weights = samples
        else:
            # use the original weighting method if nothing is specified
            data = [dist.sample(n, d=self.dimensions[i], rng=rng) * self.weights[i]
                    for i, dist in enumerate(self.distributions)]

            return np.hstack(data)

        data = [dist.sample(n, d=self.dimensions[i], rng=rng) * weights[:,i].reshape((n,1))
                for i, dist in enumerate(self.distributions)]

        return np.hstack(data)


class FunctionSpace(object):

    def __init__(self, space, n_basis, n_samples=1000, seed=None):
        self.rng = np.random.RandomState(seed=seed)
        self.space = space
        self.n_basis = n_basis
        self.n_samples = n_samples

        # these are all computed by self.compute_basis(),
        # and must be specified if space=None
        self._basis = None
        self._scale = None
        self._S = None

    @property
    def basis(self):
        if self._basis is None:
            self.compute_basis()
        return self._basis

    @property
    def scale(self):
        if self._scale is None:
            self.compute_basis()
        return self._scale

    @property
    def S(self):
        if self._S is None:
            self.compute_basis()
        return self._S

    def compute_basis(self):
        if isinstance(self.space, nengo.dists.Distribution):
            data = self.space.sample(self.n_samples, rng=self.rng)
        else:
            data = self.space

        U, S, V = np.linalg.svd(data)

        proj = np.dot(data, V[:self.n_basis].T)
        self._scale = np.mean(np.linalg.norm(proj, axis=1))**2
        # self._scale = (np.mean(S[:self.n_basis])) ** 2
        # self._scale = (S[0] / self.n_basis) ** 2
        # self._scale = np.linalg.norm(S[:self.n_basis])**2
        # TODO: which of those scalings works better?
        self._basis = V[:self.n_basis].T / np.sqrt(self.scale)
        self._S = S

    def project(self, data):
        if isinstance(data, nengo.dists.Distribution):
            return FunctionSpaceDistribution(self, data)
        return np.dot(data, self.basis)

    # NOTE: might be useful to be able to have parameter to downsample
    # right here rather than only being able to reconstruct using n_samples
    def reconstruct(self, x):
        """Decode the function from the subspace back to points"""
        return np.dot(x, self.basis.T) * self.scale

    # TODO: is function param supposed to be used here?
    def make_plot_node(self, domain, lines=1, n_pts=20, function=None,
                       max_x=None, min_x=None, max_y=1, min_y=-1):
        """Generate a Node with a custom HTML GUI plot. The node takes in a set
        of weights and generates a line plot of the represented function through
        weighted summation of the basis functions"""
        indices = None
        if len(domain) > n_pts:
            indices = np.linspace(0, len(domain) - 1, n_pts).astype(int)
            domain = domain[indices]
        elif n_pts > len(domain):
            n_pts = len(domain)

        basis = self.basis[indices] if indices is not None else self.basis
        max_x = max_x if max_x is not None else np.max(domain)
        min_x = min_x if min_x is not None else np.min(domain)

        svg_x = (domain - min_x) * 100 / (max_x - min_x)

        colors = ["#1c73b3", "#039f74", "#d65e00",
                  "#cd79a7", "#f0e542", "#56b4ea"]

        def plot_func(t, x):
            paths = []
            for i in range(lines):
                # weighted summation to calculate the represented function
                value = x[i*self.n_basis:(i+1)*self.n_basis]
                data = np.dot(value, basis.T) * self.scale
                # scale the function by min_y and max_y, * 100 for svg plot
                data = (-data - min_y) * 100 / (max_y - min_y)

                # turn the data into a string for svg plotting
                path = []
                for j in range(len(data)):
                    path.append('%1.0f %1.0f' % (svg_x[j], data[j]))
                paths.append('<path d="M%s" fill="none" stroke="%s"/>' %
                             ('L'.join(path), colors[i % len(colors)]))

            plot_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">%s</svg>
            ''' % (''.join(paths))
        plot_func._nengo_html_ = ''
        return nengo.Node(plot_func, size_in=self.n_basis * lines, size_out=0)

    def make_2Dplot_node(self, domain, n_pts=20):
        """Generate a Node with a custom GUI plot"""
        basis = self.basis
        indices = None
        if len(domain) > n_pts:
            indices = np.linspace(0, len(domain) - 1, n_pts).astype(int)
            old_num = len(domain)
            domain = domain[indices]

            # TODO: do this more elegantly
            basis = basis.reshape((old_num, old_num, self.n_basis))
            basis = basis[indices][:, indices]
            basis = basis.reshape((-1, self.n_basis))

        elif n_pts > len(domain):
            n_pts = len(domain)

        def plot_func(t, x):
            values = np.dot(basis, x) * self.scale

            # put on a scale of 0 - 255
            values *= 255.0
            values = values.reshape((n_pts, n_pts))
            # flip so up isn't left rightside down
            values = np.flipud(values)

            # generate png heat map based off values
            import base64
            from PIL import Image
            import cStringIO
            png = Image.fromarray(values).convert('RGB')
            buffer = cStringIO.StringIO()
            png.save(buffer, format="JPEG")
            img_str = base64.b64encode(buffer.getvalue())

            plot_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
            <image width="100%%" height="100%%" xlink:href="data:image/png;base64,%s">
            </svg>''' % (''.join(img_str))

        plot_func._nengo_html_ = ''
        return nengo.Node(plot_func, size_in=self.n_basis, size_out=0)

    def make_input(self, input_vals):
        """Generate a Node that can control function parameters."""
        function = self.space.function
        n_params = len(self.space.distributions)

        def stim_project(t, x):
            return self.project(function(*x))

        net = nengo.Network('function input')

        with net:
            # input control parameters to function
            net.stimulus = nengo.Node(input_vals, size_out=n_params)
            # stimulus projects into weights over generated basis functions
            net.project = nengo.Node(stim_project, size_in=n_params,
                                     size_out=self.n_basis)
            nengo.Connection(net.stimulus, net.project)
            net.output = nengo.Node(size_in=self.n_basis,
                                    size_out=self.n_basis)
            nengo.Connection(net.project, net.output, synapse=None)

        return net
