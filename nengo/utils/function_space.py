import nengo
import numpy as np


class FunctionSpace(object):

    def __init__(self, function, n_samples, n_basis, seed=None, pts=None,
                 **distributions):
        self.rng = np.random.RandomState(seed=seed)
        self.function = function
        self.pts = pts
        self.size_output = len(self.call(**distributions))
        self.n_params = len(distributions)

        self.n_basis = n_basis
        data = self.sample(n_samples=n_samples, **distributions)
        self.U, self.S, self.V = np.linalg.svd(data)
        #self.scale = (np.mean(self.S[:self.n_basis])) ** 2
        self.scale = (self.S[0] / self.n_basis) ** 2
        #TODO: which of those scalings works better?
        self.basis = self.V[:self.n_basis].T / np.sqrt(self.scale)

    def call(self, function=None, **distributions):
        """Return one sampled instance of the function"""
        if function is None:
            function = self.function
        kwargs = {}
        for k, v in distributions.items():
            if isinstance(v, nengo.dists.Distribution):
                kwargs[k] = v.sample(1, rng=self.rng)[0]
            else:
                kwargs[k] = v
        return function(self.pts, **kwargs)

    def sample(self, function=None, n_samples=1, superimpose=1,
               **distributions):
        """Sample the function n_samples times, with possible superposition"""
        result = np.zeros((n_samples, self.size_output))
        for i in range(n_samples * superimpose):
            result[i / superimpose] += self.call(function, **distributions)
            #TODO: check if it's faster to sample distributions outside loop
        return result

    def project(self, pts):
        """Encode the function from points into the subspace"""
        return np.dot(pts, self.basis)

    def reconstruct(self, x):
        """Decode the function from the subspace back to points"""
        return np.dot(x, self.basis.T) * self.scale

    def set_encoders(self, ens, function=None, **distributions):
        """Set ensemble encoders by sampling the function"""
        ens.encoders = self.project(self.sample(function=function,
                                                n_samples=ens.n_neurons,
                                                **distributions))

    def set_eval_points(self, ens, function=None, n_eval_points=100,
                        **distributions):
        """Set ensemble eval points by sampling the function"""
        ens.eval_points = self.project(self.sample(function=function,
                                                   n_samples=n_eval_points,
                                                   **distributions))

    def make_plot_node(self, lines=1, n_pts=20, function=None,
                       max_x=None, min_x=None, max_y=1, min_y=-1):
        """Generate a Node with a custom GUI plot"""
        basis = self.basis
        pts = self.pts
        if len(pts) > n_pts:
            indices = np.linspace(0, len(pts) - 1, n_pts).astype(int)
            basis = basis[indices]
            pts = pts[indices]

        if max_x is None:
            max_x = np.max(pts)
        if min_x is None:
            min_x = np.min(pts)

        svg_x = (pts - min_x) * 100 / (max_x - min_x)

        colors = ["#1c73b3", "#039f74", "#d65e00",
                  "#cd79a7", "#f0e542", "#56b4ea"]

        def plot_func(t, x):
            paths = []
            for i in range(lines):
                value = x[i*self.n_basis:(i+1)*self.n_basis]
                data = np.dot(value, basis.T) * self.scale

                svg_y = (-data - min_y) * 100 / (max_y - min_y)

                path = []
                for j in range(len(data)):
                    path.append('%1.0f %1.0f' % (svg_x[j], svg_y[j]))
                paths.append('<path d="M%s" fill="none" stroke="%s"/>' %
                             ('L'.join(path), colors[i % len(colors)]))

            plot_func._nengo_html_ = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">%s</svg>
            ''' % (''.join(paths))
        return nengo.Node(plot_func, size_in=self.n_basis * lines)

    def make_stimulus_node(self):
        """Generate a Node that can control function parameters."""
        def stimulus(t, x):
            return fs.project(self.function(fs.pts, *x[1:]) * x[0])
        return nengo.Node(stimulus, size_in=self.n_params + 1)
