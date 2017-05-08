import nengo
from nengo.utils.function_space import Func_Space


class FS_Ensemble(nengo.Network):
    """An ensemble that represents a functions space."""

    def __init__(self, FS, label=None, seed=None, add_to_container=None,
                 **ens_kwargs):

        if not isinstance(FS, Func_Space):
            raise ValueError("FS argument must be an object of type"
                             " ``Function_Space``")

        super(FS_Ensemble, self).__init__(label, seed, add_to_container)

        self.FS = FS
        n_points = FS.n_points
        n_functions = FS.n_functions
        n_basis = FS.n_basis

        # define these to ignore time argument
        def output1(t, x):
            return FS.project(x)

        def output2(t, x):
            return FS.reconstruct(x)

        with self:
            self.input = nengo.Node(size_in=n_points, output=output1,
                                    label='FS_input')
            self.ens = nengo.Ensemble(n_neurons=n_functions,
                                      dimensions=n_basis,
                                      encoders=FS.encoders(), **ens_kwargs)
            nengo.Connection(self.input, self.ens, synapse=None)
            self.output = nengo.Node(size_in=n_basis, size_out=n_points,
                                     output=output2, label='FS_output')
            nengo.Connection(self.ens, self.output, synapse=None)


def FS_Hierarchy(FS_networks, net=None):
    if net is None:
        net = nengo.Network(label="Function Space Hierarchy")

    with net:
        for ii in range(len(FS_networks) - 1):
            def connect_func(x):
                original = FS_networks[ii].FS.reconstruct(x)
                return FS_networks[ii + 1].FS.project(original)
        nengo.Connection(FS_networks[ii].ens, FS_networks[ii + 1].ens,
                         function=connect_func)
    return net
