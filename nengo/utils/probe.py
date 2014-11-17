import nengo
from .compat import iteritems


def probe_all(net, recursive=False, probe_options=None,  # noqa: C901
              **probe_args):

    """A helper function to make probing easier.

    Parameters
    ----------
    net : nengo.Network
    recursive : bool, optional
        probe subnetworks recursively, False by default.
    probe_options: dict, optional
        A dict of the form {nengo_object_class: [attributes_to_probe]}.
        If not specified, every probeable attribute of every object
        will be probed.

    Returns
    -------
    A dictionary that maps objects and their attributes to their probes.

    Examples
    --------

    Probe the decoded output and spikes in all ensembles in a network and
    its subnetworks.

    ::

        model = nengo.Network(label='_test_probing')
        with model:
            ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node1 = nengo.Node(output=[0])
            conn = nengo.Connection(node1, ens1)
            subnet = nengo.Network(label='subnet')
            with subnet:
                ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
                node2 = nengo.Node(output=[0])

        probe_options = {nengo.Ensemble: ['decoded_output', 'spikes']}
        object_probe_dict = probe(model, recursive=True,
                                  probe_options=probe_options)

    """

    probes = {}

    def probe_helper(net, recursive, probe_options):
        with net:
            for obj_type, obj_list in iteritems(net.objects):

                # recursively probe subnetworks if required
                if obj_type is nengo.Network and recursive:
                    for subnet in obj_list:
                        probe_helper(subnet, recursive=recursive,
                                     probe_options=probe_options)

                # probe all probeable objects
                elif probe_options is None:
                    for obj in obj_list:
                        if hasattr(obj, 'probeable') and len(
                                obj.probeable) > 0:
                            probes[obj] = {}
                            for probeable in obj.probeable:
                                probes[obj][probeable] = nengo.Probe(
                                    obj, probeable, **probe_args)

                # probe specified objects only
                elif obj_type in probe_options:
                    for obj in obj_list:
                        if not (hasattr(obj, 'probeable')
                                and len(obj.probeable) > 0):
                            raise ValueError("'%s' is not probeable" % obj)
                        probes[obj] = {}
                        for attr in probe_options[obj_type]:
                            if attr not in obj.probeable:
                                raise ValueError(
                                    "'%s' is not probeable for '%s'" %
                                    (obj, attr))
                            probes[obj][
                                attr] = nengo.Probe(obj, attr, **probe_args)

    probe_helper(net, recursive, probe_options)
    return probes
