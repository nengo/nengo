from nengo.objects import Probe, Network


def probe(net, recursive=False, probe_options=None):
    """
    A helper function to make probing easier.

    Parameters:
    ----------
    net: a nengo Network
    recursive: bool, optional
                  probe subnetworks recursively, False by default.
    probe_options: dict, optional
                      A dict of the form
                      {nengo_object_class: [attributes_to_probe]}.
                      If not specified, every probeable attribute
                      of every object will be probed.

    Returns:
    --------
    A dictionary that maps objects and their attributes to their probes.

    Sample Usage:
    ------------

    model = nengo.Network(label='_test_probing')
    with model:
        ens1 = nengo.Ensemble(n_neurons=1, dimensions=1)
        node1 = nengo.Node(output=[0])
        conn = nengo.Connection(node1, ens1)
        subnet = nengo.Network(label='subnet')

        with subnet:
            ens2 = nengo.Ensemble(n_neurons=1, dimensions=1)
            node2 = nengo.Node(output=[0])

    probe_options = {nengo.Ensemble:['decoded_output', 'spikes']}
    #will probe the decoded output and spikes in all ensembles
    in this network and its subnetworks.
    object_probe_dict = probe(model, recursive=True,
                              probe_options=probe_options)
    object_probe_dict[ens2]['decoded_output'] #a probe object
    """

    object_probe_dict = {}

    def probe_helper(net, recursive, probe_options):
        with net:
            for object_type, object_list in net.objects.iteritems():

                # recursively probe subnetworks if required
                if object_type is Network and recursive is True:
                    for subnet in object_list:
                        probe_helper(subnet, recursive=recursive,
                                     probe_options=probe_options)

                # probe all probeable objects
                elif probe_options is None:
                    for object in object_list:
                        if hasattr(object, 'probeable') and len(
                                object.probeable) > 0:
                            object_probe_dict[object] = {}
                            for probeable in object.probeable:
                                object_probe_dict[object][probeable] = Probe(
                                    object,
                                    probeable)

                # probe specified objects only
                elif object_type in probe_options:
                    for object in object_list:
                        if not (hasattr(object, 'probeable')
                                and len(object.probeable) > 0):
                            raise ValueError("'%s' is not probeable" % object)
                        object_probe_dict[object] = {}
                        for attr in probe_options[object_type]:
                            if attr not in object.probeable:
                                raise ValueError(
                                    "'%s' is not probeable for '%s'" %
                                    (object, attr))
                            object_probe_dict[object][
                                attr] = Probe(object, attr)

    probe_helper(net, recursive, probe_options)
    return object_probe_dict
