

class NameFinder(object):
    def __init__(self, terms, net):
        self.base_terms = terms
        self.known_name = {}
        for k, v in terms.iteritems():
            self.known_name[id(v)] = k
        self.find_names(net)

    def find_names(self, net):
        """Gets variable names associated to objects. 
        If the object is not assigned to a variable, creates new name"""
        net_name = self.known_name[id(net)]

        base_lists = ['ensembles', 'nodes', 'connections', 'networks']

        for k in dir(net):
            # If it's not a private attribute, a built in function 
            # and not a Nengo object
            if not k.startswith('_') and k not in base_lists:
                # iterate through the attributes of the network that are a list
                v = getattr(net, k)
                if isinstance(v, list):
                    for i, obj in enumerate(v):
                        # If this object is not already a known name
                        if not self.known_name.has_key(id(obj)):
                            # Combine the name, the network, the attribute and 
                            # the index for the new identifier
                            # This happens for things like connections 
                            # and other objects
                            n = '%s.%s[%d]' % (net_name, k, i)
                            self.known_name[id(obj)] = n
                else:
                    # If it's not a list, no need to iterate
                    self.known_name[id(v)] = '%s.%s' % (net_name, k)


        for base_type in base_lists:
            for i, obj in enumerate(getattr(net, type)):
                name = self.known_name.get(id(obj), None)
                # If there was no name found already in the known names
                if name is None:
                    name = '%s.%s[%d]' % (net_name, base_type, i)
                    self.known_name[id(obj)] = name

        for n in net.networks:
            self.find_names(n)

    def name(self, obj):
        return self.known_name[id(obj)]




if __name__ == '__main__':
    c = compile(open('scripts/default.py').read(), 'nengo_gui_temp.py', 'exec')
    locals = {}
    exec c in globals(), locals

    model = locals['model']
    ident = NameFinder(locals, model)
    ident.find_names(locals['model'])

    print ident.known_name


