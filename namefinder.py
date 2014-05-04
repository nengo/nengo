

class NameFinder(object):
    def __init__(self, terms, net):
        self.base_terms = terms
        self.known_name = {}
        for k, v in terms.iteritems():
            self.known_name[id(v)] = k
        self.find_names(net)

    def find_names(self, net):
        net_name = self.known_name[id(net)]

        base_lists = ['ensembles', 'nodes', 'connections', 'networks']

        for k in dir(net):
            if not k.startswith('_') and k not in base_lists:
                v = getattr(net, k)
                if isinstance(v, list):
                    for i, obj in enumerate(v):
                        if not self.known_name.has_key(id(obj)):
                            n = '%s.%s[%d]' % (net_name, k, i)
                            self.known_name[id(obj)] = n
                elif isinstance(v, dict):
                    for kk, obj in v.items():
                        if not self.known_name.has_key(id(obj)):
                            n = '%s.%s[%s]' % (net_name, k, kk)
                            self.known_name[id(obj)] = n
                else:
                    self.known_name[id(v)] = '%s.%s' % (net_name, k)


        for type in base_lists:
            for i, obj in enumerate(getattr(net, type)):
                name = self.known_name.get(id(obj), None)
                if name is None:
                    name = '%s.%s[%d]' % (net_name, type, i)
                    self.known_name[id(obj)] = name

        for n in net.networks:
            self.find_names(n)

    def name(self, obj):
        return self.known_name[id(obj)]




if __name__ == '__main__':
    c = compile(open('scripts/default.py').read(), 'nengo_gui_temp.py', 'exec')
    locals = {}
    exec c in globals(), locals

    print locals

    ident = NameFinder(locals)
    ident.find_names(locals['model'])

    print ident.known_name


