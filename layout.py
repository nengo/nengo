import numpy as np

class Item(object):
    def __init__(self, obj, pos, fixed):
        self.pos = pos
        self.fixed = fixed
        self.obj = obj
    def push(self, d):
        self.pos += d
    def distance2(self, other):
        return np.sum((self.pos-other.pos)**2)
    def distance(self, other):
        return np.sqrt(self.distance2(other))

class Group(object):
    def __init__(self, net):
        self.items = []
        self.net = net
    def add(self, item):
        self.items.append(item)

class Layout(object):
    def __init__(self, model, config):
        self.items = {}
        self.groups = []
        self.config = config
        self.connections = []
        self.process_network(model)

    def process_network(self, network):
        group = Group(network)

        for obj in network.nodes + network.ensembles:
            pos = self.config[obj].pos
            if pos is None:
                item = Item(obj, None, fixed=False)
            else:
                item = Item(obj, np.array(pos, copy=True), fixed=True)
            group.add(item)
            self.items[obj] = item
        for obj in network.networks:
            g = self.process_network(obj)
            self.groups.append(g)
            group.items.extend(g.items)
        for con in network.connections:
            self.connections.append((self.items[con.pre], self.items[con.post]))
        return group

    def center_nonfixed(self):
        for g in self.groups:
            for item in g.items:
                if item.pos is None:
                    mid = np.mean([i.pos for i in g.items if i.pos is not None])
                    item.pos = mid + np.random.uniform(-1.0, 1.0, size=2)


    def run(self):
        self.center_nonfixed()
        for i in range(1000):
            self.step(1.0)

    def step(self, dt):
        for item in self.items.values():
            if not item.fixed:
                for other in self.items.values():
                    d = item.distance(other)
                    if d < 100:
                        force = item.pos - other.pos
                        if d == 0:
                            force = np.array([1.0, 0])
                        else:
                            force /= d
                        item.push(force * dt)
                        other.push(-force * dt)

    def store_results(self):
        for k, v in self.items.items():
            self.config[k].pos = v.pos[0], v.pos[1]







