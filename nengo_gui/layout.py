import numpy as np

class Item(object):
    def __init__(self, obj, pos, scale, fixed, size=None):
        self.pos = pos
        self.scale = scale
        self.fixed = fixed
        self.size = size #Only networks have a non-zero size
        self.obj = obj
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
        self.items = {} #All items at this level and lower
        self.groups = [] #All subnets at this level and lower
        self.config = config
        self.top_level = self.process_network(model)
        #Add valid positions for top-level items 
        for obj in model.nodes + model.ensembles:
            if self.items[obj].pos is None:
                self.items[obj].pos = np.random.uniform(-1.0, 1.0, size=2)
        self.movement = 0.0

    def process_network(self, network):
        group = Group(network)

        for obj in network.nodes + network.ensembles:
            pos = self.config[obj].pos
            scale = self.config[obj].scale
            if scale is None:
                scale = 1
            if pos is None:
                item = Item(obj, None, scale, fixed=False)
            else:
                item = Item(obj, np.array(pos, copy=True), scale, fixed=True)
            group.add(item)
            self.items[obj] = item
            
        for obj in network.networks:
            pos = self.config[obj].pos
            scale = self.config[obj].scale
            size = self.config[obj].size
            if size is None:
                size = 50,50 #size of empty net
            if scale is None:
                scale = 1
            if pos is None:
                item = Item(obj, None, scale, fixed=False, size=size)
            else:
                item = Item(obj, np.array(pos, copy=True), scale, fixed=True, 
                    size=np.array(size, copy=True))

            g = self.process_network(obj)
            self.groups.append(g)
            group.items.extend(g.items)
            
        return group

    def initialize_nonfixed(self):
        for g in self.groups:
            
            for item in g.items:
                if item.pos is None:  #set position near group mean
                    positions = [i.pos for i in g.items if i.pos is not None]
                    if len(positions) != 0:
                        mid = np.mean(positions)
                    else:
                        mid = 0.0
                    item.pos = mid + np.random.uniform(-1.0, 1.0, size=2)


    def run(self):
        self.initialize_nonfixed()
        for i in range(1000):
            self.step(1.0)
            # stop early if nothing is moving
            if self.movement < 1.0:
                return

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
                        self.push(item, force * dt)
                        self.push(other, -force * dt)

    def push(self, item, distance):
        item.pos += distance
        self.movement += np.linalg.norm(distance)
        
    def store_results(self):
        for k, v in self.items.items():
            self.config[k].pos = v.pos[0], v.pos[1]
            self.config[k].scale = v.scale
            if self.config[k].size is not None: #Only networks have non-zero size
                self.config[k].size = v.size[0], v.size[1]


'''
Algorithms:

2 cases; incremental adding of stuff (could be 'big' stuff); and starting from scratch...
- nice to make these just 1 case

1. layout a network:
- figure out the sizes of all subnetworks
- count how many nodes there are, and assign 100px square to each
- fit the subnetworks within the current network size, moving
as few as possible
- if yes, then fit the nodes in the network size ('''




