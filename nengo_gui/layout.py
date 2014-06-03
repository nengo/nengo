import numpy as np

class Item(object):
    def __init__(self, obj, changed, fixed):
        self.changed = changed #did the net size change
        self.fixed = fixed #is the position fixed
        self.obj = obj

class Layout(object):
    def __init__(self, model, config):
        self.nets = {} #All nets at this level
        self.config = config
        self.model = model
        if not self.config[model].scale:
            self.config[model].scale = 1
        self.process_network(model)

    def process_network(self, network): #layout of the network
        contain_size = self.config[network].size
        contain_scale = self.config[network].scale
        contain_pos = self.config[network].pos
        if network==self.model:
            contain_pos = self.middle(network.networks + network.nodes
                + network.ensembles)
        changed = False

        for obj in network.networks:
            size, changed = self.process_network(obj)
            self.config[obj].size = size

            if pos is None:
                net = Item(obj, changed=True, fixed=False)
            else:
                net = Item(obj, changed, fixed=True)

            self.nets.append(net)

        fixed = [o.obj for o in self.nets if o.fixed]
        fixed += [o for o in network.nodes+network.ensembles
            if self.config[o].pos != None]

        floating = [o.obj for o in self.nets if not o.fixed]
        floating += [o for o in network.nodes+network.ensembles
            if self.config[o].pos == None]

        for obj in floating:
            self.config[obj].scale = 1
            size = self.config[obj].size

            if fixed:
                pos = self.middle(fixed)
            else:
                pos = contain_pos

            pos = self.find_position(fixed, pos, size)

            self.config[obj].pos = pos
            fixed.append(obj)

            if contain_size:
                if not self.is_in(obj, contain_pos, contain_size):
                    changed = True

        #compute size to return
        if len(network.nodes + network.ensembles) == 0: #empty network
            contain_size = 100,100 #default empty network size
        elif [o for o in self.nets if o.changed]: #if any subnetwork changed size
            if network == self.model: #if we're at the top, do nothing
                pass
            else:
                contain_size = network_size(network)
        else: #contain_size already set at entry
            pass

        return contain_size, changed

    def network_size(self, net):
        nodes = net.ensembles + net.nodes
        x0 = self.config[nodes[0]].pos[0] #first item in net x,y as a start
        x1 = x0
        y0 = self.config[nodes[0]].pos[1]
        y1 = y0
        m = 40 #net_inner_margin

        for obj in nodes + net.networks:
            scale = self.config[obj].scale
            xBorder = (self.config[obj].size[0] / 2)*scale
            yBorder = (self.config[obj].size[1] / 2)*scale

            x = config[obj].pos[0]
            y = config[obj].pos[1]

            x0 = np.min([x - xBorder, x0])
            x1 = np.max([x + xBorder, x1])
            y0 = np.min([y - yBorder, y0])
            y1 = np.max([y + yBorder, y1])

        self.config[net].pos[0] = (x0 + x1) / 2 # x, y mid
        self.config[net].pos[1] = (y0 + y1) / 2

        xsize = (x1 - x0)/self.config[net].scale + 2 * m
        ysize = (y1 - y0)/self.config[net].scale + 2 * m

        return xsize, ysize

    def find_position(self, objects, pos, size):
        no_position = True
        iters = 0

        while no_position and iters<500:
            no_position = False
            for obj in objects: #check if current pos is valid
                if self.is_in(obj, pos, size):
                    no_position = True
                    pos += ((np.random.random(2) - .5) * self.config[obj].size
                         + (pos-self.config[obj].pos))
                    break

            iters +=1

        print(iters, pos)

        if iters >= 499:
            print '\n***Warning: Too many iterations, exiting\n'

        return pos

    def is_in(self, obj, pos, size): 
        #Determine if obj is inside the pos/size provided
        if self.config[obj]:
            if (self.config[obj].size is not None and
               self.config[obj].pos is not None):
                this_size = self.config[obj].size
                this_pos = self.config[obj].pos
                ax = pos[0] - size[0]/2, pos[0] + size[0]/2
                ay = pos[1] - size[1]/2, pos[1] + size[1]/2
                bx = this_pos[0] - this_size[0]/2, this_pos[0] + this_size[0]/2
                by = this_pos[1] - this_size[1]/2, this_pos[1] + this_size[1]/2

                if ((ax[1] > bx[0] and ax[0] < bx[1]) and #x overlap
                    (ay[1] > by[0] and ay[0] < by[1])): #y overlap
                    return True

        return False

    def middle(self, objects):
        mid = np.array([0.0, 0.0])
        obj_cnt = 0

        for obj in objects:
            if self.config[obj].pos is not None:
                mid+=np.array(self.config[obj].pos)
                obj_cnt += 1

        if obj_cnt > 0:
            mid = mid/obj_cnt

        return mid


