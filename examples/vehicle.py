from __future__ import generators
import nef.nef_theano as nef
import space
from java.awt import Color
import ccm
import random

dt=0.001
N=10

class Bot(space.MD2):
    def __init__(self):
        space.MD2.__init__(self,'python/md2/dalekx/tris.md2', 'python/md2/dalekx/imperial.png', 
                            scale=0.02, mass=800,overdraw_scale=1.4)
        z=0
        s=0.7
        r=0.7
        self.wheels=[space.Wheel(-s,0,z,radius=r),
                     space.Wheel(s,0,z,radius=r),
                     space.Wheel(0,s,z,friction=0,radius=r),
                     space.Wheel(0,-s,z,friction=0,radius=r)]
        
    def start(self):
        self.sch.add(space.MD2.start,args=(self,))
        self.range1=space.RangeSensor(0.3,1,0,maximum=6)
        self.range2=space.RangeSensor(-0.3,1,0,maximum=6)
        self.wheel1=0
        self.wheel2=0
        while True:        
            r1=self.range1.range
            r2=self.range2.range
            input1.functions[0].value=r1-1.8
            input2.functions[0].value=r2-1.8
            
            f1=motor1.getOrigin('X').getValues().getValues()[0]
            f2=motor2.getOrigin('X').getValues().getValues()[0]
            
            self.wheels[1].force=f1*600
            self.wheels[0].force=f2*600
            yield dt
            

class Room(space.Room):
    def __init__(self):
        space.Room.__init__(self,10,10,dt=0.01)
    def start(self):    
        self.bot=Bot()
        self.add(self.bot, 0, 0,1)
        #view=space.View(self, (0, -10, 5))


        for i in range(6):
            self.add(space.Box(1,1,1,mass=1,color=Color(0x8888FF),
            flat_shading=False),random.uniform(-5,5),random.uniform(-5,5),
            random.uniform(4,6))
        
        self.sch.add(space.Room.start,args=(self,))

from ca.nengo.util.impl import NodeThreadPool, NEFGPUInterface

net=nef.Network('Braitenberg')

input1=net.make_input('right eye',[0])
input2=net.make_input('left eye',[0])

sense1=net.make("right input",N,1)
sense2=net.make("left input",N,1)
motor1=net.make("right motor",N,1)
motor2=net.make("left motor",N,1)

net.connect(input1,sense1)
net.connect(input2,sense2)
net.connect(sense2,motor1)
net.connect(sense1,motor2)

net.add_to_nengo()
    
r=ccm.nengo.create(Room)    
net.add(r)



