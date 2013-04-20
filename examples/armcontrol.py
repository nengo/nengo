import nef.nef_theano as nef
from __future__ import generators
import sys

import space

from ca.nengo.math.impl import *
from ca.nengo.model.plasticity.impl import *
from ca.nengo.util import *
from ca.nengo.plot import *

from com.bulletphysics import *
from com.bulletphysics.linearmath import *
from com.bulletphysics.dynamics.constraintsolver import *

from math import * 
import java
from java.awt import Color

import ccm
import random
random.seed(11)

from math import pi
from com.threed.jpct import SimpleVector
from com.bulletphysics.linearmath import Transform
from javax.vecmath import Vector3f

dt=0.001
N=1
pstc=0.01

net = nef.Network('simple arm controller')

class getShoulder(ca.nengo.math.Function):
  def map(self,X):
	x = float(X[0])
	y = float(X[1])
	# make sure we're in the unit circle
	if sqrt(x**2+y**2) > 1: 
		x = x / (sqrt(x**2+y**2))
		y = y / (sqrt(x**2+y**2))

	L1 = .5
	L2 = .5
	EPS = 1e-10
	D = (x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2) # law of cosines
	if (x**2+y**2) < (L1**2+L2**2):  
		D = -D

	# find elbow down angles from shoulder to elbow

        #java.lang.System.out.println("x: %f   y:%f"%(x,y))
	if D < 1 and D > -1:
		elbow = acos(D)
	else:
		elbow = 0

	if (x**2+y**2) < (L1**2+L2**2):  
		elbow = pi - elbow
	
	if x==0 and y==0: y = y+EPS

	inside = L2*sin(elbow)/(sqrt(x**2+y**2))
	if inside > 1: inside = 1
	if inside < -1: inside = -1

	if x==0: 
		shoulder = 1.5708 - asin(inside) # magic numbers from matlab
	else:
		shoulder = atan(y/x) - asin(inside)
	if x < 0:  shoulder = shoulder + pi

	return shoulder
  def getDimension(self):
	return 2

class getElbow(ca.nengo.math.Function):
  def map(self,X):
	x = float(X[0])
	y = float(X[1])
	# make sure we're in the unit circle
	if sqrt(x**2+y**2) > 1: 
		x = x / (sqrt(x**2+y**2))
		y = y / (sqrt(x**2+y**2))
	L1 = .5
	L2 = .5
	D = (x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2) # law of cosines
	if (x**2+y**2) < (L1**2+L2**2):  
		D = -D

	# find elbow down angles from shoulder to elbow

	if D < 1 and D > -1:
		elbow = acos(D)
	else:
		elbow = 0

	if (x**2+y**2) < (L1**2+L2**2):  
		elbow = pi - elbow

	return elbow
  def getDimension(self):
	return 2

class getX(ca.nengo.math.Function):
  def map(self,X):
	shoulder = X[0]
	elbow = X[1]
	L1 = .5
	L2 = .5
	
	return L1*cos(shoulder)+L2*cos(shoulder+elbow)
	
  def getDimension(self):
	return 2

class getY(ca.nengo.math.Function):
  def map(self,X):
	shoulder = X[0]
	elbow = X[1]
	L1 = .5
	L2 = .5
	
	return L1*sin(shoulder)+L2*sin(shoulder+elbow)
	
  def getDimension(self):
	return 2


# input functions
refX=net.make_input('refX',[-1])
refY=net.make_input('refY',[1])
Tfunc=net.make_input('T matrix',[1,0,0,1])
F=net.make_input('F',[-1,0,-1,0,0,-1,0,-1])


# neural populations
convertXY=net.make("convert XY",N,2)
convertAngles=net.make("convert Angles",N,2)
funcT=net.make("funcT",N,6)
FX=net.make("FX",N,12)
controlV=net.make("control signal v",N,2) # calculate 2D control signal
controlU=net.make("control signal u",500,2, quick=True) # calculates 
                                                        #jkoint torque control
                                                        #signal

# add terminations
convertXY.addDecodedTermination('refXY',[[1,0],[0,1]],pstc,False)
convertAngles.addDecodedTermination('shoulder',[[1],[0]],pstc,False)
convertAngles.addDecodedTermination('elbow',[[0],[1]],pstc,False)

FX.addDecodedTermination('inputFs',[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0], \
    [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1], \
    [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],
    pstc,False)
FX.addDecodedTermination('X1',[[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],[0],
    [0]],pstc,False)
FX.addDecodedTermination('X2',[[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[0],
    [0]],pstc,False)
FX.addDecodedTermination('X3',[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],
    [0]],pstc,False)
FX.addDecodedTermination('X4',[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
    [1]],pstc,False)

funcT.addDecodedTermination('shoulderRef',[[1],[0],[0],[0],[0],[0]],pstc,False)
funcT.addDecodedTermination('elbowRef',[[0],[1],[0],[0],[0],[0]],pstc,False)
funcT.addDecodedTermination('shoulder',[[0],[0],[0],[0],[0],[0]],pstc,False)
funcT.addDecodedTermination('elbow',[[0],[0],[0],[0],[0],[0]],pstc,False)
funcT.addDecodedTermination('inputTs',[[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0],
    [0,0,1,0],[0,0,0,1]],pstc,False)

controlV.addDecodedTermination('inputCurrentX',[[-1],[0]],pstc,False)
controlV.addDecodedTermination('inputCurrentY',[[0],[-1]],pstc,False)
controlV.addDecodedTermination('inputRefX',[[1],[0]],pstc,False)
controlV.addDecodedTermination('inputRefY',[[0],[1]],pstc,False)

controlU.addDecodedTermination('inputFuncT1',[[1],[0]],pstc,False)
controlU.addDecodedTermination('inputFuncT2',[[0],[1]],pstc,False)
controlU.addDecodedTermination('inputFX1',[[1],[0]],pstc,False)
controlU.addDecodedTermination('inputFX2',[[0],[1]],pstc,False)

# add origins
interpreter=DefaultFunctionInterpreter()

convertXY.addDecodedOrigin('elbowRef',[getElbow()],"AXON")
convertXY.addDecodedOrigin('shoulderRef',[getShoulder()],"AXON")

convertAngles.addDecodedOrigin('currentX',[getX()],"AXON")
convertAngles.addDecodedOrigin('currentY',[getY()],"AXON")

FX.addDecodedOrigin('FX1',[interpreter.parse("x0*x8+x1*x9+x2*x10+x3*x11",12)],
    "AXON")
FX.addDecodedOrigin('FX2',[interpreter.parse("x4*x8+x5*x9+x6*x10+x7*x11",12)],
    "AXON")

funcT.addDecodedOrigin('funcT1',[interpreter.parse("x0*x2+x1*x3",6)],"AXON")
funcT.addDecodedOrigin('funcT2',[interpreter.parse("x0*x4+x1*x5",6)],"AXON")

controlU.addDecodedOrigin('u1',[interpreter.parse("x0",2)],"AXON")
controlU.addDecodedOrigin('u2',[interpreter.parse("x1",2)],"AXON")

# add projections 
net.connect(controlV.getOrigin('X'),convertXY.getTermination('refXY'))

net.connect(refX.getOrigin('origin'),controlV.getTermination('inputRefX'))
net.connect(refY.getOrigin('origin'),controlV.getTermination('inputRefY'))

net.connect(convertAngles.getOrigin('currentX'),controlV.getTermination(
    'inputCurrentX'))
net.connect(convertAngles.getOrigin('currentY'),controlV.getTermination(
    'inputCurrentY'))

net.connect(F.getOrigin('origin'),FX.getTermination('inputFs'))

net.connect(convertXY.getOrigin('shoulderRef'),funcT.getTermination(
    'shoulderRef'))
net.connect(convertXY.getOrigin('elbowRef'),funcT.getTermination('elbowRef'))

net.connect(Tfunc.getOrigin('origin'),funcT.getTermination('inputTs'))

net.connect(funcT.getOrigin('funcT1'),controlU.getTermination('inputFuncT1'))
net.connect(funcT.getOrigin('funcT2'),controlU.getTermination('inputFuncT2'))
net.connect(FX.getOrigin('FX1'),controlU.getTermination('inputFX1'))
net.connect(FX.getOrigin('FX2'),controlU.getTermination('inputFX2'))

net.add_to_nengo()


class Room(space.Room):
    def __init__(self):
        space.Room.__init__(self,10,10,gravity=0,color=[Color(0xFFFFFF),
            Color(0xFFFFFF),Color(0xEEEEEE),Color(0xDDDDDD),
            Color(0xCCCCCC),Color(0xBBBBBB)])
    def start(self):    
        
        self.target=space.Sphere(0.2,mass=1,color=Color(0xFF0000))
        self.add(self.target,0,0,2)
        
        torso=space.Box(0.1,0.1,1.5,mass=100000,draw_as_cylinder=True,
            color=Color(0x4444FF))
        self.add(torso,0,0,1)
        
        upperarm=space.Box(0.1,0.7,0.1,mass=0.5,draw_as_cylinder=True,
            color=Color(0x8888FF),overdraw_radius=1.2,overdraw_length=1.2)
        self.add(upperarm,0.7,0.5,2)
        upperarm.add_sphere_at(0,0.5,0,0.1,Color(0x4444FF),self)
        upperarm.add_sphere_at(0,-0.5,0,0.1,Color(0x4444FF),self)
        
        lowerarm=space.Box(0.1,0.75,0.1,mass=0.1,draw_as_cylinder=True,
            color=Color(0x8888FF),overdraw_radius=1.2,overdraw_length=1.1)
        self.add(lowerarm,0.7,1.5,2)

        shoulder=HingeConstraint(torso.physics,upperarm.physics,
                            Vector3f(0.7,0.1,1),Vector3f(0,-0.5,0),
                            Vector3f(0,0,1),Vector3f(0,0,1))
               
        elbow=HingeConstraint(upperarm.physics,lowerarm.physics,
                            Vector3f(0,0.5,0),Vector3f(0,-0.5,0),
                            Vector3f(0,0,1),Vector3f(0,0,1))
                            
        shoulder.setLimit(-pi/2,pi/2+.1)
        elbow.setLimit(-pi,0)
        
        self.physics.addConstraint(elbow)
        self.physics.addConstraint(shoulder)
       
        #upperarm.physics.applyTorqueImpulse(Vector3f(0,0,300))
        #lowerarm.physics.applyTorqueImpulse(Vector3f(0,0,300))
            
        self.sch.add(space.Room.start,args=(self,))
        self.update_neurons()
        self.upperarm=upperarm
        self.lowerarm=lowerarm
        self.shoulder=shoulder
        self.elbow=elbow
        self.hinge1=self.shoulder.hingeAngle
        self.hinge2=self.elbow.hingeAngle
        self.upperarm.physics.setSleepingThresholds(0,0)
        self.lowerarm.physics.setSleepingThresholds(0,0)
            
    def update_neurons(self):
        while True:
            scale=0.0003
            m1=controlU.getOrigin('u1').getValues().getValues()[0]*scale
            m2=controlU.getOrigin('u2').getValues().getValues()[0]*scale
            v1=Vector3f(0,0,0)
            v2=Vector3f(0,0,0)
            #java.lang.System.out.println("m1: %f   m2:%f"%(m1,m2))

            self.upperarm.physics.applyTorqueImpulse(Vector3f(0,0,m1))
            self.lowerarm.physics.applyTorqueImpulse(Vector3f(0,0,m2))

            self.hinge1=-(self.shoulder.hingeAngle-pi/2)
            self.hinge2=-self.elbow.hingeAngle
            #java.lang.System.out.println("angle1: %f 
            #angle2:%f"%(self.hinge1,self.hinge2))   
	
            self.upperarm.physics.getAngularVelocity(v1)
            self.lowerarm.physics.getAngularVelocity(v2)
            # put bounds on the velocity possible
            if v1.z > 2: 
                self.upperarm.physics.setAngularVelocity(Vector3f(0,0,2))
            if v1.z < -2: 
                self.upperarm.physics.setAngularVelocity(Vector3f(0,0,-2))
            if v2.z > 2: 
                self.lowerarm.physics.setAngularVelocity(Vector3f(0,0,2))
            if v2.z < -2: 
                self.lowerarm.physics.setAngularVelocity(Vector3f(0,0,-2))
            self.upperarm.physics.getAngularVelocity(v1)
            self.lowerarm.physics.getAngularVelocity(v2)
            
            wt=Transform()
            #self.target.physics.motionState.getWorldTransform(wt)
            wt.setIdentity()
            
            tx=controlV.getTermination('inputRefX').input
            if tx is not None:
                wt.origin.x=tx.values[0]+0.7
            else:    
                wt.origin.x=0.7
            ty=controlV.getTermination('inputRefY').input
            if ty is not None:
                wt.origin.y=ty.values[0]+0.1
            else:    
                wt.origin.y=0.1
            wt.origin.z=2

            ms=self.target.physics.motionState            
            ms.worldTransform=wt            
            self.target.physics.motionState=ms
            


            
            
            self.vel1=v1.z
            self.vel2=v2.z
	
           
            yield 0.0001
    
r=ccm.nengo.create(Room)
net.add(r)

# need to make hinge1, hinge2, vel1, and vel external nodes and hook up 
# the output to the FX matrix
r.exposeOrigin(r.getNode('hinge1').getOrigin('origin'),'shoulderAngle')
r.exposeOrigin(r.getNode('hinge2').getOrigin('origin'),'elbowAngle')
r.exposeOrigin(r.getNode('vel1').getOrigin('origin'),'shoulderVel')
r.exposeOrigin(r.getNode('vel2').getOrigin('origin'),'elbowVel')

net.connect(r.getOrigin('shoulderAngle'),FX.getTermination('X1'))
net.connect(r.getOrigin('elbowAngle'),FX.getTermination('X2'))
net.connect(r.getOrigin('shoulderVel'),FX.getTermination('X3'))
net.connect(r.getOrigin('elbowVel'),FX.getTermination('X4'))

net.connect(r.getOrigin('shoulderAngle'),convertAngles.getTermination(
    'shoulder'))
net.connect(r.getOrigin('elbowAngle'),convertAngles.getTermination('elbow'))
net.connect(r.getOrigin('shoulderAngle'),funcT.getTermination('shoulder'))
net.connect(r.getOrigin('elbowAngle'),funcT.getTermination('elbow'))


# put everything in direct mode
net.network.setMode(ca.nengo.model.SimulationMode.DIRECT)
# except the last population
controlU.setMode(ca.nengo.model.SimulationMode.DEFAULT)


	
	
