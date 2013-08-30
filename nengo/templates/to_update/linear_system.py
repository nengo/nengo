title='General Linear System'
label='Linear\nSystem'
icon='linearsystem.png'

from ca.nengo.ui.configurable import *
from ca.nengo.ui.configurable.managers import ConfigManager
from ca.nengo.ui.configurable.descriptors import PCouplingMatrix
from ca.nengo.util import MU
from ca.nengo.ui.lib.util import UserMessages

from javax.swing import *
from javax.swing.event import DocumentListener

class SystemMatrixInputPanel(PropertyInputPanel,DocumentListener):
    def __init__(self,property):
        PropertyInputPanel.__init__(self,property)
        self.add(JLabel("State dimension: "))    
        self.state_dim=JTextField(10)
        self.state_dim.document.addDocumentListener(self)
        self.add(self.state_dim)
        self.add(JButton("Edit matrix",actionPerformed=self.edit))
        self.matrix=None
    def edit(self,event):
        try:
            dim=int(self.state_dim.text)
        except Exception:
            UserMessages.showWarning("Specify the number of state dimensions first")
            return
        parent=self.getJPanel().getParent()
        while parent is not None:
            if isinstance(parent,JDialog): break
            parent=parent.getParent()
        if self.matrix is not None:
            p=PCouplingMatrix(self.matrix)
        else:
            p=PCouplingMatrix(dim,dim)
        result=ConfigManager.configure([p],"System Matrix", parent, ConfigManager.ConfigMode.STANDARD)
        self.matrix=result.getValue(p)

    def change_dim(self,event):
        try:
            dim=int(self.state_dim.text)
            if self.matrix is None or len(self.matrix)!=dim:
                self.matrix=None
        except:
            self.matrix=None
        
    def isValueSet(self):
        return self.matrix is not None
    def getValue(self):
        return MU.clone(self.matrix)
    def setValue(self,matrix):
        self.matrix=matrix
        self.state_dim.text=str(len(matrix))

    def changedUpdate(self,event):
        self.change_dim(event)
    def insertUpdate(self,event):
        self.change_dim(event)
    def removeUpdate(self,event):
        self.change_dim(event)
    
class PSystemMatrix(Property):
    matrix=None
    def createInputPanel(self):
        return SystemMatrixInputPanel(self)
    def getTypeName(self):
        return "System Matrix"
    def getTypeClass(self):
        return PCouplingMatrix(1,1).getTypeClass()
        
description="""<html>This is a generic template for constructing a recurrent network that implements the specified dynamic linear system: dx/dt = Ax + Bu. Input Bu must be supplied after construction by adding a termination with the B matrix.</html>"""

params=[
    ('name','Name',str,'Name of the new linear system'),
    ('neurons','Number of Neurons',int,'Number of neurons in the new linear system'),
    ('tau_feedback','Feedback PSTC [s]',float,'Post-synaptic time constant for the integrative feedback, in seconds'),
    ('A','System dynamics matrix',PSystemMatrix,'The system dynamics matrix, A, of the new linear system'),
    ]

def test_params(net,p):
    try:
       net.network.getNode(p['name'])
       return 'That name is already taken'
    except:
        pass
    if p['neurons']<1: return 'Must have a positive number of neurons'


import numeric
from java.util import ArrayList
from java.util import HashMap
def make(net,name='System',neurons=100,A=[[0]],tau_feedback=0.1):
    A=numeric.array(A)
    assert len(A.shape)==2
    assert A.shape[0]==A.shape[1]
    
    dimensions=A.shape[0]
    state=net.make(name,neurons,dimensions)
    Ap=A*tau_feedback+numeric.identity(dimensions)

    net.connect(state,state,transform=Ap,pstc=tau_feedback)
    if net.network.getMetaData("linear") == None:
        net.network.setMetaData("linear", HashMap())
    linears = net.network.getMetaData("linear")

    linear=HashMap(4)
    linear.put("name", name)
    linear.put("neurons", neurons)
    linear.put("A", MU.clone(A))
    linear.put("tau_feedback", tau_feedback)

    linears.put(name, linear)

    if net.network.getMetaData("templates") == None:
        net.network.setMetaData("templates", ArrayList())
    templates = net.network.getMetaData("templates")
    templates.add(name)

    if net.network.getMetaData("templateProjections") == None:
        net.network.setMetaData("templateProjections", HashMap())
    templateproj = net.network.getMetaData("templateProjections")
    templateproj.put(name, name)
    
