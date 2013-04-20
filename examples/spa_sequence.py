from spa import *

D=16

class Rules: #Define the rules by specifying the start state and the 
             #desired next state
    def A(state='A'): #e.g. If in state A
        set(state='B') # then go to state B
    def B(state='B'):
        set(state='C')
    def C(state='C'):
        set(state='D')
    def D(state='D'):
        set(state='E')
    def E(state='E'):
        set(state='A')
    


class Sequence(SPA): #Define an SPA model (cortex, basal ganglia, thalamus)
    dimensions=16
    
    state=Buffer() #Create a working memory (recurrent network) object: 
                   #i.e. a Buffer
    BG=BasalGanglia(Rules()) #Create a basal ganglia with the prespecified 
                             #set of rules
    thal=Thalamus(BG) # Create a thalamus for that basal ganglia (so it 
                      # uses the same rules)
    
    input=Input(0.1,state='D') #Define an input; set the input to 
                               #state D for 100 ms

seq=Sequence()
