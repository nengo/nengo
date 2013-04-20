from spa import *

D=16

class Rules: #Define the rules by specifying the start state and the 
             #desired next state
    def start(vision='START'):
        set(state=vision)
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
    


class Routing(SPA): #Define an SPA model (cortex, basal ganglia, thalamus)
    dimensions=16

    state=Buffer() #Create a working memory (recurrent network) 
                   #object: i.e. a Buffer
    vision=Buffer(feedback=0) #Create a cortical network object with no 
                              #recurrence (so no memory properties, just 
                              #transient states)
    BG=BasalGanglia(Rules) #Create a basal ganglia with the prespecified 
                           #set of rules
    thal=Thalamus(BG) # Create a thalamus for that basal ganglia (so it 
                      # uses the same rules)

    input=Input(0.1,vision='0.8*START+D') #Define an input; set the input 
                                       #to state 0.8*START+D for 100 ms

model=Routing()
