from . import objects
import numpy as np

def remove_passthrough_nodes(objs, connections):
    inputs = {obj:[] for obj in objs}
    outputs = {obj:[] for obj in objs}
    
    c_removed = set()
    c_new = set()
    
    for c in connections:
        inputs[c.post].append(c)
        outputs[c.pre].append(c)

    for obj in objs:
        if isinstance(obj, objects.Node) and obj.output is None:
            print 'found passthrough node', obj
            
            for c in inputs[obj]:
                c_removed.add(c)
            for c in outputs[obj]:
                c_removed.add(c)
                
            for c_in in inputs[obj]:
                for c_out in outputs[obj]:
                    filter = c_in.filter
                    if filter is None:
                        filter = c_out.filter
                    else:
                        assert c_out.filter is None
                    
                    transform = np.dot(c_in.transform, c_out.transform)
                    #function = 
                    
                    print 'tr', c_in.transform, c_out.transform    
                    
                
            
            
            
            
            
            print '  inputs' 
            for c in inputs[obj]:
                print '    ',`c`
            print '  outputs' 
            for c in outputs[obj]:
                print '    ',`c`
            
    
            


            
        
            
            
            
    return objs, connections            
    
    
