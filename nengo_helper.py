import nengo
import traceback

class ModelHelper(nengo.Model):
    def add(self, obj):
        super(ModelHelper, nengo.Model).add(self, obj)
        
        for fn, line, function, code in reversed(traceback.extract_stack()):
            if fn == 'nengo_gui_temp.py':
                obj._created_line_number = line
                break
        else:
            obj._created_line_number = 0
        
nengo.Model = ModelHelper        

