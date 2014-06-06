import nengo
import traceback


class DummyNetwork(nengo.Network):
    @classmethod
    def add(cls, obj):
        # figure out the line number this object was created on
        for fn, line, function, code in reversed(traceback.extract_stack()):
            if fn.endswith('nengo_gui_temp.py'):
                obj._created_line_number = line
                break
        else:
            obj._created_line_number = 0

        original_Network_add(obj)


# monkeypatch this add method into Nengo
original_Network_add = nengo.Network.add
nengo.Network.add = DummyNetwork.add
