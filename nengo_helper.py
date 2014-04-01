import nengo
import traceback


def find_creation_line():
    for fn, line, function, code in reversed(traceback.extract_stack()):
        if fn == 'nengo_gui_temp.py':
            return line
    return 0

class EnsembleHelper(nengo.Ensemble):
    def add_to_network(self, network):
        super(EnsembleHelper, nengo.Ensemble).add_to_network(self, network)
        self._created_line_number = find_creation_line()

class NodeHelper(nengo.Node):
    def add_to_network(self, network):
        super(NodeHelper, nengo.Node).add_to_network(self, network)
        self._created_line_number = find_creation_line()

class NetworkHelper(nengo.Network):
    def add_to_network(self, network):
        super(NetworkHelper, nengo.Network).add_to_network(self, network)
        self._created_line_number = find_creation_line()

class ConnectionHelper(nengo.Connection):
    def add_to_network(self, network):
        super(ConnectionHelper, nengo.Connection).add_to_network(self, network)
        self._created_line_number = find_creation_line()

nengo.Ensemble = EnsembleHelper
nengo.Node = NodeHelper
nengo.Connection = ConnectionHelper
nengo.Network = NetworkHelper

