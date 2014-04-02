import swi
import os.path
import json
import traceback
import sys

import nengo_helper
import nengo

import re
import keyword
def isidentifier(s):
    if s in keyword.kwlist:
        return False
    return re.match(r'^[a-z_][a-z0-9_]*$', s, re.I) is not None

class NengoGui(swi.SimpleWebInterface):
    def swi_ace(self, *path):
        """Serve the contents of the ace directory"""

        p = os.path.join('ace', *path)
        # TODO: confirm this only allows us to read things inside the ace
        #       directory
        with open(p) as f:
            js = f.read()
        return ('text/javascript', js)

    def swi_d3_min_js(self):
        with open('d3.min.js') as f:
            js = f.read()
        return ('text/javascript', js)


    def swi_favicon_ico(self):
        with open('favicon.ico','rb') as f:
            icon = f.read()
        return ('image/ico', icon)

    def swi(self):
        with open('index.html') as f:
            html = f.read()
        return html

    def swi_graph_json(self, code):

        with open('nengo_gui_temp.py', 'w') as f:
            f.write(code.replace('\r\n', '\n'))

        try:
            c = compile(code, 'nengo_gui_temp.py', 'exec')
            locals = {}
            globals = {}
            exec c in globals, locals
        except (SyntaxError, Exception):
            try:
                e_type, e_value, e_traceback = sys.exc_info()
                tb = traceback.extract_tb(e_traceback)

                if e_type is SyntaxError:
                    error_line = e_value.lineno
                elif e_type is IndentationError:
                    error_line = e_value.lineno
                else:
                    for (fn, line, funcname, text) in reversed(tb):
                        if fn == 'nengo_gui_temp.py':
                            error_line = line
                            break
                    else:
                        print 'Unknown Error'
                        error_line = 0

                print tb
                traceback.print_exc()

                return json.dumps(dict(error_line=error_line, text=str(e_value)))
            except:
                traceback.print_exc()

        try:
            model = locals['model']
            nodes = []
            node_map = {}
            links = []
            networks = []

            def handle(network):

                for obj in network.ensembles:
                    label = obj.label
                    if label == 'Ensemble':
                        text = code.splitlines()[obj._created_line_number-1]
                        if '=' in text:
                            text = text.split('=', 1)[0].strip()
                            if isidentifier(text):
                                label = text
                    nodes.append(dict(label=label, line=obj._created_line_number-1,
                                      id=len(nodes), type='ens'))
                    node_map[obj] = nodes[-1]['id']
                for obj in network.nodes:
                    label = obj.label
                    if label == 'Node':
                        text = code.splitlines()[obj._created_line_number-1]
                        if '=' in text:
                            text = text.split('=', 1)[0].strip()
                            if isidentifier(text):
                                label = text
                    nodes.append(dict(label=label, line=obj._created_line_number-1,
                                      id=len(nodes), type='nde'))
                    node_map[obj] = nodes[-1]['id']
                for net in network.networks:
                    if not hasattr(net, '_created_line_number'):
                        for obj in net.ensembles + net.nodes + net.connections:
                            net._created_line_number = obj._created_line_number
                            break
                        else:
                            net._created_line_number = 0
                    label = net.label
                    if label == None:
                        text = code.splitlines()[obj._created_line_number-1]
                        if '=' in text:
                            text = text.split('=', 1)[0].strip()
                            if isidentifier(text):
                                label = text

                    nodes.append(dict(label=label, line=net._created_line_number-1,
                                      id=len(nodes), type='net'))
                    node_map[net] = nodes[-1]['id']
                    handle(net)
                    for obj in net.ensembles + net.nodes:
                        links.append(dict(source=node_map[net], target=node_map[obj],
                            line=0, id=len(links), type='net'))
                for c in network.connections:
                    links.append(dict(source=node_map[c.pre], target=node_map[c.post],
                                      line=obj._created_line_number-1,
                                      id=len(links), type='std'))
                items = [node_map[obj] for obj in network.ensembles + network.nodes]
                networks.append(dict(label=network.label, items=items))
            handle(model)

        except:
            traceback.print_exc()
            return json.dumps(dict(error_line=2, text='Unknown'))


        return json.dumps(dict(nodes=nodes, links=links, networks=networks))


if __name__=='__main__':
    swi.start(NengoGui, 8080, asynch=False)
    #swi.browser(8080)
