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

            def handle(network):
                for obj in network.ensembles + network.nodes:
                    node_map[obj] = len(nodes)
                    label = obj.label

                    if ((isinstance(obj, nengo.Ensemble) and label=='Ensemble') or
                            (isinstance(obj, nengo.Node) and label=='Node')):

                        text = code.splitlines()[obj._created_line_number-1]
                        if '=' in text:
                            text = text.split('=', 1)[0].strip()
                            if isidentifier(text):
                                label = text
                    nodes.append(dict(label=label, line=obj._created_line_number-1, id=len(nodes)))
                for net in network.networks:
                    handle(net)
                for c in network.connections:
                    links.append(dict(source=node_map[c.pre], target=node_map[c.post], id=len(links)))
            handle(model)

            """
            for obj in model.ensembles + model.nodes:
                node_map[obj] = len(nodes)

                label = obj.label

                if ((isinstance(obj, nengo.Ensemble) and label=='Ensemble') or
                      (isinstance(obj, nengo.Node) and label=='Node') or
                      (isinstance(obj, nengo.Network) and label=='Network')):

                    text = code.splitlines()[obj._created_line_number-1]
                    if '=' in text:
                        text = text.split('=', 1)[0].strip()
                        if isidentifier(text):
                            obj.label = text


                #nodes.append(dict(label=obj.label, line=obj._created_line_number-1, id=len(nodes)))
                nodes.append(dict(label=obj.label, line=0, id=len(nodes)))
            print node_map
            print model.ensembles
            print model.nodes
            for c in model.connections:
                links.append(dict(source=node_map[c.pre], target=node_map[c.post], id=len(links)))
            """
        except:
            traceback.print_exc()
            return json.dumps(dict(error_line=2, text='Unknown'))


        return json.dumps(dict(nodes=nodes, links=links))


if __name__=='__main__':
    swi.start(NengoGui, 8080, asynch=False)
    #swi.browser(8080)
