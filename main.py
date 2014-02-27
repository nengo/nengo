import swi
import os.path
import json

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
        
    def swi_graph_json(self):
        nodes = [
            dict(label='a', line=4),
            dict(label='b', line=5),
            dict(label='c', line=6),
            dict(label='d', line=7),
            ]
        links = [
            dict(source=0, target=1),
            dict(source=1, target=2),
            dict(source=2, target=3),
            dict(source=1, target=3),
            ]
        return json.dumps(dict(nodes=nodes, links=links))
        

if __name__=='__main__':
    swi.start(NengoGui, 8080)
    #swi.browser(8080)
