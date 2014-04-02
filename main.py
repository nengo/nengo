import swi
import os.path
import json
import traceback
import sys

import converter
import nengo_helper
import nengo

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

            conv = converter.Converter(model, code.splitlines())

        except:
            traceback.print_exc()
            return json.dumps(dict(error_line=2, text='Unknown'))

        print conv.to_json()
        return conv.to_json()


if __name__=='__main__':
    swi.start(NengoGui, 8080, asynch=False)
    #swi.browser(8080)
