import swi
import os.path
import json
import traceback
import sys

import converter
import nengo_helper
import nengo
import os
import urllib

import nengo_gui

class NengoGui(swi.SimpleWebInterface):
    default_filename = 'default.py'
    script_path = 'scripts/'

    def swi_static(self, *path):
        fn = os.path.join('static', *path)
        if fn.endswith('.js'):
            mimetype = 'text/javascript'
        elif fn.endswith('.css'):
            mimetype = 'text/css'
        elif fn.endswith('.png'):
            mimetype = 'image/png'
        elif fn.endswith('.gif'):
            mimetype = 'image/gif'
        else:
            raise Exception('unknown extenstion for %s' % fn)
        with open(fn, 'rb') as f:
            js = f.read()
        return (mimetype, js)

    def swi_favicon_ico(self):
        with open('static/favicon.ico','rb') as f:
            icon = f.read()
        return ('image/ico', icon)

    def swi(self):
        with open('templates/index.html') as f:
            html = f.read()
        return html % dict(filename=self.default_filename)

    @classmethod
    def set_default_filename(klass, fn):
        klass.default_filename = fn
        path, fn = os.path.split(fn)
        klass.script_path = path
        klass.default_filename = fn

    def swi_browse(self, dir):
        r = ['<ul class="jqueryFileTree" style="display: none;">']
        # r.append('<li class="directory collapsed"><a href="#" rel="../">..</a></li>')
        d = urllib.unquote(dir)
        for f in sorted(os.listdir(os.path.join(self.script_path, d))):
            ff = os.path.relpath(os.path.join(self.script_path, d,f), self.script_path)
            if os.path.isdir(ff):
                r.append('<li class="directory collapsed"><a href="#" rel="%s/">%s</a></li>' % (ff,f))
            else:
                e = os.path.splitext(f)[1][1:] # get .ext and remove dot
                if e == 'py':
                    r.append('<li class="file ext_%s"><a href="#" rel="%s">%s</a></li>' % (e,ff,f))
        r.append('</ul>')
        return ''.join(r)

    def swi_openfile(self, filename):
        fn = os.path.join(self.script_path, filename)
        try:
            with open(fn, 'r') as f:
                text = f.read()
            return text
        except:
            return ''

    def swi_savefile(self, filename, code):
        fn = os.path.join(self.script_path, filename)
        with open(fn, 'w') as f:
            f.write(code.replace('\r\n', '\n'))
        return 'success'



    def swi_graph_json(self, code):

        with open('nengo_gui_temp.py', 'w') as f:
            f.write(code.replace('\r\n', '\n'))

        try:
            c = compile(code, 'nengo_gui_temp.py', 'exec')
            locals = {}
            exec c in globals(), locals
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
            cfg = locals.get('gui', None)
            if cfg is None:
                cfg = nengo_gui.Config()

            conv = converter.Converter(model, code.splitlines(), locals, cfg)

        except:
            traceback.print_exc()
            return json.dumps(dict(error_line=2, text='Unknown'))

        return conv.to_json()


if __name__=='__main__':
    import sys
    if len(sys.argv) > 1:
        NengoGui.set_default_filename(sys.argv[1])
    swi.browser(8080)
    swi.start(NengoGui, 8080)
