import nengo_gui.swi
import os.path
import json
import traceback
import sys

from nengo_gui.feedforward_layout import feedforward_layout
import nengo_gui.converter
import nengo_gui.layout
import nengo_gui.nengo_helper
import nengo_gui.namefinder
import nengo
import os
import urllib

import nengo_gui
import pkgutil

import socket
try:
    import rpyc
    s = rpyc.classic.connect('localhost')
    assert s.modules.timeview.javaviz.__name__ == 'timeview.javaviz'
    import javaviz
    javaviz_message = 'run with JavaViz'
except ImportError:
    javaviz_message = 'JavaViz disabled as rpyc is not installed.'
    javaviz_message += ' Try "pip install rpyc"'
    javaviz = None
except socket.error:
    javaviz_message = 'JavaViz disabled as the javaviz server is not running'
    javaviz = None
except AssertionError:
    javaviz_message = 'JavaViz disabled due to an unknown server error.'
    javaviz_message += ' Please reinstall and re-run the JavaViz server'
    javaviz = None



class NengoGui(nengo_gui.swi.SimpleWebInterface):
    default_filename = 'default.py'
    script_path = os.path.join(os.path.dirname(nengo_gui.__file__), 'scripts')
    refresh_interval = 0

    def swi_static(self, *path):
        if self.user is None: return
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

        data = pkgutil.get_data('nengo_gui', fn)
        return (mimetype, data)

    def swi_favicon_ico(self):
        icon = pkgutil.get_data('nengo_gui', 'static/favicon.ico')
        return ('image/ico', icon)

    def swi(self):
        if self.user is None:
            return self.create_login_form()
        html = pkgutil.get_data('nengo_gui', 'templates/index.html')
        if javaviz is None:
            use_javaviz = 'false'
        else:
            use_javaviz = 'true'
        return html % dict(filename=self.default_filename,
                           refresh_interval=self.refresh_interval,
                           use_javaviz=use_javaviz,
                           javaviz_message=javaviz_message)

    def create_login_form(self):
        message = "Enter the password:"
        if self.attemptedLogin:
            message = "Invalid password"
        return """<form action="/" method="POST">%s<br/>
        <input type=hidden name=swi_id value="" />
        <input type=password name=swi_pwd>
        </form>""" % message

    @classmethod
    def set_default_filename(klass, fn):
        klass.default_filename = fn
        path, fn = os.path.split(fn)
        klass.script_path = path
        klass.default_filename = fn

    @classmethod
    def set_refresh_interval(klass, interval):
        klass.refresh_interval = interval

    def swi_browse(self, dir):
        if self.user is None: return
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
        if self.user is None: return
        fn = os.path.join(self.script_path, filename)
        try:
            with open(fn, 'r') as f:
                text = f.read()
            # make sure there are no tabs in the file, since the editor is
            # supposed to use spaces instead
            text = text.replace('\t', '    ')
            modified_time = os.stat(fn).st_mtime
        except:
            text = ''
            modified_time = None
        return json.dumps(dict(text=text, mtime=modified_time))


    def swi_savefile(self, filename, code):
        if self.user is None: return
        fn = os.path.join(self.script_path, filename)
        with open(fn, 'w') as f:
            f.write(code.replace('\r\n', '\n'))
        return 'success'

    def swi_modified_time(self, filename):
        if self.user is None: return
        fn = os.path.join(self.script_path, filename)
        return repr(os.stat(fn).st_mtime)

    def swi_javaviz(self, code):
        if self.user is None: return
        code = code.replace('\r\n', '\n')

        locals = {}
        exec code in globals(), locals

        model = locals['model']
        cfg = locals.get('gui', None)
        if cfg is None:
            cfg = nengo_gui.Config()

        nf = nengo_gui.namefinder.NameFinder(locals, model)

        javaviz.View(
            model, default_labels=nf.known_name, config=cfg)

        sim = nengo.Simulator(model)
        try:
            while True:
                sim.run(1)
        except javaviz.VisualizerExitException:
            print('Finished running JavaViz simulation')


    def swi_graph_json(self, code):
        if self.user is None: return

        code = code.replace('\r\n', '\n')

        try:
            index = code.index('\nimport nengo_gui\n')
            code_gui = code[index:]
            code = code[:index]
        except ValueError:
            code_gui = ''

        with open('nengo_gui_temp.py', 'w') as f:
            f.write(code)

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

        # run gui code lines, skipping ones that cause name errors
        for i, line in enumerate(code_gui.splitlines()):
            try:
                exec line in globals(), locals
            except NameError:
                # this is generally caused by having a gui[x].pos statement
                #  for something that has been deleted
                pass
            except IndexError:
                # this is generally caused by having a statement like
                # gui[model.ensemble[i]].pos statement for something that has
                # been deleted
                pass

        try:
            model = locals['model']
            cfg = locals.get('gui', None)
            if cfg is None:
                cfg = nengo_gui.Config()
        except:
            traceback.print_exc()
            return json.dumps(dict(error_line=2, text='Unknown'))

        feedforward = True

        if feedforward:
            conv = nengo_gui.converter.Converter(model, code.splitlines(), locals, cfg)
            feedforward_layout(model, cfg, locals, conv.links, conv.objects)
        else:
            #import pdb
            #pdb.set_trace()
            gui_layout = nengo_gui.layout.Layout(model, cfg)
            #pdb.set_trace()
            cfg = gui_layout.config

        conv = nengo_gui.converter.Converter(model, code.splitlines(), locals, cfg)

        return conv.to_json()
