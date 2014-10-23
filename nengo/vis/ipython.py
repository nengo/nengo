import pkgutil

from IPython.display import display, HTML


class ModelGraphDisplay(object):
    def __init__(self, graph):
        self.graph = graph

    def _ipython_display_(self):
        js = pkgutil.get_data('nengo.vis', 'static/js/main.js')
        d3 = pkgutil.get_data('nengo.vis', 'static/js/vendor/d3.min.js')
        css = pkgutil.get_data('nengo.vis', 'static/css/graph.css')
        display(HTML('''
            <script>
                if (typeof(d3) == 'undefined') {{
                    {d3}
                }}
            </script>
            <style type="text/css">{css}</style>
            <script>
                {js}
                main({data});
            </script>
            <div id="#graph"><svg></svg></div>'''.format(
                js=js, d3=d3, css=css, data=self.graph.to_json())))
