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
            <div id="#graph"><svg>
                <defs>
                    // the arrow on the links
                    <marker id="TriangleMarker"
                            viewBox="0 0 10 10" refX="0" refY="5"
                            markerUnits="strokeWidth"
                            markerWidth="6" markerHeight="4"
                            orient="auto">
                        <path d="M 0 0 L 10 5 L 0 10 z" />
                    </marker>
                    <g id ="ensemble" transform="translate(-17.2735,-17.7015)">
                        <circle id="mainCircle" cx="16" cy="18" r="18"/>
                        <circle cx="4.843" cy="10.519" r="4.843"/>
                        <circle cx="16.186" cy="17.873" r="4.843"/>
                        <circle cx="21.012" cy="30.56" r="4.843"/>
                        <circle cx="29.704" cy="17.229" r="4.843"/>
                        <circle cx="5.647" cy="26.413" r="4.843"/>
                        <circle cx="19.894" cy="4.842" r="4.843"/>
                    </g>
                    <g id = "recur">
                        <path
                            d="M6.451,28.748C2.448,26.041,0,22.413,0,18.425C0,
                            10.051,10.801,3.262,24.125,3.262
                            S48.25,10.051,48.25,18.425c0,6.453-6.412,11.964-15.45,14.153"/>
                    </g>
                    <g>
                        <path id = "recurTriangle"
                        d="M 8 0 L 0 4 L 8 8 z""/>
                    </g>
                </defs>
            </svg></div>'''.format(
                js=js, d3=d3, css=css, data=self.graph.to_json())))
