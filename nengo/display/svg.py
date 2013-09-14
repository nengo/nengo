import json

class SVG(object):
    """Formats nengo models into a D3 force layout.
    
    This is meant for use in IPython notebooks.
    
    Parameters
    ------------
    width: int
        The pixel width of the SVG canvas
    height: int
        The pixel height of the SVG canvas    
    """
    
    def __init__(self, width=800, height=300):
        self.nodes = []         # json list of node information
        self.links = []         # json list of link information
        self.node_index = {}    # maps from object name to index in self.nodes
        self.width = width      # in pixels
        self.height = height    # in pixels
        
    def to_html(self):    
        """Generates the html for the current graph."""
        graph = json.dumps(dict(nodes=self.nodes, links=self.links))
        nid = 'nengoSVG%d'%id(self)
        return html%dict(graph=graph, id=nid, width=self.width, 
                            height=self.height)
        
    def add_node(self, node):
        """Adds the given nengo node to the graph.

        Parameters
        -----------
        node: an Ensemble, ConstantNode, Template, or Node
        """
        assert node.name not in self.node_index   # don't add it multple times
        
        # build up the information about this node.  This is all the
        # data that will be available to javascript
        info = {}
        info['name'] = node.name 
        info['neurons'] = getattr(node, 'n_neurons', None)
        info['dimensions'] = getattr(node, 'dimensions', None)
        title = ""
        if info['neurons']: 
            title = title + "%d neurons\n"%info['neurons']
        if info['dimensions']: 
            title = title + "%d dimensions\n"%info['dimensions']
        info['title'] = title[:-1]
        
        self.node_index[node.name] = len(self.nodes)  # so we can find it later
        self.nodes.append(info)

    def add_connection(self, connection):
        """Adds the given connection to the graph.
        
        If the pre or post nodes in the connection have not been added, this
        will automatically add them.
        
        Parameters
        ------------
        connection: a SignalConnection
        """
        if connection.pre.name not in self.node_index:
            self.add_node(connection.pre)
        if connection.post.name not in self.node_index:
            self.add_node(connection.post)
    
        source = self.node_index[connection.pre.name]
        target = self.node_index[connection.post.name]
        self.links.append(dict(source=source, target=target))

        
    def add_model(self, model):
        """Adds the given model to the graph.
        
        This iterates through the nodes and their connections.
        
        Parameters
        ------------
        model: a nengo.model.Model
        """
        for obj in model.objs.values():
            self.add_node(obj)
            
        for node in self.nodes:
            obj = model.objs[node['name']]
            for conn in obj.connections_out:
                self.add_connection(conn)
            
# helper functions to embed these displays within IPython
def model2html(model):
    svg = SVG()
    svg.add_model(model)
    return svg.to_html()
def node2html(node):
    svg = SVG(height=50)
    svg.add_node(node)
    return svg.to_html()
def connection2html(connection):
    svg = SVG(height=100)
    svg.add_connection(connection)
    return svg.to_html()

# if we are running IPython, tell it how to display these components
try:
    import IPython
    ip = IPython.get_ipython()
    html_formatter = ip.display_formatter.formatters['text/html']
    html_formatter.for_type_by_name('nengo.model', 'Model', model2html)            
    html_formatter.for_type_by_name('nengo.objects', 'Ensemble', node2html) 
    html_formatter.for_type_by_name('nengo.objects', 'Node', node2html)            
    html_formatter.for_type_by_name('nengo.objects', 'ConstantNode', node2html) 
    html_formatter.for_type_by_name('nengo.connections', 'SignalConnection', connection2html) 
except:
    pass
    
# the base HTML to use.    
html = """
<style>

/* the background of each node */
g.node rect {
  stroke: #fff;
  stroke-width: 1.5px;
  fill: #ccc;
}

/* the text label for each node */
g.node text {
    fill: #000;
    text-anchor: middle;
    alignment-baseline: central;
    font-family: sans-serif;
    font-weight: bold;
    }

/* connections between nodes */
.link {
  stroke: #000;
  stroke-width: 2;
}

</style>
<script src="http://d3js.org/d3.v3.min.js"></script>
<svg id="%(id)s">
<defs>
    // the arrow on the links
    <marker id="TriangleMarker"
      viewBox="0 0 10 10" refX="0" refY="5" 
      markerUnits="strokeWidth"
      markerWidth="6" markerHeight="4"
      orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" />
    </marker>    
    
</svg>
<script>

var width = %(width)d,
    height = %(height)d;

// initialize the force system    
var force = d3.layout.force()
    .charge(-120)
    .linkDistance(60)
    .size([width, height]);

var svg = d3.select("#%(id)s")
    .attr("width", width)
    .attr("height", height)
    
var graph = %(graph)s

// connect the force system to the graph data
force
  .nodes(graph.nodes)
  .links(graph.links)
  .start();

// generate the connections  
var link = svg.selectAll(".link")
  .data(graph.links)
  .enter().append("polyline")
  .attr("class", "link")
  .style("marker-mid", "url(#TriangleMarker)")

// generate the nodes  
var node = svg.selectAll(".node")
  .data(graph.nodes)
  .enter().append("svg:g").attr("class", "node")
  .call(force.drag);

// label the nodes  
var text = node.append("svg:text")
  .text(function(d) { return d.name; })

border_x = 4;
border_y = 2;  

// put the background behind the node labels  
node.insert("svg:rect", ":first-child")
  .attr("x", function(d, i) {return text[0][i].getBBox().x - border_x;})
  .attr("y", function(d, i) {return text[0][i].getBBox().y - border_y;})
  .attr("width", function(d, i) {
                            return text[0][i].getBBox().width + 2*border_x;})
  .attr("height", function(d, i) {
                            return text[0][i].getBBox().height + 2*border_y;});
  
// tool-tip text
node.append("title")
  .text(function(d) { return d.title; });

// update the locations of everything every tick as the force system works  
force.on("tick", function() {
    link.attr("points", function (d) {
                 return ""+d.source.x+","+d.source.y+" "+
                           (d.source.x*0.45+d.target.x*0.55)+","+
                           (d.source.y*0.45+d.target.y*0.55)+" "+
                           d.target.x+","+d.target.y});

    node.attr("transform", function(d) { 
                             return "translate(" + d.x + "," + d.y + ")"; });
    });

</script>
"""