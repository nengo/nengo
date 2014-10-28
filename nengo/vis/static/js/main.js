var ModelVis = {
//*****************
// Helper functions
//*****************

//**************
// Drag and zoom
//**************
dragstarted: function(d) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
},

dragged: function(mv, d) {
    d3.event.sourceEvent.preventDefault();
    d.x = d3.event.x;
    d.y = d3.event.y;
    dx = d3.event.dx;
    dy = d3.event.dy;
    
    d3.select(this)
        .attr("translate(" + [d.x, d.y] + ")scale(" + d.scale + ")");

    var node_list = graph.nodes.slice(0) //copy the list
    mv.update_node_positions(d, dx, dy, d3.map(node_list)); 
    mv.update_net_position(d, dx, dy);
    mv.update_net_sizes();
    mv.update_line_locations();
},

dragended: function(d) {
    d3.select(this).classed("dragging", false);
},

resizeBRstarted: function(d) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this.parentElement).classed("resizing", true);
},

resizeBRdragged: function(d) {
    a = d3.select(this.parentElement.children[0]);

    a.each(function(d) {
        curWidth = d.width*d.scale  
        ds = (d3.event.dx+curWidth)/curWidth //ratio change
        this.zoomers[d.id].scale(Math.max(.25, d.scale*ds))
        this.zoomers[d.id].event(a)
    })
},

resizeBRended: function(d) {
    d3.select(this.parentElement).classed("resizing", false);
},

//global_zoom_scale: 1.0,

zoomed: function(mv, node) {
    if (d3.event.sourceEvent !== null && (d3.event.sourceEvent.type != "drag")) {
        try {d3.event.sourceEvent.stopPropagation();}
        catch (e) {if (e instanceof TypeError) {console.log('Zoomed Ignored Error: ' + e)}}
    }
        
    var scale = d3.event.scale;
    var translate = d3.event.translate;
    
    if (typeof node == 'undefined') {
        global_zoom_scale = scale;

        mv.container.attr("transform", function (d) { //scale & translate everything
            return "translate(" + translate + ")scale(" + scale + ")"
        })        
    } else {
        scale = scale/node.scale;  //determine the scaling ratio
        
        if (node.type != 'net') {  //if you're on a ens scale the containing net
            node = graph.nodes[node.contained_by]
        }
        
        if (d3.select(this.parentElement).classed('resizing')) { //drag zoom
            //resize from top left corner (bottom right drag)
            mouseX = node.x - (node.width/2)*node.scale
            mouseY = node.y - (node.height/2)*node.scale
        } else {
            mouseX = d3.mouse(mv.container[0][0])[0]
            mouseY = d3.mouse(mv.container[0][0])[1]
        }
        node.scale *= scale;  //scale and translate this net
        node.x = scale*(node.x-mouseX) + mouseX; //translation with scaling
        node.y =  scale*(node.y-mouseY) + mouseY;
        for (i in node.full_contains) {  //scale everything it contains
            curNode = graph.nodes[node.full_contains[i]]
            curNode.scale *= scale;
            curNode.x = scale*(curNode.x-mouseX) + mouseX;
            curNode.y =  scale*(curNode.y-mouseY) + mouseY;
            if (curNode.type == 'net') { //update contained zoomers 
                mv.zoomers[curNode.id].scale(curNode.scale);
            }
        }
                        
        nodes.attr("transform", function (d) { //redraw scale & translate of everything
                return "translate(" + [d.x, d.y] + ")scale(" + d.scale + ")"          
            })
    }


    mv.update_net_sizes();  
    mv.update_line_locations();
    mv.update_text();
},

update_text: function() {
    //could be faster if keep track of whether it was just drawn
    var mv = this;
    var scales={}
    if (this.zoom_mode == "geometric") {
        nodes.selectAll('text') //Change the fonts size as a fcn of scale
        .style("font-size", function (d) {
            if (d.type =='net') {
                scales[d.id] = mv.zoomers[d.id].scale() * mv.global_zoom_scale
            } else if (d.contained_by == -1) {//top level ens or nde
                scales[d.id] = mv.global_zoom_scale
            } else { //contained by a network
                scales[d.id] = mv.zoomers[graph.nodes[d.contained_by].id].scale()
                    * mv.global_zoom_scale
            }
            newsize = mv.node_fontsize / scales[d.id]
            return newsize +"px"
        })
        
        nodes.selectAll("g.node.node_ens text, g.node.node_nde text")
            .text(function (d) {
                //if (scales[d.id]<.75){
                    //return ""
                //} else { 
                    return d.label
                //}
            })
                    
    } else if (this.zoom_mode == "semantic") {
        fix_labels(scale);
        
        nodes.selectAll("g.node.node_ens text, g.node.node_nde text")
            .text(function(d) {return d.label;})
    }

    if (this.zoom_mode == "geometric") {
        //Don't draw node/ens text if scale out far 

    } else {
    }

    nodes.selectAll("g.node.node_net text") //place label under nets
        .text(function (d) {return d.label;})
        .attr('y', function (d) {
            fontSize = parseFloat(d3.select(this).style('font-size'))
            return d.height/2 + fontSize/2 + "px"
        })
},

zoomCenter: function(d) { //zoom full screen and center the network clicked on
    var zoomNet = d
    
    if (d3.event !== null) {
        try {d3.event.stopPropagation();}
        catch (e) {if (e instanceof TypeError) {
            console.log('ZoomCenter Ignored Error: ' + e)
        }}
    }
    
    if (d == undefined) { //background click
        zoomNet = -1
    } else if (d.type !== 'net') { //if node or ens
         if (d.contained_by == -1) { //background click
            zoomNet = -1 
         } else { //use containing network
            zoomNet = graph.nodes[d.contained_by]
        }
    }

    var width=400, height=400;
    //var width = nengoLayout.center.state.innerWidth;
    //var height = nengoLayout.center.state.innerHeight;

    if (zoomNet == -1) { //zoom out to full model
        var netWidth = d3.select('#modelGroup').node()
            .getBBox().width;
        var netHeight = d3.select('#modelGroup').node()
            .getBBox().height;

    } else { //zoom to fit zoomNet
        var netWidth = zoomNet.width*zoomNet.scale
        var netHeight = zoomNet.height*zoomNet.scale
        var netX = zoomNet.x
        var netY = zoomNet.y
    }
    
    if (width/height >= netWidth/netHeight) {
        //zoom to height
        scale = .9*height/netHeight
    } else {
        //zoom to width
        scale = .9*width/netWidth
    }

    console.log(scale);
    if (isNaN(scale)) {
        scale = 1.0;
    } 
    
    if (scale == Infinity) {
        scale = 1.0;
    }

    this.zoom.scale(scale)
    
    if (zoomNet == -1) {
        var netX = d3.select('#modelGroup').node().getBBox().x;
        var netY = d3.select('#modelGroup').node().getBBox().y

        this.zoom.translate([width/2 - (netWidth/2 + netX)*scale,
            height/2 - (netHeight/2 + netY)*scale])            
    } else {
        this.zoom.translate([width/2 - netX*scale, height/2-netY*scale])
    }

    this.zoom.event(this.container.transition().duration(500))
},

//[>function parseTranslate(inString) {
    //var split = inString.split(",");
    //var x = split[0] ? split[0].split("(")[1] : 0;
    //var y = split[1] ? split[1].split(")")[0] : 0;
    //var s = split[1] ? split[1].split(")")[1].split("(")[1] : null;
    //return [x, y, s];
//};*/

//***********************
// Drawing graph elements
//***********************
// Move objects to be drawn on top
// put nodes on top, lowest level nets 2nd, and so on
layer_container: function() {
    //if we had a list of what's 'contained' (i.e top level only) 
    //by model from the server, this would be more efficient
    for (var i in graph.nodes) {
        if (graph.nodes[i].type == "net") {
            this.layer_network(graph.nodes[i])
        }
    }

    this.container.selectAll('g.node').filter(function (d) {return d.type != 'net';})
        .moveToFront();
},

//Layer all the items in the network
layer_network: function(curNode) {
    if (curNode.type == "net") {
        this.container.selectAll('g.node').filter(function (d) {return d == curNode;})
            .moveToFront()

        for (var obj in curNode.contains) {
            if (graph.nodes[curNode.contains[obj]].type == "net") {
                this.layer_network(graph.nodes[curNode.contains[obj]])
            }
        }
    }
},

update_line_locations: function() {
    var mv = this;

    links.filter(function (d) {return d.type == 'std';})
        .attr('points', function (d) {        
            x0 = graph.nodes[d.source].x;
            y0 = graph.nodes[d.source].y;
            x1 = graph.nodes[d.target].x;
            y1 = graph.nodes[d.target].y;
            return "" + x0 + "," + y0 + " " + 
                (x0 * 0.45 + x1 * 0.55) + "," + 
                (y0 * 0.45 + y1 * 0.55) + " " +
                x1 + "," + y1;
        })
        .attr('stroke-width', function(d) {
            if (mv.constant_line_width) {
                return 2/mv.global_zoom_scale +'px';  
            } else {
                return 2;
            }
        });
        
    linkRecur
        .attr('x', function (d) {return graph.nodes[d.source].x-20
            * graph.nodes[d.source].scale})
        .attr('y', function (d) {return graph.nodes[d.source].y-34
            * graph.nodes[d.source].scale})
        .attr('width', function (d) {return graph.nodes[d.source].scale*100})
        .select('use')
        .attr('stroke-width', function(d) {
            if (mv.constant_line_width) {        
                return 2/(mv.global_zoom_scale*graph.nodes[d.source].scale) +'px'; 
            } else {  
                return 2;
            }
        })  
    recurMarker
        .attr('transform', function (d) {
            if (mv.constant_line_width) {
                return "translate("
                + [ -4/(graph.nodes[d.source].scale*mv.global_zoom_scale) + 26, 
                    -4/(graph.nodes[d.source].scale*mv.global_zoom_scale) + 3.3 ]
                 + ")scale(" 
                + 1/(graph.nodes[d.source].scale*mv.global_zoom_scale) + ")"
            } else {
                return "translate(22,-.7)"
            }
        })
},

//Update all network sizes based on node positions
update_net_sizes: function() {
    nodes.filter(function (d) {return d.type == 'net';})
        .each(this.update_net_size.bind(this))
        .selectAll('.net')
        .attr('x', function (d) {return -d.width / 2;})
        .attr('y', function (d) {return -d.height / 2;})
        .attr('width', function (d) {return d.width;})
        .attr('height', function (d) {return d.height;})
    
    nodes.attr('transform', function (d) {return 'translate(' + [d.x, d.y] 
        + ')scale(' + d.scale + ')';});
 
    if (this.resizeBR) {
        this.resizeBR //place the rescale area
            .attr("x", function(d) { return d.width/2 - resizew; })
            .attr("y", function(d) { return d.height/2 - resizew; })
    }
    
    this.update_text();
    
    if (this.zoom_mode=='semantic') {
        fix_node_scales(global_zoom_scale);
    }
},

//Update given network size based on node position
update_net_size: function(d) {
    if (d.contains.length == 0) {return;} //empty network
    var xstart = d.x
    var ystart = d.y
    var x0 = graph.nodes[d.contains[0]].x; //first item in net x,y as a start
    var x1 = x0;
    var y0 = graph.nodes[d.contains[0]].y;
    var y1 = y0;
    var m = this.net_inner_margin;
    var m2 = this.net_net_margin*d.scale;
    if (this.zoom_mode=='semantic') {
        m = m / global_zoom_scale;
        m2 = m2 / global_zoom_scale;
    }
    for (var obj in d.contains) { //min/max of y and x of nodes in net
        var xBorder = 0
        var yBorder = 0
        var curNode = graph.nodes[d.contains[obj]]

        var nodeText = nodes.selectAll('text') //to take text into account
            .filter(function (d) {return d.id==curNode.id})
            .node()

        var textWidth = nodeText.getBBox()
            .width 

        if (curNode.type == "net") {
            xBorder = (curNode.width / 2)*curNode.scale
            yBorder = (curNode.height / 2)*curNode.scale
            if (isNaN(xBorder) || isNaN(yBorder)) {
                xBorder = 0;
                yBorder = 0;
            } //happens on load
            if (textWidth/2*curNode.scale > xBorder) {
                xBorder = textWidth*curNode.scale/2
            }
            
        } else {
            if (textWidth > 20*d.scale && nodeText.textContent !="") {
                xBorder = (textWidth-20)*d.scale/2
            }        
        }
        
        if (nodeText.textContent !="") { //Adjust heights and center
            yBorder += (this.node_fontsize)*curNode.scale
        }
        
        x0 = Math.min(curNode.x - xBorder, x0);
        x1 = Math.max(curNode.x + xBorder, x1);
        y0 = Math.min(curNode.y - yBorder, y0);
        y1 = Math.max(curNode.y + yBorder, y1);
    }
    d.x = (x0 + x1) / 2; // x, y mid
    d.y = (y0 + y1) / 2; // + textHeight;

    d.width = (x1 - x0)/d.scale + 2 * m; //track heights/widths
    d.height = (y1 - y0)/d.scale + 2 * m;
    

    if (xstart!=undefined && ystart!=undefined) {    
    
        dx = d.x - xstart;
        dy = d.y - ystart;
        
        var node_list = graph.nodes.slice(0)
        this.update_node_positions(d, 2 * dx, 2 * dy, d3.map(node_list))
    }
},

//Move all the nodes in a network if network position changes
update_net_position: function(d, dx, dy) {
    if (d.type == "net") {
        for (var obj in d.full_contains) {
            graph.nodes[d.full_contains[obj]].x += dx
            graph.nodes[d.full_contains[obj]].y += dy
        }
    }
},

//Update the position of any nodes and what they affect
update_node_positions: function(d, dx, dy, node_list) {
    this.removeValue(node_list, d)
    if (d.type == 'net') { //stop all sub items from moving
        for (var i in d.full_contains) {
            node_list.remove(d.full_contains[i])
        }
    }
    for (var n in node_list.keys()) {
        var curNode = node_list.get(node_list.keys()[n])
        if (this.close_to(curNode, d)) {//if curNode is close to d
            if (d3.event != null && d3.event.type == "zoom") { //figure out which way to move things on zoom bump
                if (d3.event.sourceEvent !== null && d3.event.sourceEvent.type=="wheel") {
                    del = d3.event.sourceEvent.wheelDelta/3;
                    if (curNode.x < d.x) {
                        dx = -del;
                    } else {
                        dx = del;
                    }
                    if (curNode.y < d.y) {
                        dy = -del;
                    } else {
                        dy = del;
                    }
                }
            }
            this.move_node(curNode, dx, dy)
            this.update_node_positions(curNode, dx, dy, d3.map(node_list))
        }
    }
},

//Move the provided nodes the provided distance and note that
move_node: function(node, dx, dy) {
    if (node.type == "net") { //move a network
        this.update_net_position(node, dx, dy)
    } else { //move ens or nde
        node.x += dx
        node.y += dy
    }
},

//Redraw if the window is resized
resize: function() {
    width = window.innerWidth;
    height = window.innerHeight;
    svg.attr("width", width).attr("height", height);
},

//****************
//// Miscellaneous
//****************
//Remove object from a map.  Removes all matching items.
removeValue: function(map, d) {
    keys = map.keys()
    for (el in map.keys()) {
        if (map.get(map.keys()[el]) == d) {
            map.remove(map.keys()[el])
        }
    }
},

//Check if node, n is close to origin object, o
close_to: function(n, o) { //n is node, o is origin
    netm = this.net_margin;
    nodem = this.node_margin;
    ns = n.scale;
    os = o.scale;
    if (this.zoom_mode=="semantic") {
        netm = netm / global_zoom_scale;
        nodem = nodem / global_zoom_scale;
    }
    if (o.type == "net") { //if origin is net
        if (!(n.type == "net")) { //if node is nde or ens
            if (!this.netContains(n, o)) {
                if (Math.abs(o.x - n.x) < (netm + o.width / 2)*os &&
                    Math.abs(o.y - n.y) < (netm + o.height / 2)*os) {
                    return true;
                }
            }
        } else if (!(this.netContains(n, o) || this.netContains(o, n))) { //if node is net
            if (Math.abs(o.x - n.x) < (n.width*ns / 2 
                + o.width*os / 2) && Math.abs(o.y - n.y) < 
                (n.height*ns / 2 + o.height*os / 2)) {
                return true;
            }
        }
    } else { //if origin is nde or ens
        if (!(n.type == "net")) { //if node nde or ens
            if (Math.abs(o.x - n.x) < nodem && Math.abs(o.y - n.y) < nodem) {
                return true;
            }
        } else { //if node is net
            if (!this.netContains(o, n)) {
                if (Math.abs(o.x - n.x) < (netm + n.width*ns / 2) &&
                    Math.abs(o.y - n.y) < (netm + n.height*ns / 2)) {
                    return true;
                }
            }
        }
    }
    return false;
},

//True if net or any of its subnets contains node
netContains: function(node, net) {
    ind = graph.nodes.indexOf(node)
    if (net.full_contains.indexOf(ind) > -1) {
        return true
    } else {
        return false
    }
},

//Comparing full_contains length if it exists, for sorting
containsCompare: function(a,b) {
    if (a.type!='net') {
        return 1;
    } else if (b.type != 'net') {
        return -1;
    } else if (b.type!='net' && a.type!='net') {
        return 0;
    } else {
        return b.full_contains.length-a.full_contains.length;
    }
},

//******************
// Reload the graph
//******************
graph: null,
link: null,
linkRecur: null,
recurMarker: null,
node: null,
resizeBR: null,
zoomers: {},

constant_line_width: true, //keep the lines the same absolute size on zoom
node_margin: 35,
net_inner_margin: 40,
net_margin: 15,
net_net_margin: 10,  // spacing between network and subnetwork
node_fontsize: 16,
resizew: 15,  //width and height of resize region in bottom right corner of net 

newLayout: false, //check if this is a new layout

//Redraw the graph given server response
update_graph: function(graph2) {
    graph = graph2

    //separate links into recurrent and nonrecurrent ?move to converter?  
    var nonrecurlink = []
    var recurlink = []
    for (i in graph.links) {
        if (graph.links[i].target != graph.links[i].source) {
            nonrecurlink.push(graph.links[i])
        } else {
            recurlink.push(graph.links[i])
        }
    }

    //update the links
    links = this.container.selectAll('.link.link_std, .link.link_net')
        .data(nonrecurlink, function (d) {return d.id})
    links.enter().append('polyline')
        .attr('class', function (d) {return 'link link_' + d.type;})

    linkRecur = this.container.selectAll('.link.link_rec')
        .data(recurlink, function (d) {return d.id})
    linkRecur.enter().append('svg')       
        .attr('class', function (d) {return 'link link_' + d.type;})
        .attr("viewBox", "-2 -2 100 100")
        .attr("preserveAspectRatio", "xMinYMin meet")
        .attr('width', '100')
        .append('use')
        .attr('xlink:href', "#recur")
    recurMarker = linkRecur
        .append('use')
        .attr('xlink:href', "#recurTriangle")

    this.container.selectAll('g.node_net').remove()

    //get all the nodes, for updating
    nodes = this.container.selectAll('g.node')
        .data(graph.nodes, function (d) {return d.id})
    this.container.selectAll('g.node text')
        .data(graph.nodes, function (d) {return d.id})

    //Create html objects to draw
    var nodeEnter = nodes
        .enter()
        .append('g')
        .attr('class', function (d) {return 'node node_' + d.type;})
        .attr('cursor', 'pointer')
        .on('dblclick.zoom', this.zoomCenter)
        .call(this.drag);

    var mv = this;

    nodeEnter.filter(function (d) {return d.type == 'net';})
        .append('rect')
        .attr('class', 'net')
        .attr('x', function (d) {return -d.width / 2;})
        .attr('y', function (d) {return -d.height / 2;})
        .attr('width', function (d) {return d.width;})
        .attr('height', function (d) {return d.height;})
        .attr('rx', '15')
        .attr('ry', '15')
        .each(function (d) {
            mv.zoomers[d.id] = d3.behavior.zoom()
                .scaleExtent([.05, 4])
                .on('zoom', function(d) { mv.zoomed.bind(this)(mv, d); })
            mv.zoomers[d.id](d3.select(this))
            //d.scale = graph.nodes[d.contains[0]].scale
            mv.zoomers[d.id].scale(d.scale)
        })
        .on('dblclick.zoom', this.zoomCenter);
                              
    nodeEnter.filter(function (d) {return d.type == 'ens';})
        .append('use')
        .attr('xlink:href', "#ensemble")
        .each(function (d) {
            if (d.contained_by > -1) {
                id = graph.nodes[d.contained_by].id
                mv.zoomers[id](d3.select(this))
            }
        })
        .on('dblclick.zoom', this.zoomCenter);

    nodeEnter.filter(function (d) {return d.type == 'nde';})
        .append('circle')
        .attr('r', '20')
        .each(function (d) {
            if (d.contained_by > -1) {
                id = graph.nodes[d.contained_by].id
                mv.zoomers[id](d3.select(this))
            }
        })  
        .on('dblclick.zoom', this.zoomCenter);

    nodeEnter.attr('transform', function (d) {return 'translate(' + [d.x, d.y] 
        + ')scale(' + d.scale + ')';});

    nodeEnter.append('text')     //label everything
        .text(function (d) {return d.label});

    nodeEnter.selectAll('.node_nde text, .node_ens text')
        .attr('y', '30')
        .style('font-size', this.node_fontsize+"px");

/*    nodeEnter.selectAll('text')
        //.each(function (d) {
            //if (d.contained_by > -1) {
                //id = graph.nodes[d.contained_by].id
                //zoomers[id](d3.select(this))
            //}
        //})   */     

    nodes.exit().remove();
    links.exit().remove();
    linkRecur.exit().remove();

    if (graph.global_scale == 1 && graph.global_offset.toString() == [0, 0].toString()) {
        newLayout = true;
    }

    this.zoom.scale(graph.global_scale);     // go to the stored gui location
    this.zoom.translate(graph.global_offset);
    this.zoom.event(this.container.transition().duration(500)); //or event on svg?

    //redraw so nodes are on top, lowest level nets 2nd, and so on
    this.layer_container();
    this.update_net_sizes();

    this.update_line_locations();
    this.update_text();

    this.resize();
    if (newLayout) {
        this.zoomCenter();
        this.newLayout = false;
    }
},

zoom_mode: 'geometric',
//net_text_margin: 10,

zoom: null,

drag: null,

resizeBotR: d3.behavior.drag()
    .origin(function (d) {return d})
    .on('dragstart', this.resizeBRstarted)
    .on('drag', this.resizeBRdragged)
    .on('dragend', this.resizeBRended),

container: null,

//***********
//Main script
//***********
common_init: function() {
    var mv = this;

    d3.selection.prototype.moveToFront = function () {
        return this.each(function () {this.parentNode.appendChild(this);});
    };

    this.zoom = d3.behavior.zoom()
        .scaleExtent([.00005, 10])
        .on('zoom', function(d) { mv.zoomed.bind(this)(mv, d); });

    this.drag = d3.behavior.drag()
        .origin(function (d) {return d})
        .on('dragstart', this.dragstarted)
        .on('drag', function(d) { mv.dragged.bind(this)(mv, d); })
        .on('dragend', this.dragended);

    //initialize graph
    svg = d3.select("svg");
    this.container = svg.append('g').attr('id','modelGroup');
    svg.call(this.zoom).on('dblclick.zoom', this.zoomCenter); // set up zooming on the graph
    d3.select(window).on("resize", this.resize);
},

init: function(gdata) {
    this.common_init();
    //start this puppy up
    this.update_graph(gdata);
},
};
