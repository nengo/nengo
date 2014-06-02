//*****************
// Setup the editor
//*****************
//Functions for interaction with editor
var aceRange = ace.require('ace/range').Range;
var editor = null;
var marker = null;
var gui_updating = false;

function removeMarker() {
    if (marker != null) {
        editor.getSession().removeMarker(marker);
        marker = null;
    }
}

function annotateLine(d) { //Called on mouseover in graph
    removeMarker();
    marker = editor.getSession()
        .addMarker(new aceRange(d.line, 0, d.line, 10), 
            'highlight', 'fullLine', true);

    editor.getSession().setAnnotations([{row: d.line, type: 'info'}]);
}

function clearAnnotation(d) { //Called on mouseout in graph
    removeMarker();
    editor.getSession().clearAnnotations();
}

function update_gui_text() {
    gui_updating = true;
    var gui = '\n\nimport nengo_gui\ngui = nengo_gui.Config()\n';
    gui += "gui[model].scale = " + zoom.scale() + "\n";
    gui += "gui[model].offset = " + zoom.translate() + "\n";
    for (var i=0; i<graph.nodes.length; i++) {
        d = graph.nodes[i];
        if ((d.type == 'ens') || (d.type == 'nde')) {
            gui += "gui[" + d.id + "].pos = " + 
                            d.x.toFixed(3) + ", " + d.y.toFixed(3) + "\n";
            gui += "gui[" + d.id + "].scale = " + d.scale.toFixed(3) + "\n";
        }
        if (d.type == "net") {
            gui += "gui[" + d.id + "].pos = " + 
                            d.x.toFixed(3) + ", " + d.y.toFixed(3) + "\n";
            gui += "gui[" + d.id + "].scale = " + d.scale.toFixed(3) + "\n";
            gui += "gui[" + d.id + "].size = " + 
                            d.width.toFixed(3) + ", " + d.height.toFixed(3) + "\n";
        }
    }
    
    text = editor.getValue();
    index = text.indexOf('\n\nimport nengo_gui\n');
    if (index!=-1) {
        text = text.substring(0, index);
    }
    
    new_text = text + gui;
    
    cursor = editor.getCursorPosition();
    scroll_top = editor.session.getScrollTop();
    scroll_left = editor.session.getScrollLeft();
    editor.session.setValue(new_text);
    editor.moveCursorToPosition(cursor);
    editor.session.setScrollTop(scroll_top);
    editor.session.setScrollLeft(scroll_left);
    gui_updating = false;    
}
//*****************
// Helper functions
//*****************

//**************
// Drag and zoom
//**************
function dragstarted(d) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
}

function dragged(d) {
    d3.event.sourceEvent.preventDefault();
    d.x = d3.event.x;
    d.y = d3.event.y;
    dx = d3.event.dx;
    dy = d3.event.dy;
    
    d3.select(this)
        .attr("translate(" + [d.x, d.y] + ")scale(" + d.scale + ")");

    var node_list = graph.nodes.slice(0) //copy the list
    update_node_positions(d, dx, dy, d3.map(node_list)); 
    update_net_position(d, dx, dy);
    update_net_sizes();
    update_line_locations();
    update_gui_text();
}

function dragended(d) {
    d3.select(this).classed("dragging", false);
}

function resizeBRstarted(d) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this.parentElement).classed("resizing", true);
}

function resizeBRdragged(d) {
    a = d3.select(this.parentElement.children[0]);

    a.each(function(d) {
        curWidth = d.width*d.scale  
        ds = (d3.event.dx+curWidth)/curWidth //ratio change
        zoomers[d.id].scale(Math.max(.25, d.scale*ds))
        zoomers[d.id].event(a)
    })
}

function resizeBRended(d) {
    d3.select(this.parentElement).classed("resizing", false);
}

var global_zoom_scale = 1.0;

function zoomed(node) { 
    if (d3.event.sourceEvent !== null && (d3.event.sourceEvent.type != "drag")) {
        try {d3.event.sourceEvent.stopPropagation();}
        catch (e) {if (e instanceof TypeError) {console.log('Zoomed Ignored Error: ' + e)}}
    }
        
    var scale = d3.event.scale;
    var translate = d3.event.translate;
    
    if (typeof node == 'undefined') {
        global_zoom_scale = scale;

        container.attr("transform", function (d) { //scale & translate everything
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
            mouseX = d3.mouse(container[0][0])[0]
            mouseY = d3.mouse(container[0][0])[1]
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
                zoomers[curNode.id].scale(curNode.scale);
            }
        }
                        
        nodes.attr("transform", function (d) { //redraw scale & translate of everything
                return "translate(" + [d.x, d.y] + ")scale(" + d.scale + ")"          
            })
    }


    update_net_sizes();  
    update_line_locations();
    update_text();
    update_gui_text();    
}

function update_text() {
    //could be faster if keep track of whether it was just drawn
    var scales={}
    if (zoom_mode == "geometric") {
        nodes.selectAll('text') //Change the fonts size as a fcn of scale
        .style("font-size", function (d) {
            if (d.type =='net') {
                scales[d.id] = zoomers[d.id].scale() * global_zoom_scale
            } else if (d.contained_by == -1) {//top level ens or nde
                scales[d.id] = global_zoom_scale
            } else { //contained by a network
                scales[d.id] = zoomers[graph.nodes[d.contained_by].id].scale()
                    * global_zoom_scale
            }
            newsize = node_fontsize / scales[d.id]
            return newsize +"px"
        })
        
        nodes.selectAll("g.node.node_ens text, g.node.node_nde text")
            .text(function (d) {
                if (scales[d.id]<.75){
                    return ""
                } else { 
                    return d.label
                }
            })
                    
    } else if (zoom_mode == "semantic") {
        fix_labels(scale);
        
        nodes.selectAll("g.node.node_ens text, g.node.node_nde text")
            .text(function(d) {return d.label;})
    }

    if (zoom_mode == "geometric") {
        //Don't draw node/ens text if scale out far 

    } else {
    }

    nodes.selectAll("g.node.node_net text") //place label under nets
        .text(function (d) {return d.label;})
        .attr('y', function (d) {
            fontSize = parseFloat(d3.select(this).style('font-size'))
            return d.height/2 + fontSize/2 + "px"
        })
}

function zoomCenter(d) { //zoom full screen and center the network clicked on
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

    var width = nengoLayout.center.state.innerWidth;
    var height = nengoLayout.center.state.innerHeight;

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

    zoom.scale(scale)
    
    if (zoomNet == -1) {
        var netX = d3.select('#modelGroup').node().getBBox().x;
        var netY = d3.select('#modelGroup').node().getBBox().y

        zoom.translate([width/2 - (netWidth/2 + netX)*scale,
            height/2 - (netHeight/2 + netY)*scale])            
    } else {
        zoom.translate([width/2 - netX*scale, height/2-netY*scale])
    }

    zoom.event(container.transition().duration(500))
}

/*function parseTranslate(inString) {
    var split = inString.split(",");
    var x = split[0] ? split[0].split("(")[1] : 0;
    var y = split[1] ? split[1].split(")")[0] : 0;
    var s = split[1] ? split[1].split(")")[1].split("(")[1] : null;
    return [x, y, s];
};*/

//***********************
// Drawing graph elements
//***********************
// Move objects to be drawn on top
var net_text_margin = 10;

d3.selection.prototype.moveToFront = function () {
    return this.each(function () {this.parentNode.appendChild(this);});
};

// put nodes on top, lowest level nets 2nd, and so on
function layer_container() {
    //if we had a list of what's 'contained' (i.e top level only) 
    //by model from the server, this would be more efficient
    for (var i in graph.nodes) {
        if (graph.nodes[i].type == "net") {
            layer_network(graph.nodes[i])
        }
    }

    container.selectAll('g.node').filter(function (d) {return d.type != 'net';})
        .moveToFront();
}

//Layer all the items in the network
function layer_network(curNode) {
    if (curNode.type == "net") {
        container.selectAll('g.node').filter(function (d) {return d == curNode;})
            .moveToFront()

        for (var obj in curNode.contains) {
            if (graph.nodes[curNode.contains[obj]].type == "net") {
                layer_network(graph.nodes[curNode.contains[obj]])
            }
        }
    }
}

function update_line_locations() {

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
            if (constant_line_width) {
                return 2/global_zoom_scale +'px';  
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
            if (constant_line_width) {        
                return 2/(global_zoom_scale*graph.nodes[d.source].scale) +'px'; 
            } else {  
                return 2;
            }
        })  
    recurMarker
        .attr('transform', function (d) {
            if (constant_line_width) {
                return "translate("
                + [ -4/(graph.nodes[d.source].scale*global_zoom_scale) + 26, 
                    -4/(graph.nodes[d.source].scale*global_zoom_scale) + 3.3 ]
                 + ")scale(" 
                + 1/(graph.nodes[d.source].scale*global_zoom_scale) + ")"
            } else {
                return "translate(22,-.7)"
            }
        })
}

//Update all network sizes based on node positions
function update_net_sizes() {
    nodes.filter(function (d) {return d.type == 'net';})
        .each(update_net_size)
        .selectAll('.net')
        .attr('x', function (d) {return -d.width / 2;})
        .attr('y', function (d) {return -d.height / 2;})
        .attr('width', function (d) {return d.width;})
        .attr('height', function (d) {return d.height;})
    
    nodes.attr('transform', function (d) {return 'translate(' + [d.x, d.y] 
        + ')scale(' + d.scale + ')';});
 
    if (resizeBR) {
        resizeBR //place the rescale area
            .attr("x", function(d) { return d.width/2 - resizew; })
            .attr("y", function(d) { return d.height/2 - resizew; })
    }
    
    update_text();
    
    if (zoom_mode=='semantic') {
        fix_node_scales(global_zoom_scale);
    }
}

//Update given network size based on node position
function update_net_size(d) {
    var xstart = d.x
    var ystart = d.y
    var x0 = graph.nodes[d.contains[0]].x; //first item in net x,y as a start
    var x1 = x0;
    var y0 = graph.nodes[d.contains[0]].y;
    var y1 = y0;
    var m = net_inner_margin;
    var m2 = net_net_margin*d.scale;
    if (zoom_mode=='semantic') {
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
            if (textWidth/2 > xBorder) {
                xBorder = textWidth*curNode.scale/2
            }
        } else {
            if (textWidth > 20*d.scale && nodeText.textContent !="") {
                xBorder = (textWidth-20)*d.scale/2
            }        
        }
        
        x0 = Math.min(curNode.x - xBorder, x0);
        x1 = Math.max(curNode.x + xBorder, x1);
        y0 = Math.min(curNode.y - yBorder, y0);
        y1 = Math.max(curNode.y + yBorder, y1);
    }
    d.x = (x0 + x1) / 2; // x, y mid
    d.y = (y0 + y1) / 2;

    d.width = (x1 - x0)/d.scale + 2 * m; //track heights/widths
    d.height = (y1 - y0)/d.scale + 2 * m;

    if (xstart!=undefined && ystart!=undefined) {    
    
        dx = d.x - xstart;
        dy = d.y - ystart;
        
        var node_list = graph.nodes.slice(0)
        update_node_positions(d, 2 * dx, 2 * dy, d3.map(node_list))
    }
}

//Move all the nodes in a network if network position changes
function update_net_position(d, dx, dy) {
    if (d.type == "net") {
        for (var obj in d.full_contains) {
            graph.nodes[d.full_contains[obj]].x += dx
            graph.nodes[d.full_contains[obj]].y += dy
        }
    }
}

//Update the position of any nodes and what they affect
function update_node_positions(d, dx, dy, node_list) { 
    removeValue(node_list, d)
    if (d.type == 'net') { //stop all sub items from moving
        for (var i in d.full_contains) {
            node_list.remove(d.full_contains[i])
        }
    }
    for (var n in node_list.keys()) {
        var curNode = node_list.get(node_list.keys()[n])
        if (close_to(curNode, d)) {//if curNode is close to d
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
            move_node(curNode, dx, dy)
            update_node_positions(curNode, dx, dy, d3.map(node_list))
        }
    }
}

//Move the provided nodes the provided distance and note that
function move_node(node, dx, dy) {
    if (node.type == "net") { //move a network
        update_net_position(node, dx, dy)
    } else { //move ens or nde
        node.x += dx
        node.y += dy
    }
}

//Redraw if the window is resized
function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    svg.attr("width", width).attr("height", height);
}

//**************
// Miscellaneous
//**************
//Remove object from a map.  Removes all matching items.
function removeValue(map, d) {
    keys = map.keys()
    for (el in map.keys()) {
        if (map.get(map.keys()[el]) == d) {
            map.remove(map.keys()[el])
        }
    }
}

//Check if node, n is close to origin object, o
function close_to(n, o) { //n is node, o is origin
    netm = net_margin;
    nodem = node_margin;
    ns = n.scale;
    os = o.scale;
    if (zoom_mode=="semantic") {
        netm = netm / global_zoom_scale;
        nodem = nodem / global_zoom_scale;
    }
    if (o.type == "net") { //if origin is net
        if (!(n.type == "net")) { //if node is nde or ens
            if (!netContains(n, o)) {
                if (Math.abs(o.x - n.x) < (netm + o.width / 2)*os &&
                    Math.abs(o.y - n.y) < (netm + o.height / 2)*os) {
                    return true;
                }
            }
        } else if (!(netContains(n, o) || netContains(o, n))) { //if node is net
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
            if (!netContains(o, n)) {
                if (Math.abs(o.x - n.x) < (netm + n.width*ns / 2) &&
                    Math.abs(o.y - n.y) < (netm + n.height*ns / 2)) {
                    return true;
                }
            }
        }
    }
    return false;
}

//True if net or any of its subnets contains node
function netContains(node, net) {
    ind = graph.nodes.indexOf(node)
    if (net.full_contains.indexOf(ind) > -1) {
        return true
    } else {
        return false
    }
}

//Comparing full_contains length if it exists, for sorting
function containsCompare(a,b) {
    if (a.type!='net') {
        return 1;
    } else if (b.type != 'net') {
        return -1;
    } else if (b.type!='net' && a.type!='net') {
        return 0;
    } else {
        return b.full_contains.length-a.full_contains.length;
    }
}

//*****************
// Reload the graph
//*****************
var graph = null;
var link = null;
var linkRecur = null;
var recurMarker = null;
var node = null;
var resizeBR = null;
var zoomers = {};

var constant_line_width = true; //keep the lines the same absolute size on zoom
var node_margin = 35;
var net_inner_margin = 40;
var net_margin = 15;
var net_net_margin = 10;  // spacing between network and subnetwork
var node_fontsize = 16;
var resizew = 15;  //width and height of resize region in bottom right corner of net 

var waiting_for_result = false;
var pending_change = false;

function reload_graph_data() {
    // don't send a new request while we're still waiting for another one
    if (waiting_for_result) {
        pending_change = true;
        return;
    }
    
    waiting_for_result = true;
    
    var data = new FormData();
    data.append('code', editor.getValue());

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/graph.json', true);
    xhr.onload = update_graph;
    xhr.send(data);
}

//Redraw the graph given server response
function update_graph() {
    waiting_for_result = false;
    
    if (pending_change) {
        pending_change = false;
        reload_graph_data();
    }
    
	graph = JSON.parse(this.responseText);

    // was there a parsing error?
    if (graph.error_line != undefined) {
        removeMarker();
        marker = editor.getSession()
            .addMarker(new aceRange(graph.error_line - 1, 0, graph.error_line - 1, 10), 
            'highlight', 'fullLine', true);
        editor.getSession().setAnnotations([{
            row: graph.error_line - 1,
            type: 'error',
            text: graph.text,
        }]);
        return;
    } else {
        if (marker != null) {
            editor.getSession().removeMarker(marker);
            marker = null;
        }
        editor.getSession().clearAnnotations();
    }
    
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
    links = container.selectAll('.link.link_std, .link.link_net')
        .data(nonrecurlink, function (d) {return d.id})
    links.enter().append('polyline')
        .attr('class', function (d) {return 'link link_' + d.type;})

    linkRecur = container.selectAll('.link.link_rec')
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



    //get all the nodes, for updating
    nodes = container.selectAll('g.node')
        .data(graph.nodes, function (d) {return d.id})
    container.selectAll('g.node text')
        .data(graph.nodes, function (d) {return d.id})

    //Create html objects to draw
    var nodeEnter = nodes
        .enter()
        .append('g')
        .attr('class', function (d) {return 'node node_' + d.type;})
        .attr('cursor', 'pointer')
        .on('mouseover', annotateLine)
        .on('mouseout', clearAnnotation)
        .on('dblclick.zoom', zoomCenter)
        .call(drag);  

    nodeEnter.filter(function (d) {return d.type == 'net';})
        .append('rect')
        .attr('class', 'net')
        .attr('x', function (d) {return d.x})
        .attr('y', function (d) {return d.y})
        .attr('rx', '15')
        .attr('ry', '15')
        .attr('width', function (d) {return d.width})
        .attr('height', function (d) {return d.height})
        .each(function (d) {
            zoomers[d.id] = d3.behavior.zoom()
                .scaleExtent([.05, 4])
                .on('zoom', zoomed)
            zoomers[d.id](d3.select(this))
            d.scale = graph.nodes[d.contains[0]].scale
            zoomers[d.id].scale(d.scale)
        })
        .on('dblclick.zoom', zoomCenter)
                              
    nodeEnter.filter(function (d) {return d.type == 'ens';})
        .append('use')
        .attr('xlink:href', "#ensemble")
        .each(function (d) {
            if (d.contained_by > -1) {
                id = graph.nodes[d.contained_by].id
                zoomers[id](d3.select(this))
            }
        })
        .on('dblclick.zoom', zoomCenter)

    nodeEnter.filter(function (d) {return d.type == 'nde';})
        .append('circle')
        .attr('r', '20')
        .each(function (d) {
            if (d.contained_by > -1) {
                id = graph.nodes[d.contained_by].id
                zoomers[id](d3.select(this))
            }
        })  
        .on('dblclick.zoom', zoomCenter)

    nodeEnter.append('text')     //label everything
        .text(function (d) {return d.label})

    nodeEnter.selectAll('.node_nde text, .node_ens text')
        .attr('y', '30')
        .style('font-size', node_fontsize)

    nodes.exit().remove();
    links.exit().remove();
    linkRecur.exit().remove();
    
    // go to the stored gui location
    var newLayout = false;
    if (graph.global_scale == null) {
        zoom.scale(1);
        newLayout = true;
    } else {
        zoom.scale(graph.global_scale);
    }
    zoom.translate(graph.global_offset);
    zoom.event(container.transition().duration(500)); //or event on svg?
    
    //redraw so nodes are on top, lowest level nets 2nd, and so on
    layer_container();
    update_net_sizes();
    
    resizeBR = nodeEnter.filter(function (d) {return d.type == 'net';})
        .append('rect') //bottom right drag region
        .attr("x", function(d) { return d.width/2 - resizew; })
        .attr("y", function(d) { return d.height/2 - resizew; })
        .attr("height", resizew)
        .attr("id", "resizeBR")
        .attr("width", resizew)
        .attr("cursor", "se-resize")
        .call(resizeBotR);
    
    update_line_locations();
    update_text();
    resize();
    if (newLayout) {
        zoomCenter();
    }
}

//***********
//Main script
//***********
$(document).ready(function () {
    zoom = d3.behavior.zoom()
        .scaleExtent([.00005, 10])
        .on('zoom', zoomed);

    drag = d3.behavior.drag()
        .origin(function (d) {return d})
        .on('dragstart', dragstarted)
        .on('drag', dragged)
        .on('dragend', dragended);

    resizeBotR = d3.behavior.drag()
        .origin(function (d) {return d})
        .on('dragstart', resizeBRstarted)
        .on('drag', resizeBRdragged)
        .on('dragend', resizeBRended);

    //initialize editor
    editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
    editor.getSession().setUseSoftTabs(true);
    editor.getSession().setMode("ace/mode/python");
    editor.on('change', function(event) {
        $('#menu_save').removeClass('disable');
        if (!gui_updating) reload_graph_data();
    });

    //initialize graph
    svg = d3.select("svg");
    container = svg.append('g').attr('id','modelGroup');
    svg.call(zoom).on('dblclick.zoom', zoomCenter); // set up zooming on the graph
    d3.select(window).on("resize", resize);
    
    //setup the window panes, and manipulations
    nengoLayout = $('body').layout({ 
	    north__slidable:			false,	
		north__resizsable:			false,	
		north__spacing_open:        0,
		north__size:                55, //pixels
		east__livePaneResizing:		true,
		east__size:					.4,
		east__minSize:				.1,
		east__maxSize:				.8 // 80% of layout width        
    });
    
    //start this puppy up
    reload_graph_data();
});
