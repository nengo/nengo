//*****************
// Setup the editor
//*****************
//Functions for interaction with editor
var aceRange = ace.require('ace/range').Range;
var editor = null;
var marker = null;

function removeMarker() {
    if (marker!=null) {
        editor.getSession().removeMarker(marker);
        marker = null;
        }
}

function annotateLine(d) { //Called on mouseover in graph
	removeMarker();
	marker = editor.getSession().addMarker(new aceRange(d.line, 0, d.line, 10), 'highlight', 'fullLine', true);
	
	editor.getSession().setAnnotations([{row: d.line, type:'info'}]);
}
	
function clearAnnotation(d) { //Called on mouseout in graph
	removeMarker();
	editor.getSession().clearAnnotations();
}

//***********************
// Setup the file browser
//***********************
//Load the browser and hide it

function open_file(file) {
    $('#filebrowser').hide();

    container.selectAll('.link').remove();
    container.selectAll('.node').remove();
    editor.setValue('');

    var data = new FormData();
    data.append('filename', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/openfile', true);
    xhr.onload = function(event) {editor.setValue(this.responseText);};
    xhr.send(data);
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
    d.x = d3.event.x;
    d.y = d3.event.y;

    d3.select(this).attr('transform', "translate("+d.x+","+d.y+")");

    var node_list = d3.map(graph.nodes) //create a map of the nodes
    update_node_positions(d, d3.event.dx, d3.event.dy, node_list);
    update_net_position(d, d3.event.dx, d3.event.dy);
    update_net_sizes();
    update_line_locations();
}

function dragended(d) {
      d3.select(this).classed("dragging", false);
}

function zoomed() {
    scale = d3.event.scale
    container.attr("transform", "translate(" + //scale everything
        d3.event.translate + ")scale(" + scale + ")");
        
    node.selectAll('text') //Change the fonts size as a fcn of scale
        .style("font-size", function(d) {
            newsize = node_fontsize / scale
            if (newsize>node_fontsize) {
                return newsize + "px";
            } else {
                return node_fontsize + "px";
            }
        })
    
    //could be faster if keep track of whether it was just drawn
    if (scale<.75) { //Don't draw node/ens text if scale out far 
        node.selectAll("g.node.node_ens text, g.node.node_nde text")
            .text("")
        }
    else {
        node.selectAll("g.node.node_ens text, g.node.node_nde text")
		.text(function(d) {return d.label;});
    }
            
    update_net_text();
}

//***********************
// Drawing graph elements
//***********************
// Move objects to be drawn on top
var net_widths = {};
var net_heights = {};
var net_text_margin = 10;

d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

function update_line_locations() {
    link.filter(function(d) {return d.type=='std';})
        .attr('points', function(d) {
            x0 = graph.nodes[d.source].x;
            y0 = graph.nodes[d.source].y;
            x1 = graph.nodes[d.target].x;
            y1 = graph.nodes[d.target].y;
            return ""+x0+","+y0+" "+(x0*0.45+x1*0.55)+","+(y0*0.45+y1*0.55)+" "+
                      x1+","+y1;
        });
        
    linkRecur
        .attr('x', function(d) {return graph.nodes[d.source].x})
        .attr('y', function(d) {return graph.nodes[d.source].y})
}

//Update all network sizes based on node positions
function update_net_sizes() {
    node.filter(function(d) {return d.type=='net';})
        .each(update_net_size)
        .selectAll('rect')
            .attr('x', function(d) {return -net_widths[d.id]/2;})
            .attr('y', function(d) {return -net_heights[d.id]/2;})
            .attr('width', function(d) {return net_widths[d.id];})
            .attr('height', function(d) {return net_heights[d.id];})
            ;
    node.attr('transform', function(d) {
        return 'translate('+[d.x, d.y]+')';
        });
    update_net_text();
}

function update_net_text() {
    node.selectAll("g.node.node_net text") //Position net text by scale
	    .attr('y', function(d) {
	        if (zoom.scale()<1) {
    	        return net_heights[d.id]/2+net_text_margin/zoom.scale()+"px"
    	    } else {
    	        return net_heights[d.id]/2+net_text_margin +"px"
    	    }
    	})
}

//Update given network size based on node position
function update_net_size(d) {
	xstart = d.x
	ystart = d.y
    x0 = graph.nodes[d.contains[0]].x; //first item in net x,y as a start
    x1 = x0;
    y0 = graph.nodes[d.contains[0]].y;
    y1 = y0;
    for (obj in d.contains) {  //min/max of y and x of nodes in net
        xBorder = 0
        yBorder = 0
        tmp = graph.nodes[d.contains[obj]]
        if (tmp.type =="net") {
            xBorder = net_widths[tmp.id] / 2 - net_inner_margin
            yBorder = net_heights[tmp.id] / 2 - net_inner_margin
        }
        x0 = Math.min(graph.nodes[d.contains[obj]].x-xBorder, x0);
        x1 = Math.max(graph.nodes[d.contains[obj]].x+xBorder, x1);
        y0 = Math.min(graph.nodes[d.contains[obj]].y-yBorder, y0);
        y1 = Math.max(graph.nodes[d.contains[obj]].y+yBorder, y1);
    }
    d.x = (x0+x1)/2;  // x, y mid
    d.y = (y0+y1)/2;
    dx = d.x - xstart;
    dy = d.y - ystart;
    net_widths[d.id] = x1 - x0+2*net_inner_margin; //track heights/widths
    net_heights[d.id] = y1 - y0+2*net_inner_margin;

    update_node_positions(d, 2*dx, 2*dy, d3.map(graph.nodes))
}

//Move all the nodes in a network if network position changes
function update_net_position(d, dx, dy) {
    if (d.type == "net") {
        for (var obj in d.contains) {
            if (graph.nodes[d.contains[obj]].type == "net") {
                update_net_position(graph.nodes[d.contains[obj]], dx, dy)
            }
            graph.nodes[d.contains[obj]].x += dx
            graph.nodes[d.contains[obj]].y += dy
        }
    }
}

//Update the position of any nodes and what they affect
function update_node_positions(d, dx, dy, node_list) {
    removeValue(node_list, d)
    for (n in node_list.keys()) {
        if (close_to(node_list.get(node_list.keys()[n]), d)) {
        	move_node(node_list.get(node_list.keys()[n]), dx, dy)
        	update_node_positions(node_list.get(node_list.keys()[n]), 
        	        dx, dy, d3.map(node_list))
        }
    }
}

//Move the provided nodes the provided distance and note that
function move_node(node, dx, dy) {
    if (node.type == "net") { //move a network
        update_net_position(node, dx,dy)
    }
    else{ //move ens or nde
            node.x += dx
            node.y += dy
    }
}

//Redraw if the window is resized
function resize() {
    width = window.innerWidth/2;
    height = window.innerHeight;
    svg.attr("width", width).attr("height", height);
}

//*****************
// Reload the graph
//*****************
var graph=null;
var link=null;
var linkRecur=null;
var node=null;
var node_margin = 35;
var net_inner_margin = 40;
var net_margin = 15;
var node_fontsize = 16;

function reload_graph_data() {
    var data = new FormData();
    data.append('code', editor.getValue());

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/graph.json', true);
    xhr.onload = update_graph;
    xhr.send(data);
}

//Redraw the graph given server response
function update_graph() {
	graph = JSON.parse(this.responseText);

	// was there a parsing error?
	if (graph.error_line != undefined) {
        removeMarker();
		marker = editor.getSession().addMarker(new aceRange(graph.error_line-1, 0, graph.error_line-1, 10), 'highlight', 'fullLine', true);
		editor.getSession().setAnnotations([{row: graph.error_line-1, type:'error'}]);
		return;
	} else {
		if (marker!=null) {
			editor.getSession().removeMarker(marker);
			marker = null;
			}
		editor.getSession().clearAnnotations();
	}

	//separate links into recurrent and nonrecurrent ?move to convert?	
	var nonrecurlink = []
	var recurlink = []
	for (i in graph.links) {
	    if (graph.links[i].target!=graph.links[i].source) {
	        nonrecurlink.push(graph.links[i])
	    } else {
	        recurlink.push(graph.links[i])
	    }
	}
	
	//update the links
	link = container.selectAll('.link.link_std, .link.link_net')
		.data(nonrecurlink, function(d) {return d.id})
	link.enter().append('polyline')
		.attr('class', function(d) {return 'link link_'+d.type;})

	linkRecur = container.selectAll('.link.link_recur')
		.data(recurlink, function(d) {return d.id})
	linkRecur.enter().append('use')
	    .attr('class', function(d) {return 'link link_recur';})
	    .attr('xlink:href', "#recur")	

	//get all the nodes, for updating
	node = container.selectAll('g.node')
		.data(graph.nodes, function(d) {return d.id})
	container.selectAll('g.node text')
		.data(graph.nodes, function(d) {return d.id})
    
    //Create html objects to draw
	var nodeEnter = node
		.enter()
		.append('g')
		.attr('class', function(d) {return 'node node_'+d.type;})
		.on('mouseover', annotateLine)
		.on('mouseout', clearAnnotation)
		.call(drag);

	nodeEnter.filter(function(d) {return d.type=='ens';})
		.append('use')
		.attr('xlink:href', "#ensemble")
		
	nodeEnter.filter(function(d) {return d.type=='nde';})
		.append('circle')
		.attr('r', '20')

	nodeEnter.filter(function(d) {return d.type=='net';})
		.append('rect')
		.attr('x', '-50')
		.attr('y', '-50')
		.attr('rx', '15')
		.attr('ry', '15')
		.attr('width', '100')
		.attr('height', '100')

	//label everything
	nodeEnter.append('text')
	    .text(function(d) {return d.label})

	nodeEnter.selectAll('.node_nde text, .node_ens text')
		.attr('y', '30')
		.style('font-size', node_fontsize)
	
	node.exit().remove();
	link.exit().remove();
	linkRecur.exit().remove();

		
	//redraw so nodes are on top
	container.selectAll('g.node').filter(function(d) {return d.type!='net';})
		.moveToFront();

    update_net_sizes();
	update_line_locations();
	resize();
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

// is the point x, y inside the net
function isin(d, x, y) { 
	return (x<d.x+net_widths[d.id]/2) &&
		   (x>d.x-net_widths[d.id]/2) &&
		   (y<d.y+net_heights[d.id]/2) &&
		   (y>d.y-net_heights[d.id]/2);
}

//Check if node, n is close to origin object, o
function close_to(n, o) { //n is node, o is origin
	if (o.type == "net") { //if origin is net
		if (!(n.type == "net")) { //if node is nde or ens
        	if (!netContains(n, o)) {
            	if (Math.abs(o.x-n.x) < (net_margin+net_widths[o.id]/2) &&
                    Math.abs(o.y-n.y) < (net_margin+net_heights[o.id]/2)) {
                	//console.log('true 1')
                	return true
                }
            }
        }
        else if (!(netContains(n, o) || netContains(o, n))) { //if node is net
            if (Math.abs(o.x-n.x) < (net_widths[n.id]/2+net_widths[o.id]/2)
                && Math.abs(o.y-n.y)<(net_heights[n.id]/2
                                                    +net_heights[o.id]/2)) {
                //console.log('true 2')
            	return true
            }
        }
    }
    else { //if origin is nde or ens
		if (!(n.type == "net")) { //if node nde or ens
			if (Math.abs(o.x-n.x) < node_margin
			   && Math.abs(o.y-n.y) < node_margin) {
             	//console.log('true 3')
				return true
			}
		}
        else { //if node is net
        	if(!netContains(o, n)) {
            	if (Math.abs(o.x-n.x) < (net_margin+net_widths[n.id]/2) &&
                    Math.abs(o.y-n.y) < (net_margin+net_heights[n.id]/2)) {
                	//console.log('true 4')
                	return true
                }
            }
        }
    }
    return false
}

//True if net or any of its subnets contains node
function netContains(node, net) {
    var contain_bool = false
    for (var i in net.contains) {
        if (graph.nodes[net.contains[i]].id == node.id) {
            return true
        }
        else if (graph.nodes[net.contains[i]].type == "net") {
            contain_bool = netContains(node, graph.nodes[net.contains[i]])
            if (contain_bool) {
                return true
            }
        }
    }
    return contain_bool
}

//***********
//Main script
//***********
$(document).ready(function () {
    zoom = d3.behavior.zoom()
        .scaleExtent([.05, 10])
        .on('zoom', zoomed);

    drag = d3.behavior.drag()
        .origin(function(d){return d})
        .on('dragstart', dragstarted)
        .on('drag', dragged)
        .on('dragend', dragended);
      
    //initialize editor
    editor = ace.edit("editor");
    editor.setTheme("ace/theme/monokai");
    editor.getSession().setMode("ace/mode/python");
    editor.on('change', function(event) {reload_graph_data();});

    //initialize file browser
    $('#filebrowser').hide()
    $('#menu_open').click(function() {$('#filebrowser').toggle(200);})
    $('#filebrowser').fileTree({ root: '.', script: '/browse' }, open_file);

    //initialize graph
    svg = d3.select("svg");
    container = svg.append('g');
    zoom(svg);  // set up zooming on the graph
    d3.select(window).on("resize", resize);

    //start this puppy up
    reload_graph_data();
});