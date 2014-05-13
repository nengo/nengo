
function fix_node_scales(scale) {
    container.selectAll('g.node').filter(function(d){return d.type=="nde" || d.type=="ens";}).attr('transform', function (d) {return 'translate(' + [d.x, d.y] + ')scale('+(1.0/scale)+')';});
    
    container.selectAll('polyline').style('stroke-width', ''+(2.0/scale)+'px');
    container.selectAll('g.node_net rect').attr('rx', 15.0/scale).attr('ry', 15.0/scale);
}

function fix_labels(scale) {
    node.selectAll('.node_ens text, .node_nde text') // already scaled with g 
        .style("font-size", function (d) { return node_fontsize;});
    node.selectAll('.node_net text') //Change the fonts size as a fcn of scale
        .style("font-size", function (d) { return node_fontsize / scale;});
}