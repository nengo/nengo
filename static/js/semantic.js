
function fix_node_scales(scale) {
    container.selectAll('g.node').filter(function(d){return d.type=="nde" || d.type=="ens";}).attr('transform', function (d) {return 'translate(' + [d.x, d.y] + ')scale('+(1.0/scale)+')';});
    
    container.selectAll('polyline').style('stroke-width', ''+(2.0/scale)+'px');
}