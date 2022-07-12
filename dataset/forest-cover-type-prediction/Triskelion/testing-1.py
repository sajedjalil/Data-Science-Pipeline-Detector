with open("miserables.json","wb") as outfile:
    outfile.write("""{
  "nodes":[
    {"name":"Myriel","group":1},
    {"name":"Napoleon","group":1},
    {"name":"Mlle.Baptistine","group":1},
    {"name":"Mme.Magloire","group":1},
    {"name":"CountessdeLo","group":1},
    {"name":"Geborand","group":1},
    {"name":"Champtercier","group":1},
    {"name":"Cravatte","group":1},
    {"name":"Count","group":1},
    {"name":"OldMan","group":1},
    {"name":"Labarre","group":2},
    {"name":"Valjean","group":2},
    {"name":"Marguerite","group":3},
    {"name":"Mme.deR","group":2},
    {"name":"Isabeau","group":2},
    {"name":"Gervais","group":2},
    {"name":"Tholomyes","group":3},
    {"name":"Listolier","group":3},
    {"name":"Fameuil","group":3},
    {"name":"Blacheville","group":3},
    {"name":"Favourite","group":3},
    {"name":"Dahlia","group":3},
    {"name":"Zephine","group":3},
    {"name":"Fantine","group":3},
    {"name":"Mme.Thenardier","group":4},
    {"name":"Thenardier","group":4},
    {"name":"Cosette","group":5},
    {"name":"Javert","group":4},
    {"name":"Fauchelevent","group":0},
    {"name":"Bamatabois","group":2},
    {"name":"Perpetue","group":3},
    {"name":"Simplice","group":2},
    {"name":"Scaufflaire","group":2},
    {"name":"Woman1","group":2},
    {"name":"Judge","group":2},
    {"name":"Champmathieu","group":2},
    {"name":"Brevet","group":2},
    {"name":"Chenildieu","group":2},
    {"name":"Cochepaille","group":2},
    {"name":"Pontmercy","group":4},
    {"name":"Boulatruelle","group":6},
    {"name":"Eponine","group":4},
    {"name":"Anzelma","group":4},
    {"name":"Woman2","group":5},
    {"name":"MotherInnocent","group":0},
    {"name":"Gribier","group":0},
    {"name":"Jondrette","group":7},
    {"name":"Mme.Burgon","group":7},
    {"name":"Gavroche","group":8},
    {"name":"Gillenormand","group":5},
    {"name":"Magnon","group":5},
    {"name":"Mlle.Gillenormand","group":5},
    {"name":"Mme.Pontmercy","group":5},
    {"name":"Mlle.Vaubois","group":5},
    {"name":"Lt.Gillenormand","group":5},
    {"name":"Marius","group":8},
    {"name":"BaronessT","group":5},
    {"name":"Mabeuf","group":8},
    {"name":"Enjolras","group":8},
    {"name":"Combeferre","group":8},
    {"name":"Prouvaire","group":8},
    {"name":"Feuilly","group":8},
    {"name":"Courfeyrac","group":8},
    {"name":"Bahorel","group":8},
    {"name":"Bossuet","group":8},
    {"name":"Joly","group":8},
    {"name":"Grantaire","group":8},
    {"name":"MotherPlutarch","group":9},
    {"name":"Gueulemer","group":4},
    {"name":"Babet","group":4},
    {"name":"Claquesous","group":4},
    {"name":"Montparnasse","group":4},
    {"name":"Toussaint","group":5},
    {"name":"Child1","group":10},
    {"name":"Child2","group":10},
    {"name":"Brujon","group":4},
    {"name":"Mme.Hucheloup","group":8}
  ],
  "links":[
    {"source":1,"target":0,"value":1},
    {"source":2,"target":0,"value":8},
    {"source":27,"target":25,"value":5},
    {"source":27,"target":24,"value":1},
    {"source":27,"target":26,"value":1},
    {"source":76,"target":48,"value":1},
    {"source":76,"target":58,"value":1}
  ]
}""".encode('utf-8'))


with open("output.html","wb") as outfile:
    outfile.write("""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>

.node {
  stroke: #fff;
  stroke-width: 1.5px;
}

.link {
  stroke: #999;
  stroke-opacity: .6;
}

</style>
</head>
<body>


<script src="https://www.jasondavies.com/wordcloud/d3.layout.cloud.js"></script>
<script>


graph = {
  "nodes":[
    {"name":"Myriel","group":1},
    {"name":"Napoleon","group":1},
    {"name":"Mlle.Baptistine","group":1},
    {"name":"Mme.Magloire","group":1},
    {"name":"CountessdeLo","group":1},
    {"name":"Geborand","group":1},
    {"name":"Champtercier","group":1},
    {"name":"Cravatte","group":1},
    {"name":"Count","group":1},
    {"name":"OldMan","group":1},
    {"name":"Labarre","group":2},
    {"name":"Valjean","group":2},
    {"name":"Marguerite","group":3},
    {"name":"Mme.deR","group":2},
    {"name":"Isabeau","group":2},
    {"name":"Gervais","group":2},
    {"name":"Tholomyes","group":3},
    {"name":"Listolier","group":3},
    {"name":"Fameuil","group":3},
    {"name":"Blacheville","group":3},
    {"name":"Favourite","group":3},
    {"name":"Dahlia","group":3},
    {"name":"Zephine","group":3},
    {"name":"Fantine","group":3},
    {"name":"Mme.Thenardier","group":4},
    {"name":"Thenardier","group":4},
    {"name":"Cosette","group":5},
    {"name":"Javert","group":4},
    {"name":"Fauchelevent","group":0},
    {"name":"Bamatabois","group":2},
    {"name":"Perpetue","group":3},
    {"name":"Simplice","group":2},
    {"name":"Scaufflaire","group":2},
    {"name":"Woman1","group":2},
    {"name":"Judge","group":2},
    {"name":"Champmathieu","group":2},
    {"name":"Brevet","group":2},
    {"name":"Chenildieu","group":2},
    {"name":"Cochepaille","group":2},
    {"name":"Pontmercy","group":4},
    {"name":"Boulatruelle","group":6},
    {"name":"Eponine","group":4},
    {"name":"Anzelma","group":4},
    {"name":"Woman2","group":5},
    {"name":"MotherInnocent","group":0},
    {"name":"Gribier","group":0},
    {"name":"Jondrette","group":7},
    {"name":"Mme.Burgon","group":7},
    {"name":"Gavroche","group":8},
    {"name":"Gillenormand","group":5},
    {"name":"Magnon","group":5},
    {"name":"Mlle.Gillenormand","group":5},
    {"name":"Mme.Pontmercy","group":5},
    {"name":"Mlle.Vaubois","group":5},
    {"name":"Lt.Gillenormand","group":5},
    {"name":"Marius","group":8},
    {"name":"BaronessT","group":5},
    {"name":"Mabeuf","group":8},
    {"name":"Enjolras","group":8},
    {"name":"Combeferre","group":8},
    {"name":"Prouvaire","group":8},
    {"name":"Feuilly","group":8},
    {"name":"Courfeyrac","group":8},
    {"name":"Bahorel","group":8},
    {"name":"Bossuet","group":8},
    {"name":"Joly","group":8},
    {"name":"Grantaire","group":8},
    {"name":"MotherPlutarch","group":9},
    {"name":"Gueulemer","group":4},
    {"name":"Babet","group":4},
    {"name":"Claquesous","group":4},
    {"name":"Montparnasse","group":4},
    {"name":"Toussaint","group":5},
    {"name":"Child1","group":10},
    {"name":"Child2","group":10},
    {"name":"Brujon","group":4},
    {"name":"Mme.Hucheloup","group":8}
  ],
  "links":[
    {"source":1,"target":0,"value":1},
    {"source":2,"target":0,"value":8},
    {"source":27,"target":25,"value":5},
    {"source":27,"target":24,"value":1},
    {"source":27,"target":26,"value":1},
    {"source":76,"target":48,"value":1},
    {"source":76,"target":58,"value":1}
  ]
};



var width = 960,
    height = 500;

var color = d3.scale.category20();

var force = d3.layout.force()
    .charge(-120)
    .linkDistance(30)
    .size([width, height]);

var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height);


function draw(graph) {
    console.log(graph); // this is your data
    
    function(error, graph) {
  force
      .nodes(graph.nodes)
      .links(graph.links)
      .start();

  var link = svg.selectAll(".link")
      .data(graph.links)
    .enter().append("line")
      .attr("class", "link")
      .style("stroke-width", function(d) { return Math.sqrt(d.value); });

  var node = svg.selectAll(".node")
      .data(graph.nodes)
    .enter().append("circle")
      .attr("class", "node")
      .attr("r", 5)
      .style("fill", function(d) { return color(d.group); })
      .call(force.drag);

  node.append("title")
      .text(function(d) { return d.name; });

  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });
  });
};
    
    
}

</script>
    
</body>
</html>
    
    
    
    
    """.encode('utf-8'))