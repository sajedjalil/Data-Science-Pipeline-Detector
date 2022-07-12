"""
	__author__: 	 Triskelion
	__description__: Normalized Kaggle Distance for visualization, topic modeling and semantic 
					 knowledge base creation.
	
					 Based on Normalized Google Distance, which in turn is based on Normalized 
					 Compression Distance, which in turn is based on Information Distance, 
					 which in turn is based on Kolmogorov Complexity.
					 
					 A bit messy to mix HTML with Python, but that's how it goes.
					 
					 Ref: http://homepages.cwi.nl/~paulv/papers/crc08.pdf
"""
import csv
import re
from collections import defaultdict
import math
import json

def clean(s):
	# Returns unique token-sorted cleaned lowercased text
	return " ".join(sorted(set(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)))).lower()

def index_document(s,d):
	# Creates half the matrix of pairwise tokens
	# This fits into memory, else we have to choose a Count-min Sketch probabilistic counter 
	tokens = s.split()
	for x in range(len(tokens)):
		d[tokens[x]] += 1
		for y in range(x+1,len(tokens)):
			d[tokens[x]+"_X_"+tokens[y]] += 1
	return d

def index_corpus():	
	# Create our count dictionary and fill it with train and test set (pairwise) token counts	
	d = defaultdict(int)
	for e, row in enumerate( csv.DictReader(open("../input/train.csv",'r', newline='', encoding='utf8'))):
		s = clean("medianrellabel%s %s %s"%(row["median_relevance"], row["product_description"],row["product_title"]))
		d = index_document(s,d)
	for e, row in enumerate( csv.DictReader(open("../input/test.csv",'r', newline='', encoding='utf8'))):
		s = clean("%s %s"%(row["product_description"],row["product_title"]))
		d = index_document(s,d)
	return d	
	
def nkd(token1, token2, d):
	# Returns the semantic Normalized Kaggle Distance between two tokens
	sorted_tokens = sorted([clean(token1), clean(token2)])
	token_x = sorted_tokens[0]
	token_y = sorted_tokens[1]
	if d[token_x] == 0 or d[token_y] == 0 or d[token_x+"_X_"+token_y] == 0:
		return 2.
	else:
		#print d[token_x], d[token_y], d[token_x+"_X_"+token_y], token_x+"_X_"+token_y
		logcount_x = math.log(d[token_x])
		logcount_y = math.log(d[token_y])
		logcount_xy = math.log(d[token_x+"_X_"+token_y])
		log_index_size = math.log(100000) # fixed guesstimate
		nkd = (max(logcount_x,logcount_y)-logcount_xy) / (log_index_size-min(logcount_x,logcount_y))	
		return nkd

def generate_json_graph(targets,d):
	# From a comma seperated string this creates the JSON to build a force-directed graph in D3.js
	targets = targets.split(",")
	result = defaultdict(list)
	
	for i in range(len(targets)):
		result["nodes"].append({"s": targets[i], "y": d[targets[i]] })
		for j in range(i+1,len(targets)):
			result["links"].append({"source": i, "target":  j, "strength": nkd(targets[i], targets[j], d)})
	return json.dumps(result)	

def multiple_choice(d,question,anchor,choices):
	# Answers a multiple choice question in HTML where 'anchor' is the keyword
	q = """<li class="pane"><h3>%s</h3><ul>%s</ul></li>"""%(question,
			"".join(["<li><span>%s</span>%s</li>"%(round(w[0],3),w[1]) for w in sorted([(nkd(f,anchor,d),f) for f in choices])]))
	return q
	
def topic_modeling(d,labeled_topics):
	# Labeled_topics is a list of topics you want to create
	# Uses only words in train set
	v = {}
	for topic in labeled_topics:
		v[topic] = []
	for e, row in enumerate( csv.DictReader(open("../input/train.csv",'r', newline='', encoding='utf8'))):
		words = clean("%s %s"%(row["product_description"],row["product_title"])).split()
		for word in words:
			for k in v:
				v[k].append( (nkd(word,k,d),word) )
	out = ""
	for k in v:
		out += "<h3>Topic: %s</h3><p>"%k
		l = []
		for t in sorted(set(v[k]))[:25]:
			l.append(t[1])
		out += ", ".join(l)+ "</p>"
	return out
	
def edge_bundeling(d):
    # Create edge bundeling visualization
	# Uses only words in train set
	w = []
	for e, row in enumerate( csv.DictReader(open("../input/train.csv",'r', newline='', encoding='utf8'))):
		w += clean("%s %s"%(row["product_description"],row["product_title"])).split()
	w = list(set(w))	
	
	# Find 115 words closest to LABEL_4
	c = []
	for word in w:
		c.append((nkd(word,"medianrellabel4",d),word))
	c = sorted(c)[:115]
	
	edge = defaultdict(list)
	# Find 5 closest words for every closest word
	for e, (distance,word) in enumerate(c):
		for anchor_distance, anchor_word in c:
			edge[(e,word)].append((nkd(word,anchor_word,d),anchor_word))
		edge[(e,word)] = [f[1] for f in sorted(edge[(e,word)])[:4]]
	
	# Format json
	result = []
	for k in sorted(edge):
		result.append({"name":k[1],"imports":edge[k]})
	return json.dumps(result)
	
if __name__ == "__main__":
	d = index_corpus()

	manufacturers = ["Amazon","Apple","Google","Microsoft","Motorola"]
	html1 = multiple_choice(d,"Who created the iPhone phone?","iPhone",manufacturers)
	html2 = multiple_choice(d,"Who created the Nexus phone?","Nexus",manufacturers)
	html3 = multiple_choice(d,"Who created the Moto phone?","Moto",manufacturers)
	html4 = multiple_choice(d,"Who created the Fire phone?","Fire",manufacturers)
	
	film_studios = ["Disney","20th","Fox","Paramount","Sony","Colombia","Goldwyn","Universal","Warner"]
	html5 = multiple_choice(d,"Which major film studio produced the Batman films?","Batman",film_studios)
	html6 = multiple_choice(d,"Which major film studio produced the film Frozen?","Frozen",film_studios)
	
	topics = ["Laptop","Children","Movies"]
	html7 = topic_modeling(d,topics)
	
	targets = "one,two,white,green,pc,laptop,desktop,mouse,red"
	json1 = generate_json_graph(targets,d)
	
	json2 = edge_bundeling(d)
	
	with open("output.html","wb") as outfile:
		html = """<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>Triskelion &bull; Normalized Kaggle Distance</title>
		<meta name="robots" content="noindex,nofollow,noodp,nonothing,goaway">
		<link href='https://fonts.googleapis.com/css?family=Open+Sans:400,800' rel='stylesheet' type='text/css'>
		<style>
			* { margin:0; padding: 0;}
			body { font-family: "Open Sans",Verdana,sans-serif; margin: 30px auto; width: 700px; }
			p { font-size: 18px; line-height: 27px; padding-bottom: 27px; color: #111; }
			h1 { font-weight: 800; font-size: 33px; color: #2C3E50; padding-bottom: 10px;}
			h2 { font-weight: 800; font-size: 22px; color: #2C3E50; padding-bottom: 10px; }
			h3 { font-weight: 800; font-size: 18px; color: #34495E; padding-bottom: 10px; }
			small { color: #7F8C8D; }
			ul.panes { display: block; overflow: hidden; list-style-type: none; padding: 10px 0px;}
			li.pane { float: left; width: 300px; margin-right: 40px; padding-bottom: 40px; }
			ul { list-style-type: none; }
			span { display: block; width: 50px; padding-right: 15px; text-align: right; height: 20px; float: left; }
			li ul li { color: #7F8C8D; }
			li ul li:first-child { color: #111; }
			.node { font: 400 14px "Open Sans", Verdana, sans-serif; fill: #333; cursor:pointer;}
			.node:hover {fill: #000;}
			.link {stroke: steelblue; stroke-opacity:.4;fill: none; pointer-events: none;}
			.node:hover,.node--source,.node--target { font-weight: 700;}
			.node--source { fill: #2ca02c;}
			.node--target { fill: #d62728;}
			.link--source,.link--target { stroke-opacity: 1; stroke-width: 2px;}
			.link--source { stroke: #d62728;}
			.link--target { stroke: #2ca02c;}
		</style>
	</head>
	<body>
	    <noscript>Uses JavaScript for visualizations (d3.js)</noscript>
		<h1>Normalized Kaggle Distance</h2>
		<p>Normalized Kaggle Distance (NKD) uses a search engine index as a compressor by looking at page count statistics. Semantically related words get a closer distance. This allows us to do all sorts of fun stuff.</p>
		<p>NKD is based on <a href="http://en.wikipedia.org/wiki/Normalized_Google_distance">Normalized Google Distance</a> which uses the Google index. We show that, even with a relatively small corpus, we can still extract useful semantic information.</p>
		
		<h2>Semantic Knowledge Base</h2>
		
		<ul class="panes">
			%s %s %s %s %s %s
		</ul>
		
		<h2>Topic Modeling</h2>
		%s
		
		<h2>Force-Directed Clustering <small>(drag to interact)</small></h2>
		
		<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>
		<script>
		function draw_cluster_graph(){
			var json = %s;	

			var width = 700,
				height = 600;

			var svg = d3.select("body").append("svg")
				.attr("width", width)
				.attr("height", height);

			var force = d3.layout.force()
						.gravity(.05)
						.distance(800)
						.charge(-800)
						.size([width, height]);

			force
				.nodes(json.nodes)
				.links(json.links)
				.linkDistance(function(d) { return  300 - ((1 - d.strength) * 300); })
				.start();

			var link = svg.selectAll(".link")
						.data(json.links)
						.enter().append("line")
						.attr("class", "link")
						.style("stroke", "#2C3E50")
						.style("stroke-opacity", function(d) { return 0.2 - (d.strength/4); })
						.style("stroke-width", 1)
						  ;
			var node = svg.selectAll(".node")
						.data(json.nodes)
						.enter().append("g")
						.attr("class", "node")
						.call(force.drag);
						  
			node.append("circle")
				.attr("r", function(d) { return (Math.log(d.y) * 5)+3; })
				.style("fill", "#3498DB")
				.style("fill-opacity", 0.23);

			node.append("text")
				.attr("dx", 12)
				.attr("class", "text")
				.attr("dy", ".35em")
				.style("font-family", "Open Sans")
				.style("font-weight", "400")
				.style("color", "#2C3E50")
				.text(function(d) { return d.s });

			force.on("tick", function() {
				link.attr("x1", function(d) { return d.source.x; })
					.attr("y1", function(d) { return d.source.y; })
					.attr("x2", function(d) { return d.target.x; })
					.attr("y2", function(d) { return d.target.y; });

				node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
			});

		};
		draw_cluster_graph();

		d3.select("body").append("h2").text("Hierarchical edge bundling");
		var diameter = 700,
			radius = diameter / 2,
			innerRadius = radius - 120;

		var cluster = d3.layout.cluster()
			.size([360, innerRadius])
			.sort(null)
			.value(function(d) { return d.size; });

		var bundle = d3.layout.bundle();

		var line = d3.svg.line.radial()
			.interpolate("bundle")
			.tension(.85)
			.radius(function(d) { return d.y; })
			.angle(function(d) { return d.x / 180 * Math.PI; });

		var svg = d3.select("body").append("svg")
			.attr("width", diameter)
			.attr("height", diameter)
		  .append("g")
			.attr("transform", "translate(" + radius + "," + radius + ")");

		var link = svg.append("g").selectAll(".link"),
			node = svg.append("g").selectAll(".node");

		var classes = %s;	

		  var nodes = cluster.nodes(packageHierarchy(classes)),
			  links = packageImports(nodes);

		  link = link
			  .data(bundle(links))
			.enter().append("path")
			  .each(function(d) { d.source = d[0], d.target = d[d.length - 1]; })
			  .attr("class", "link")
			  .attr("d", line);

		  node = node
			  .data(nodes.filter(function(n) { return !n.children; }))
			.enter().append("text")
			  .attr("class", "node")
			  .attr("dy", ".31em")
			  .attr("transform", function(d) { return "rotate(" + (d.x - 90) + ")translate(" + (d.y + 8) + ",0)" + (d.x < 180 ? "" : "rotate(180)"); })
			  .style("text-anchor", function(d) { return d.x < 180 ? "start" : "end"; })
			  .text(function(d) { return d.key; })
			  .on("mouseover", mouseovered)
			  .on("mouseout", mouseouted);


		function mouseovered(d) {
		  node
			  .each(function(n) { n.target = n.source = false; });

		  link
			  .classed("link--target", function(l) { if (l.target === d) return l.source.source = true; })
			  .classed("link--source", function(l) { if (l.source === d) return l.target.target = true; })
			.filter(function(l) { return l.target === d || l.source === d; })
			  .each(function() { this.parentNode.appendChild(this); });

		  node
			  .classed("node--target", function(n) { return n.target; })
			  .classed("node--source", function(n) { return n.source; });
		}

		function mouseouted(d) {
		  link
			  .classed("link--target", false)
			  .classed("link--source", false);

		  node
			  .classed("node--target", false)
			  .classed("node--source", false);
		}

		d3.select(self.frameElement).style("height", diameter + "px");

		// Lazily construct the package hierarchy from class names.
		function packageHierarchy(classes) {
		  var map = {};

		  function find(name, data) {
			var node = map[name], i;
			if (!node) {
			  node = map[name] = data || {name: name, children: []};
			  if (name.length) {
				node.parent = find(name.substring(0, i = name.lastIndexOf(".")));
				node.parent.children.push(node);
				node.key = name.substring(i + 1);
			  }
			}
			return node;
		  }

		  classes.forEach(function(d) {
			find(d.name, d);
		  });

		  return map[""];
		}

		// Return a list of imports for the given array of nodes.
		function packageImports(nodes) {
		  var map = {},
			  imports = [];

		  // Compute a map from name to node.
		  nodes.forEach(function(d) {
			map[d.name] = d;
		  });

		  // For each import, construct a link from the source to target node.
		  nodes.forEach(function(d) {
			if (d.imports) d.imports.forEach(function(i) {
			  imports.push({source: map[d.name], target: map[i]});
			});
		  });

		  return imports;
		}
		
		
		
		</script>
	</body>
</html>"""%(html1,html2,html3,html4,html5,html6,html7,json1,json2)
		outfile.write(html.encode('utf-8'))
