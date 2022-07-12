
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
		s = clean("%s %s"%(row["product_description"],row["product_title"]))
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
	
if __name__ == "__main__":
	d = index_corpus()
	print(nkd("apple","macbook",d))
	print(nkd("chicken","tenders",d))
	print(nkd("ladder","google",d))
