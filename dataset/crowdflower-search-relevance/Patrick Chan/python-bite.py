#-----------------------------------------
# Team Jeet Kune Do ML
# Mashing Styles
# Python Bite Script
#----------------------------------------

from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction import text

# array declarations
sw=[]
s_labels = []
t_data = []
t_labels = []
t_rationale=[]
stemmer = PorterStemmer()
d={}
w=[]
s=""
dlabels={"1":0,"2":1,"3":2,"4":3}
dvals={"0":1,"1":2,"2":3,"3":4}
t = [0,0,0,0]

#stopwords tweak
stop_words1 = ['http','www','img','border','color','style','padding','table','font','thi','inch','ha','width','height']
stop_words1 += list(text.ENGLISH_STOP_WORDS)
for stw in stop_words1:
    sw.append("q"+str(stw))
    sw.append("z"+str(stw))
stop_words1 += sw
for i in range(len(stop_words1)):
    stop_words1[i]=stemmer.stem(stop_words1[i])
#load data
train = pd.read_csv("../input/train.csv").fillna(" ")
test  = pd.read_csv("../input/test.csv").fillna(" ")
# Clean up & BOW build
for i in range(len(train.id)):
    tx = BeautifulSoup(train.product_description[i])
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip()
    #if "translation tool" in tx:
    #    tx = tx[-500:]
    s = (" ").join(["q"+ str(z) for z in train["query"][i].split(" ")]) + " " + (" ").join(["z"+ str(z) for z in train.product_title[i].split(" ")]) + " " + tx
    s = re.sub("[^a-zA-Z0-9]"," ", s)
    s = re.sub("[0-9]{1,3}px"," ", s)
    s = re.sub(" [0-9]{1,6} |000"," ", s)
    s = (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s = s.lower()
    s_labels.append(str(train["median_relevance"][i]))
    cl = int(train["median_relevance"][i])-1
    w = list(s.split(" "))
    #ngrams
    ngr=[]
    for ng in range(len(w)-1):
        ngr.append(w[ng]+"_"+w[ng+1])
    w += ngr
    for l in range(len(w)):
        if str(w[l]) in d:
            wc = list(d[str(w[l])])
            wc[cl]+=(1/len(w)) *(2-float(train["relevance_variance"][i]))
            t[cl]+=1
            d[str(w[l])]=list(wc)
        else:
            wc = [0.0 for k in range(len(dlabels))]
            wc[cl]+=(1/len(w)) * (2-float(train["relevance_variance"][i]))
            t[cl]+=1
            d[str(w[l])]=list(wc)
d1={}
h=0.0
t_labels=[]
t_labels1=[]

for st in stop_words1:
    if st in d:
        del d[st]

for i in range(len(test.id)):
    t_labels.append(test["id"][i])
    tx = BeautifulSoup(test.product_description[i])
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip()
    tx = (" ").join([z for z in tx.split(" ")])
    #if "translation tool" in tx:
    #    tx = tx[-500:]
    s = (" ").join(["q"+ str(z) for z in test["query"][i].split(" ")]) + " " + (" ").join(["z"+ str(z) for z in test.product_title[i].split(" ")]) + " " + tx
    s = re.sub("[^a-zA-Z0-9]"," ", s)
    s = re.sub("[0-9]{1,3}px"," ", s)
    s = re.sub(" [0-9]{1,6} |000"," ", s)
    s = (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s = s.lower()
    w = list(s.split(" "))
    #ngrams
    ngr=[]
    for ng in range(len(w)-1):
        ngr.append(w[ng]+"_"+w[ng+1])
    w += ngr
    we = [0.0 for k in range(len(dlabels))]

    for l in range(len(w)):
        if str(w[l]) in d:
            wc = list(d[str(w[l])])
            for j in range(len(wc)):
                h= float(wc[j]) / float(sum(wc))
                if h > 0.75 and sum(wc) > 0:
                    h= h * 100
                #if sum(wc)>1000: 
                    #print (str(w[l]))
                #print(wc[j],sum(wc),h,t[j],we[j])
                we[j]= float(we[j]) + float(h / t[j])
    c=0.0
    g=0
    t_rationale.append((",").join(map(str,we)))
    for z in range(len(we)):
        if we[z] > c:
                g = z
                c = we[z]
    t_labels1.append(dvals[str(g)])
#HTML Comments
h=[]
h.append("<html><head><script type='text/javascript' src='https://www.google.com/jsapi'></script><script type='text/javascript'>google.load('visualization', '1',{packages:['treemap']});google.setOnLoadCallback(drawChart);function drawChart() {var data = google.visualization.arrayToDataTable([['Words', 'Parent', 'Count'],['B_O_W', null, 0],")
for k,v in d.items():
    p=int(sum(v)*100)
    if p>500:
        h.append("['" + str(k) + "','B_O_W'," + str(p) + " ],")
h.append("]);tree = new google.visualization.TreeMap(document.getElementById('chart_div'));tree.draw(data, {});}</script></head><body><div id='chart_div' style='width: 700px; height: 500px;'></div></body></html>")
with open("output.html","w",encoding="ascii", errors="surrogateescape") as f:
    for i in range(len(h)):
        f.write(h[i]+"\n")
f.close()
#End Comments
with open("submission.csv","w") as f:
    f.write("id,prediction\n")
    for i in range(len(t_labels)):
        f.write(str(t_labels[i])+","+str(t_labels1[i])+"\n")
f.close()

with open("submission_rationale.csv","w") as f:
    f.write("id,prediction,1,2,3,4\n")
    for i in range(len(t_labels)):
        f.write(str(t_labels[i])+","+str(t_labels1[i])+","+t_rationale[i]+"\n")
f.close()