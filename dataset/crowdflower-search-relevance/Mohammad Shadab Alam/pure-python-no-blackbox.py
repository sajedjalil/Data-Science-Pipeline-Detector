import csv
import re
import math
d={}
d2={}
li=0
la=0
ws=""
cl=0
w2=[]
w=[]
s=""
a=[]
dlabels={"1":0,"2":1,"3":2,"4":3}
dvals={"0":1,"1":2,"2":3,"3":4}
t = [0,0,0,0]

with open("../input/train.csv",encoding="ascii", errors="surrogateescape") as f:
    d2= csv.DictReader(f)
    for line in d2:
        #print (line)
        cl=int(dlabels[line['median_relevance']])
        s=(" ").join(["q"+ z for z in line["query"].split(" ")])  + " " + line["product_title"] + " " + line["product_description"]
        s=re.sub(r'<([^>]+)>', ' ', s)
        s=s.replace('\r',' ').replace('\n',' ').replace('\t',' ').lower()
        s=re.sub(r'[^a-z0-9]',' ', s)
        w=list(set(s.split(" ")))
        for i in range(len(w)):
            if len(str(w[i]))>2:
                if str(w[i]) in d:
                    wc = list(d[str(w[i])])
                    wc[cl]+=1
                    t[cl]+=1
                    d[str(w[i])]=list(wc)
                else:
                    wc = [0 for k in range(len(dlabels))]
                    wc[cl]+=1
                    t[cl]+=1
                    d[str(w[i])]=list(wc)
    li+=1
f.close

d1={}
h=0.0
t_labels=[]
t_labels1=[]
stopw=['to', 'in', 'and', 'can', 'has', 'with', 'features', 'your', 'black', 'easy', 'from', 'style', 'all', 'you', 'design', 'use', 'will', 'made', 'as', 'it', 'up', 'is', 'are', 'for', 'that', 'of', 'set', 'an', 'be', 'by', 'on', 'the', 'or', 'this', 'one']
for st in stopw:
    if st in d:
        del d[st]
with open("../input/test.csv",encoding="ascii", errors="surrogateescape") as f:
    d1= csv.DictReader(f)
    for line in d1:
        t_labels.append(line["id"])
        s=(" ").join(["q"+ z for z in line["query"].split(" ")])  + " " + line["product_title"] + " " + line["product_description"]
        s=re.sub(r'<([^>]+)>', ' ', s)
        s=s.replace('\r',' ').replace('\n',' ').replace('\t',' ').lower()
        s=re.sub(r'[^a-z0-9]',' ', s)
        w=list(set(s.split(" ")))
        we = [0 for k in range(len(dlabels))]
        for i in range(len(w)):
            if str(w[i]) in d:
                wc = list(d[str(w[i])])
                for j in range(len(wc)):
                    h= float(wc[j]) / float(sum(wc))
                    if h > 0.75 and sum(wc) > 5:
                        h= h * 100
                    if sum(wc)>1000: 
                        print (str(w[i]))
                    #print(wc[j],sum(wc),h,t[j],we[j])
                    we[j]= float(we[j]) + float(h / t[j])
                    #tf_idf= (wc[j]/float(sum(wc))) *(sum(t)/float(sum(wc)))
                    #if sum(wc)>1000:
                    #   stopw.append(str(w[i]))
                    #print(wc[j],sum(wc),t[j],we[j])
                    #we[j]+=tf_idf
        c=0.0
        g=0
        for z in range(len(we)):
            if we[z] > c:
                    g = z
                    c = we[z]
        t_labels1.append(dvals[str(g)])
#print(list(set(stopw)))
#HTML Comments Only
h=[]
h.append("<html><head><script type='text/javascript' src='https://www.google.com/jsapi'></script><script type='text/javascript'>google.load('visualization', '1',{packages:['treemap']});google.setOnLoadCallback(drawChart);function drawChart() {var data = google.visualization.arrayToDataTable([['Words', 'Parent', 'Count'],['B_O_W', null, 0],")
for k,v in d.items():
    p=int(sum(v))
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