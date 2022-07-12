from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import text

#clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=1000, subsample=0.7, 
#min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=10, init=None, 
#random_state=None, max_features=None, verbose=2, max_leaf_nodes=None, warm_start=False)
#clf = DecisionTreeClassifier()
clf = MultinomialNB(alpha=.09)
v = TfidfVectorizer(use_idf=True,min_df=0,max_df=0.05,ngram_range=(1,3),lowercase=True,sublinear_tf=True,stop_words='english')
# array declarations
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)
#load data
train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")
stemmer = PorterStemmer()
for i in range(len(train.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    s_data.append(s)
    s_labels.append(str(train["median_relevance"][i]))
for i in range(len(test.id)):
    s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
    s=re.sub("[^a-zA-Z0-9]"," ", s)
    s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
    t_data.append(s)
clf.fit(v.fit_transform(s_data), s_labels)
#model training performance
p=0
n=0
t_labels = clf.predict(v.transform(s_data))
for i in range(len(t_labels)):
    if int(s_labels[i])==int(t_labels[i]):
        p+=1
    else:
        n+=1
print(p,n)
#result
t_labels = clf.predict(v.transform(t_data))
#HTML Comments Only
h=[]
h.append("<html><head><title>First Submission</title>")
h.append("<script type=\'text/javascript\' src=\'https://www.google.com/jsapi\'></script><script type=\'text/javascript\'>google.load(\'visualization\', \'1\', {packages:[\'corechart\']});google.setOnLoadCallback(drawChart);function drawChart() {var data = google.visualization.arrayToDataTable([[\'Pass/Fail\', \'Training Records\'],")
h.append("[\'Pass\',     "+str(p)+"],[\'Fail\',      "+str(n)+"]")
h.append("]);var options = {title: \'Training Pass Fail\',is3D: true,};var chart = new google.visualization.PieChart(document.getElementById(\'piechart_3d\'));chart.draw(data, options);}</script>")
h.append("</head><body>")
h.append("<div id=\'piechart_3d\' style=\'width: 700px; height: 400px;\'></div>")
h.append("</body></html>")
with open("output.html","w") as f:
    for i in range(len(h)):
        f.write(h[i]+"\n")
f.close()
#End Comments
with open("submission.csv","w") as f:
    f.write("id,prediction\n")
    for i in range(len(t_labels)):
        f.write(str(test.id[i])+","+str(t_labels[i])+"\n")
f.close()

#for further review and improvement
print("Feature Names ---------------------")
print(v.get_feature_names())
print("Parameters ---------------------")
print(v.get_params(deep=True))
print("Stop Words List ---------------------")
print(stop_words)