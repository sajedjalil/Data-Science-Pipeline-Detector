from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
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
#remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
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
#create sklearn pipeline, fit all, and predit test data
clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('svm', SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
clf.fit(s_data, s_labels)
t_labels = clf.predict(t_data)

#output results for submission
with open("submission.csv","w") as f:
    f.write("id,prediction\n")
    for i in range(len(t_labels)):
        f.write(str(test.id[i])+","+str(t_labels[i])+"\n")
f.close()
#html fun - not necessary
h=[]
h.append("<html><head><script type='text/javascript' src='https://www.google.com/jsapi'></script><script type='text/javascript'>google.load('visualization', '1', {packages:['gauge']});google.setOnLoadCallback(drawChart);function drawChart() {var data = google.visualization.arrayToDataTable([['Label', 'Value'],['Score', 59.004]]);var options = {width: 400, height: 400,redFrom: 0, redTo: 40,yellowFrom:40, yellowTo: 60,greenFrom: 60, greenTo: 100,minorTicks: 5};var chart = new google.visualization.Gauge(document.getElementById('chart_div'));chart.draw(data, options);}</script></head><body><div id='chart_div' style='width: 400px; height: 120px;'></div></body></html>")
with open("output.html","w") as f:
    for i in range(len(h)):
        f.write(h[i]+"\n")
f.close()
#for further review and improvement
print("Feature Names ---------------------")
print(clf.named_steps['v'].get_feature_names())
print("Parameters ---------------------")
print(clf.named_steps['v'].get_params(deep=True))
print("Stop Words List ---------------------")
print(stop_words)