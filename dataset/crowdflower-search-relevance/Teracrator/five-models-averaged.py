from bs4 import BeautifulSoup
from nltk.stem.porter import *
#from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math
import numpy as np
import pandas as pd
import re

# declarations
stemmer = PorterStemmer()
sw=[]
s_data = []
t_data = []
t_queries = []
t_labels = []
t_labelsf = []
#stopwords tweak
ML_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height']
ML_STOP_WORDS += list(text.ENGLISH_STOP_WORDS)
for stw in ML_STOP_WORDS:
    sw.append("z"+str(stw))
ML_STOP_WORDS += sw
for i in range(len(ML_STOP_WORDS)):
    ML_STOP_WORDS[i]=stemmer.stem(ML_STOP_WORDS[i])

def ML_TEXT_CLEAN(f2,f3):
    if len(f2)<3:
        f2="feature2null"
    if len(f3)<3:
        f3="feature3null"
    tx = BeautifulSoup(f3)
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip()
    s = (" ").join(["z"+ str(z) for z in f2.split(" ")]) + " " + tx
    s = re.sub("[^a-zA-Z0-9]"," ", s)
    s = re.sub("[0-9]{1,3}px"," ", s)
    s = re.sub(" [0-9]{1,6} |000"," ", s)
    s = (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s = s.lower()
    return s

#load data
train = pd.read_csv("../input/train.csv").fillna(" ")
test  = pd.read_csv("../input/test.csv").fillna(" ")

for i in range(len(train.id)):
    s = ML_TEXT_CLEAN(train.product_title[i], train.product_description[i])
    s_data.append((train["query"][i], s, str(train["median_relevance"][i])))
for i in range(len(test.id)):
    s = ML_TEXT_CLEAN(test.product_title[i], test.product_description[i])
    t_data.append((test["query"][i], s, test.id[i]))
    if test["query"][i] not in t_queries:
        t_queries.append(test["query"][i])
df1 = pd.DataFrame(s_data)
df2 = pd.DataFrame(t_data)
for tq in t_queries:
    df1_s = df1[df1[0]==tq]
    df2_s = df2[df2[0]==tq]
    #Naive Bayes
    clf = MultinomialNB(alpha=0.01)
    v = TfidfVectorizer(use_idf=True,min_df=0,ngram_range=(1,6),lowercase=True,sublinear_tf=True,stop_words=ML_STOP_WORDS)
    clf.fit(v.fit_transform(df1_s[1]), df1_s[2])
    t_labels_nb = clf.predict(v.transform(df2_s[1]))
    #SDG
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('sdg', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True, verbose=0, warm_start=False))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_sdg = clf.predict(df2_s[1])
    #SVD/Standard Scaler/SVM
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('svd', TruncatedSVD(n_components=100)),  ('scl', StandardScaler()), ('svm', SVC(C=10))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_sv_ = clf.predict(df2_s[1])

    #print(tq)
    for i in range(len(t_labels_nb)):
        t_labels1 = list(df2_s[2])
        t_labelsf.append((t_labels1[i],t_labels_nb[i],t_labels_sdg[i],t_labels_sv_[i]))
df3 = pd.DataFrame(t_labelsf)
df3 = df3.sort([0])
preds2 = list(df3[1])
preds3 = list(df3[2])
preds4 = list(df3[3])


#eye chart for review
df3.columns = ['ID','Naive Bayes', 'SDG', 'SVD SVM','Decision Tree','KNeighbors']
print(df3[:50])
print("looks like DecisionTreeClassifier has highest disagreement")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
data = df3[:50]
data = data.drop('ID', 1)
fig = plt.figure()
fig.set_size_inches(10, 4)
#plt.legend()
plt.plot(data)
plt.axis([0, 50, 0, 5])
plt.savefig('plot.png')

p3 = []
for i in range(len(preds2)):
    x = round((int(preds4[i]) + int(preds3[i]) + int(preds2[i]))/5,0)
    p3.append(int(x))
    
submission = pd.DataFrame({"id": test.id, "prediction": p3})
submission.to_csv("ensemble5models.csv", index=False)