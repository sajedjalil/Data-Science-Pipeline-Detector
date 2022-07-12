
import pandas as pd

# Use Pandas to read in the training and test data

__author__ = 'sonu'
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from bs4 import BeautifulSoup
stemmer = PorterStemmer()
sw=[]
ML_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height']
ML_STOP_WORDS += list(text.ENGLISH_STOP_WORDS)
for stw in ML_STOP_WORDS:
    sw.append("z"+str(stw))
ML_STOP_WORDS += sw
for i in range(len(ML_STOP_WORDS)):
    ML_STOP_WORDS[i]=stemmer.stem(ML_STOP_WORDS[i])

def clean(f3):
    f2=""
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

train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

idx = test.id.values.astype(int)

train.drop('id',axis=1)
test.drop('id',axis=1)

y = train.median_relevance.values
train.drop(['median_relevance','relevance_variance'],axis=1)

traindata = np.array(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))
testdata = np.array(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))

for i in range(len(traindata)):
    traindata[i]=clean(traindata[i])
for i in range(len(testdata)):
    testdata[i]=clean(testdata[i])


tsvd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
scl = StandardScaler()
tfidf = TfidfVectorizer(max_features=2500,min_df=0.0,max_df=0.4,stop_words=ML_STOP_WORDS,ngram_range=(1,3),use_idf=True,smooth_idf=True,sublinear_tf=True,analyzer='word')
classifier = SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
parameters = { 'class__n_estimators': (10,50,100) }
pipe = Pipeline([('tfidf',tfidf),
                 ('tsvd',tsvd),
                 ('scl',scl),
                 ('class',classifier)])
#grid_search = GridSearchCV(pipe,parameters)
pipe.fit(traindata,y)
predicted = pipe.predict(testdata) #pipe.predict(testdata)

df = pd.DataFrame()
df['id']=idx
df['prediction']=predicted


df.to_csv('search_result_output.csv',index=False)

print(df.prediction.value_counts())

# Now it's yours to take from here!
