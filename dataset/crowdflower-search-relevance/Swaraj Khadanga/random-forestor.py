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
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

def clean(string):
    #print string
    #string = re.sub(r'http?:[^\s]*','',string,flags=re.MULTILINE)
    string = re.sub(r'[^a-zA-Z]',' ',string)
    words = string.lower().split()
    stemmer = PorterStemmer()
    for i in range(len(words)):
        words[i]=stemmer.stem(words[i])
    string = ' '.join(words)
    return string

train = pd.read_csv('../input/train.csv') #read train
test = pd.read_csv('../input/test.csv') #read test

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


tsvd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0) #dimension reduction
scl = StandardScaler() 
tfidf = TfidfVectorizer(max_features=2500,min_df=0.0,max_df=0.4,stop_words='english',ngram_range=(1,3),use_idf=True,smooth_idf=True,sublinear_tf=True,analyzer='word')
classifier = RandomForestClassifier()
parameters = { 'class__n_estimators': (10,50,100) }
pipe = Pipeline([('tfidf',tfidf),
                 ('tsvd',tsvd),
                 ('scl',scl),
                 ('class',RandomForestClassifier())])
grid_search = GridSearchCV(pipe,parameters)
grid_search.fit(traindata,y)
predicted = grid_search.best_estimator_.predict(testdata) #pipe.predict(testdata)

df = pd.DataFrame()
df['id']=idx
df['prediction']=predicted


df.to_csv('search_result_output.csv',index=False)

#print df.prediction.value_counts()