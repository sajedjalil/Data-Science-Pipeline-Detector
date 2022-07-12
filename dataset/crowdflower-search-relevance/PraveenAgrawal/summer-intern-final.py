import pandas as pd			# Pandas to read in the training and test data
import numpy as np                      # fast precompiled functions for numerical routines
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

########NLP=================

stemmer = PorterStemmer()
sw=[]
ML_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height']
ML_STOP_WORDS += list(text.ENGLISH_STOP_WORDS)
for stw in ML_STOP_WORDS:
    sw.append("z"+str(stw))
ML_STOP_WORDS += sw
for i in range(len(ML_STOP_WORDS)):
    ML_STOP_WORDS[i]=stemmer.stem(ML_STOP_WORDS[i])

#============cleaning feature data if length >3==========================
def clean(f3):
    f2=""
    if len(f2)<3:
        f2="feature2null"
    if len(f3)<3:
        f3="feature3null"
    tx = BeautifulSoup(f3)

    #extracted features of each row of train data scripts
    tx1 = [x.extract() for x in tx.findAll('script')]			
    tx = tx.get_text(" ").strip()						#stripping ' '
    s = (" ").join(["z"+ str(z) for z in f2.split(" ")]) + " " + tx
    s = re.sub("[^a-zA-Z0-9]"," ", s)						#string pattern matching based on regular expressions
    s = re.sub("[0-9]{1,3}px"," ", s)						# 0-9 repeating from 1 to 3
    s = re.sub(" [0-9]{1,6} |000"," ", s)
    s = (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s = s.lower()
    return s

#==============taking input test and train data=========================
train = pd.read_csv("../input/train.csv").fillna("")                                     #skipping non-Null missing data in train and test
test  = pd.read_csv("../input/test.csv").fillna("")

idx = test.id.values.astype(int)      	                                        #taking id values in idx of test data

train.drop('id',axis=1)   				                        #dropping id column in test n train
test.drop('id',axis=1)

y = train.median_relevance.values		                                #storing median_variance column values to y
train.drop(['median_relevance','relevance_variance'],axis=1)                    #deleting both columns from train data

#============filtering======================
traindata = np.array(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))   #filtering
testdata = np.array(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))

#============cleaning=======================
for i in range(len(traindata)):
    traindata[i]=clean(traindata[i])		#calling clean function for cleaning data
for i in range(len(testdata)):
    testdata[i]=clean(testdata[i])

#=========dimensionality reduction==========
tsvd = TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0) 		
scl = StandardScaler()					#Standardize features by removing the mean and scaling to unit variance

#===========Converting a collection of raw documents to a matrix of TF-IDF features.============
tfidf = TfidfVectorizer(max_features=2500,min_df=0.0,max_df=0.4,stop_words=ML_STOP_WORDS,ngram_range=(1,3),use_idf=True,smooth_idf=True,sublinear_tf=True,analyzer='word')		#Convert a collection of raw documents to a matrix of TF-IDF features.

#===========Support Vector Classification....class============
classifier = SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
parameters = { 'class__n_estimators': (10,50,100) }			#A random forest classifier.

#==============Pipeline of transforms with a final estimator.==============
pipe = Pipeline([('tfidf',tfidf),
                 ('tsvd',tsvd),
                 ('scl',scl),
                 ('class',classifier)])

#=============creating relevance feature function based on traindata attributes and median_variance(y)======
pipe.fit(traindata,y)

#============predicting median variance of test data==============
predicted = pipe.predict(testdata) 

df = pd.DataFrame()			#=====creating output frame file====== 

df['id']=idx
df['prediction']=predicted


df.to_csv('search_result_outp.csv',index=False)		#====creating csv file from output frame======

print(df.prediction.value_counts())

