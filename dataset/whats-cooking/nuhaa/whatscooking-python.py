
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re

from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("../input/train.json")
traindf['ingredients_clean_string'] = [', '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       
print(traindf.tail())
print("----------")
testdf = pd.read_json("../input/test.json") 
testdf['ingredients_clean_string'] = [', '.join(z).strip() for z in testdf['ingredients']]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       
print(testdf.head())
print("first record in test data: "+testdf['ingredients_string'][0])
print('')

def top_features(vectorizer):
	indices = np.argsort(vectorizer.idf_)[::-1]
	features = vectorizer.get_feature_names()
	top_features = [features[i] for i in indices[:20]]
	return top_features

# convert to document-term matrix
corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
print(top_features(vectorizertr))
print(tfidftr.shape) # size - rows x columns
print(tfidftr)
print(type(tfidftr))

corpusts = testdf['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts)
print(tfidfts)
print(type(tfidfts))
#exit()

#classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[1, 10]}
#clf = LinearSVC()
clf = LogisticRegression()

classifier = grid_search.GridSearchCV(clf, parameters)

# train model
print("trining model...")
predictors_tr = tfidftr
targets_tr = traindf['cuisine']
classifier=classifier.fit(predictors_tr,targets_tr)

# make predictions
print("making predictions...")
predictors_ts = tfidfts
predictions=classifier.predict(predictors_ts)
print(predictions)


# write output
testdf['cuisine'] = predictions
testdf = testdf.sort_values(by='id' , ascending=True)
testdf[['id' , 'ingredients_clean_string' , 'cuisine' ]].to_csv("submission.csv", encoding='utf-8')

