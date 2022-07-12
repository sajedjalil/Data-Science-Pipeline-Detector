from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re,sys
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("../input/train.json")
#traindf = traindf.iloc[:1000,:]
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       

testdf = pd.read_json("../input/test.json") 
#testdf = testdf.iloc[:1000,:]
testdf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testdf['ingredients']]       
from sklearn.preprocessing import LabelEncoder

corpustr = traindf['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 2 ),analyzer="word", 
                             max_df = .7 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr=vectorizertr.fit_transform(corpustr).todense()
corpusts = testdf['ingredients_string']
#vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts=vectorizertr.transform(corpusts).todense()

#hvectorizer = TfidfVectorizer(stop_words='english',ngram_range = ( 1 , 2 ),sublinear_tf=True, max_df=0.7,
#                                 stop_words='english')
#htr=vectorizertr.fit_transform(corpustr).todense()
#hts = TfidfVectorizer(stop_words='english')
#tfidfts=vectorizertr.transform(corpusts).todense()

x_train = np.array(tfidftr)
print(x_train.shape)
targets_tr = traindf['cuisine']
le = LabelEncoder()
y_train = le.fit_transform(targets_tr) 
print(y_train)
#sys.exit()
x_test = np.array(tfidfts)
print(type(x_test))
ch2 = SelectKBest(chi2, k=1000)
#sys.exit()
x_train = ch2.fit_transform(x_train, y_train)
x_test  = ch2.transform(x_test)
print(x_train.shape)
#sys.exit()
#### xgb ####
# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
# gbm = XGBClassifier(n_estimators=100).fit(x_train, y_train)
# predictions = gbm.predict(x_test)
#sumission
# print(predictions)
# print(le.inverse_transform(predictions))
# submission = pd.DataFrame({ 'id': testdf['id'],
#                             'cuisine': le.inverse_transform(predictions)})
# submission.to_csv("submission.csv", index=False)
#### linear SVC ####
# classifier = LinearSVC(C=0.80, penalty="l2", dual=False)
parameters = {'C':[0.01,0.1,1, 10]}
clf = LinearSVC(penalty="l1", dual=False)
# #clf = LogisticRegression()

classifier = grid_search.GridSearchCV(clf, parameters)
classifier=classifier.fit(x_train,y_train)

predictions=classifier.predict(x_test)
submission = pd.DataFrame({ 'id': testdf['id'],
                             'cuisine': le.inverse_transform(predictions)})
submission.to_csv("submission.csv", index=False)